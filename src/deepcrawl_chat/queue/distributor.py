import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from enum import Enum

from src.deepcrawl_chat.core.redis import RedisPool
from src.deepcrawl_chat.core.utils import DeepCrawlError, ErrorCode
from src.deepcrawl_chat.queue.manager import QueueManager
from src.deepcrawl_chat.queue.schemas import Task, TaskPriority, TaskType

logger = logging.getLogger(__name__)

class DistributionStrategy(str, Enum):
    ROUND_ROBIN = "ROUND_ROBIN"
    LEAST_LOADED = "LEAST_LOADED"
    PRIORITY_FIRST = "PRIORITY_FIRST"
    WORKER_TYPE_BASED = "WORKER_TYPE_BASED"

@dataclass
class WorkerInfo:
    worker_id: str
    worker_type: str  # e.g., "crawler", "processor", "embedder"
    supported_task_types: Set[str]
    current_load: int = 0
    max_capacity: int = 10
    last_heartbeat: float = field(default_factory=time.time)
    is_active: bool = True

    @property
    def load_percentage(self) -> float:
        """Returns current load as percentage of max capacity."""
        if self.max_capacity == 0:
            return 100.0
        return (self.current_load / self.max_capacity) * 100.0

    @property
    def is_available(self) -> bool:
        """Returns True if worker can accept more tasks."""
        return self.is_active and self.current_load < self.max_capacity

    def update_heartbeat(self) -> None:
        """Updates the last heartbeat timestamp."""
        self.last_heartbeat = time.time()

class BaseDistributionStrategy(ABC):
    """Abstract base class for task distribution strategies."""

    @abstractmethod
    async def select_worker(self, task: Task, available_workers: List[WorkerInfo]) -> Optional[WorkerInfo]:
        """Select the best worker for the given task."""
        pass

class RoundRobinStrategy(BaseDistributionStrategy):
    """Round-robin distribution strategy."""

    def __init__(self):
        self._last_selected_index = -1

    async def select_worker(self, task: Task, available_workers: List[WorkerInfo]) -> Optional[WorkerInfo]:
        if not available_workers:
            return None

        # Filter workers that can handle this task type
        capable_workers = [w for w in available_workers if task.task_type in w.supported_task_types]
        if not capable_workers:
            return None

        # Select next worker in round-robin fashion
        self._last_selected_index = (self._last_selected_index + 1) % len(capable_workers)
        return capable_workers[self._last_selected_index]

class LeastLoadedStrategy(BaseDistributionStrategy):
    """Least loaded distribution strategy."""

    async def select_worker(self, task: Task, available_workers: List[WorkerInfo]) -> Optional[WorkerInfo]:
        if not available_workers:
            return None

        # Filter workers that can handle this task type
        capable_workers = [w for w in available_workers if task.task_type in w.supported_task_types]
        if not capable_workers:
            return None

        # Select worker with lowest load percentage
        return min(capable_workers, key=lambda w: w.load_percentage)

class PriorityFirstStrategy(BaseDistributionStrategy):
    """Priority-first distribution strategy with load balancing for same priority."""

    async def select_worker(self, task: Task, available_workers: List[WorkerInfo]) -> Optional[WorkerInfo]:
        if not available_workers:
            return None

        # Filter workers that can handle this task type
        capable_workers = [w for w in available_workers if task.task_type in w.supported_task_types]
        if not capable_workers:
            return None

        # For high/urgent priority tasks, select least loaded worker
        if task.priority in [TaskPriority.HIGH, TaskPriority.URGENT]:
            return min(capable_workers, key=lambda w: w.load_percentage)

        # For normal/low priority, use round-robin among available workers
        return min(capable_workers, key=lambda w: w.load_percentage)

class WorkerTypeBasedStrategy(BaseDistributionStrategy):
    """Worker type-based distribution strategy."""

    def __init__(self):
        # Define preferred worker types for specific task types
        self.task_to_worker_type: Dict[str, str] = {
            TaskType.CRAWL_URL: "crawler",
            TaskType.PROCESS_CONTENT: "processor",
            # Add more mappings as needed
        }

    async def select_worker(self, task: Task, available_workers: List[WorkerInfo]) -> Optional[WorkerInfo]:
        if not available_workers:
            return None

        preferred_worker_type = self.task_to_worker_type.get(task.task_type)

        # First, try to find workers of the preferred type
        if preferred_worker_type:
            preferred_workers = [
                w for w in available_workers
                if w.worker_type == preferred_worker_type and task.task_type in w.supported_task_types
            ]
            if preferred_workers:
                return min(preferred_workers, key=lambda w: w.load_percentage)

        # Fall back to any capable worker
        capable_workers = [w for w in available_workers if task.task_type in w.supported_task_types]
        if capable_workers:
            return min(capable_workers, key=lambda w: w.load_percentage)

        return None

class TaskDistributor:
    """Manages task distribution to available workers."""

    WORKER_TIMEOUT_SECONDS = 60  # Workers are considered inactive after 60 seconds

    def __init__(self, redis_pool: RedisPool, queue_manager: QueueManager):
        self.redis = redis_pool.pool
        self.queue_manager = queue_manager
        self.workers: Dict[str, WorkerInfo] = {}
        self.distribution_strategies: Dict[DistributionStrategy, BaseDistributionStrategy] = {
            DistributionStrategy.ROUND_ROBIN: RoundRobinStrategy(),
            DistributionStrategy.LEAST_LOADED: LeastLoadedStrategy(),
            DistributionStrategy.PRIORITY_FIRST: PriorityFirstStrategy(),
            DistributionStrategy.WORKER_TYPE_BASED: WorkerTypeBasedStrategy(),
        }
        self.current_strategy = DistributionStrategy.PRIORITY_FIRST
        self._cleanup_task: Optional[asyncio.Task] = None

    async def register_worker(
        self,
        worker_id: str,
        worker_type: str,
        supported_task_types: Set[str],
        max_capacity: int = 10
    ) -> None:
        """Register a new worker or update existing worker info."""
        worker_info = WorkerInfo(
            worker_id=worker_id,
            worker_type=worker_type,
            supported_task_types=supported_task_types,
            max_capacity=max_capacity
        )
        self.workers[worker_id] = worker_info

        # Store worker info in Redis for persistence across distributor instances
        await self._store_worker_info(worker_info)

        logger.info(f"Registered worker {worker_id} of type {worker_type} with capacity {max_capacity}")

    async def unregister_worker(self, worker_id: str) -> None:
        """Unregister a worker."""
        if worker_id in self.workers:
            del self.workers[worker_id]
            await self.redis.hdel("workers", worker_id)
            logger.info(f"Unregistered worker {worker_id}")

    async def update_worker_heartbeat(self, worker_id: str) -> None:
        """Update worker heartbeat to indicate it's still active."""
        if worker_id in self.workers:
            self.workers[worker_id].update_heartbeat()
            # Update heartbeat timestamp in Redis
            await self.redis.hset(f"worker:{worker_id}", "last_heartbeat", time.time())

    async def update_worker_load(self, worker_id: str, current_load: int) -> None:
        """Update worker's current load."""
        if worker_id in self.workers:
            self.workers[worker_id].current_load = max(0, current_load)
            # Update load in Redis
            await self.redis.hset(f"worker:{worker_id}", "current_load", current_load)

    async def set_distribution_strategy(self, strategy: DistributionStrategy) -> None:
        """Set the task distribution strategy."""
        if strategy not in self.distribution_strategies:
            raise DeepCrawlError(f"Unknown distribution strategy: {strategy}", ErrorCode.QUEUE_ERROR)

        self.current_strategy = strategy
        logger.info(f"Distribution strategy set to {strategy}")

    async def distribute_task(self, queue_name: str, timeout: int = 5) -> Optional[tuple[Task, WorkerInfo]]:
        """
        Distribute a task from the queue to an available worker.

        Returns:
            Tuple of (task, worker_info) if successful, None if no task or no available worker.
        """
        # Get available workers first to know if we have any
        available_workers = await self._get_available_workers()
        if not available_workers:
            logger.warning(f"No workers available for queue {queue_name}")
            return None

        # Try to get a task from the queue - pass None for worker_id since we don't know yet
        task = await self.queue_manager.dequeue_task(queue_name, timeout)
        if not task:
            return None

        # Select worker using current strategy
        strategy = self.distribution_strategies[self.current_strategy]
        selected_worker = await strategy.select_worker(task, available_workers)

        if not selected_worker:
            # No suitable worker found, put task back in queue
            await self.queue_manager.fail_task(queue_name, task, f"No suitable worker found for task type {task.task_type}")
            logger.warning(f"No suitable worker found for task {task.task_id} of type {task.task_type}")
            return None

        # Update worker load
        selected_worker.current_load += 1
        await self.update_worker_load(selected_worker.worker_id, selected_worker.current_load)

        # Now that we've assigned the task to a worker, update the status tracker
        if hasattr(self.queue_manager, 'status_tracker') and self.queue_manager.status_tracker:
            from src.deepcrawl_chat.queue.schemas import TaskStatus
            await self.queue_manager.status_tracker.track_task_status(
                task,
                TaskStatus.ASSIGNED,
                f"Task assigned to worker {selected_worker.worker_id}",
                worker_id=selected_worker.worker_id
            )

        logger.info(
            f"Distributed task {task.task_id} ({task.task_type}) to worker {selected_worker.worker_id} "
            f"(load: {selected_worker.current_load}/{selected_worker.max_capacity})"
        )

        return task, selected_worker

    async def complete_task(self, worker_id: str, task: Task, queue_name: str, success: bool = True, error_message: str = None) -> None:
        """
        Mark a task as completed by a worker.

        Args:
            worker_id: ID of the worker that completed the task
            task: The completed task
            queue_name: Queue name for task acknowledgment/failure
            success: Whether task completed successfully
            error_message: Error message if task failed
        """
        # Update worker load
        if worker_id in self.workers:
            self.workers[worker_id].current_load = max(0, self.workers[worker_id].current_load - 1)
            await self.update_worker_load(worker_id, self.workers[worker_id].current_load)

        # Handle task completion - pass worker_id for status tracking
        if success:
            await self.queue_manager.ack_task(queue_name, task, worker_id=worker_id)
            logger.info(f"Task {task.task_id} completed successfully by worker {worker_id}")
        else:
            await self.queue_manager.fail_task(queue_name, task, error_message or "Task failed", worker_id=worker_id)
            logger.warning(f"Task {task.task_id} failed on worker {worker_id}: {error_message}")

    async def get_worker_stats(self) -> Dict[str, Dict]:
        """Get statistics for all registered workers."""
        stats = {}
        for worker_id, worker in self.workers.items():
            stats[worker_id] = {
                "worker_type": worker.worker_type,
                "supported_task_types": list(worker.supported_task_types),
                "current_load": worker.current_load,
                "max_capacity": worker.max_capacity,
                "load_percentage": worker.load_percentage,
                "is_available": worker.is_available,
                "is_active": worker.is_active,
                "last_heartbeat": worker.last_heartbeat,
            }
        return stats

    async def start_worker_cleanup(self) -> None:
        """Start the worker cleanup task to remove inactive workers."""
        if self._cleanup_task and not self._cleanup_task.done():
            return  # Already running

        self._cleanup_task = asyncio.create_task(self._worker_cleanup_loop())
        logger.info("Started worker cleanup task")

    async def stop_worker_cleanup(self) -> None:
        """Stop the worker cleanup task."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped worker cleanup task")

    async def _get_available_workers(self) -> List[WorkerInfo]:
        """Get list of available workers."""
        current_time = time.time()
        available_workers = []

        for worker in self.workers.values():
            # Check if worker is active based on heartbeat
            if current_time - worker.last_heartbeat > self.WORKER_TIMEOUT_SECONDS:
                worker.is_active = False

            if worker.is_available:
                available_workers.append(worker)

        return available_workers

    async def _store_worker_info(self, worker: WorkerInfo) -> None:
        """Store worker information in Redis."""
        worker_key = f"worker:{worker.worker_id}"
        worker_data = {
            "worker_type": worker.worker_type,
            "supported_task_types": ",".join(worker.supported_task_types),
            "current_load": worker.current_load,
            "max_capacity": worker.max_capacity,
            "last_heartbeat": worker.last_heartbeat,
            "is_active": str(worker.is_active).lower(),
        }
        await self.redis.hset(worker_key, mapping=worker_data)

    async def _worker_cleanup_loop(self) -> None:
        """Background task to clean up inactive workers."""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                await self._cleanup_inactive_workers()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in worker cleanup loop: {e}", exc_info=True)

    async def _cleanup_inactive_workers(self) -> None:
        """Remove workers that haven't sent heartbeat recently."""
        current_time = time.time()
        inactive_workers = []

        for worker_id, worker in self.workers.items():
            if current_time - worker.last_heartbeat > self.WORKER_TIMEOUT_SECONDS:
                inactive_workers.append(worker_id)

        for worker_id in inactive_workers:
            await self.unregister_worker(worker_id)
            logger.info(f"Removed inactive worker {worker_id}")