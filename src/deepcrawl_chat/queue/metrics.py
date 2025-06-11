import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque

from src.deepcrawl_chat.core.redis import RedisPool
from src.deepcrawl_chat.core.utils import DeepCrawlError, ErrorCode
from src.deepcrawl_chat.queue.schemas import TaskStatus

logger = logging.getLogger(__name__)


@dataclass
class QueueMetrics:
    """Metrics for a specific queue."""
    queue_name: str
    pending_count: int = 0
    processing_count: int = 0
    failed_count: int = 0
    completed_count: int = 0
    cancelled_count: int = 0

    # Throughput metrics
    tasks_enqueued_per_minute: float = 0.0
    tasks_processed_per_minute: float = 0.0

    # Timing metrics
    avg_processing_time_seconds: float = 0.0
    avg_wait_time_seconds: float = 0.0

    # Error metrics
    error_rate_percentage: float = 0.0
    retry_rate_percentage: float = 0.0

    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class WorkerMetrics:
    """Metrics for individual workers."""
    worker_id: str
    worker_type: str

    # Load metrics
    current_load: int = 0
    max_capacity: int = 10
    load_percentage: float = 0.0

    # Performance metrics
    tasks_completed: int = 0
    tasks_failed: int = 0
    avg_task_duration_seconds: float = 0.0

    # Availability metrics
    uptime_percentage: float = 100.0
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True

    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SystemMetrics:
    """Overall system performance metrics."""
    total_workers: int = 0
    active_workers: int = 0
    total_queues: int = 0

    # System-wide task metrics
    total_tasks_pending: int = 0
    total_tasks_processing: int = 0
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0

    # System performance
    overall_throughput_per_minute: float = 0.0
    system_load_percentage: float = 0.0
    avg_response_time_seconds: float = 0.0

    # Resource utilization
    redis_memory_usage_mb: float = 0.0
    redis_connections: int = 0

    timestamp: datetime = field(default_factory=datetime.utcnow)


class MetricsCollector:
    """Collects and aggregates metrics for the queue system."""

    def __init__(self, redis_pool: RedisPool, collection_interval_seconds: int = 60):
        """
        Initialize the metrics collector.

        Args:
            redis_pool: Redis connection pool
            collection_interval_seconds: How often to collect metrics
        """
        self.redis = redis_pool.pool
        self.collection_interval = collection_interval_seconds

        # In-memory metrics storage for recent data
        self.queue_metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.worker_metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.system_metrics_history: deque = deque(maxlen=100)

        # Performance tracking
        self.task_timing_samples: Dict[str, List[float]] = defaultdict(list)
        self.throughput_counters: Dict[str, List[tuple[datetime, int]]] = defaultdict(list)

        # Redis key patterns for metrics
        self.METRICS_PREFIX = "metrics:"
        self.QUEUE_METRICS_KEY = f"{self.METRICS_PREFIX}queue:{{queue_name}}"
        self.WORKER_METRICS_KEY = f"{self.METRICS_PREFIX}worker:{{worker_id}}"
        self.SYSTEM_METRICS_KEY = f"{self.METRICS_PREFIX}system"
        self.TIMING_SAMPLES_KEY = f"{self.METRICS_PREFIX}timing:{{queue_name}}"

    async def collect_queue_metrics(self, queue_name: str, queue_manager) -> QueueMetrics:
        """
        Collect metrics for a specific queue.

        Args:
            queue_name: Name of the queue to collect metrics for
            queue_manager: QueueManager instance to get queue data

        Returns:
            QueueMetrics object with current metrics
        """
        try:
            # Get basic queue sizes
            queue_sizes = await queue_manager.get_all_queue_sizes(queue_name)

            metrics = QueueMetrics(
                queue_name=queue_name,
                pending_count=queue_sizes["pending"],
                processing_count=queue_sizes["processing"],
                failed_count=queue_sizes["failed"]
            )

            # Get status-based counts if status tracker is available
            if hasattr(queue_manager, 'status_tracker') and queue_manager.status_tracker:
                metrics.completed_count = len(await queue_manager.status_tracker.get_tasks_by_status(TaskStatus.COMPLETED, limit=1000))
                metrics.cancelled_count = len(await queue_manager.status_tracker.get_tasks_by_status(TaskStatus.CANCELLED, limit=1000))

            # Calculate throughput metrics
            metrics.tasks_enqueued_per_minute = await self._calculate_throughput(queue_name, "enqueued")
            metrics.tasks_processed_per_minute = await self._calculate_throughput(queue_name, "processed")

            # Calculate timing metrics
            timing_data = await self._get_timing_metrics(queue_name)
            metrics.avg_processing_time_seconds = timing_data.get("avg_processing_time", 0.0)
            metrics.avg_wait_time_seconds = timing_data.get("avg_wait_time", 0.0)

            # Calculate error rates
            total_tasks = metrics.completed_count + metrics.failed_count
            if total_tasks > 0:
                metrics.error_rate_percentage = (metrics.failed_count / total_tasks) * 100.0

            # Store metrics in Redis and local history
            await self._store_queue_metrics(metrics)
            self.queue_metrics_history[queue_name].append(metrics)

            return metrics

        except Exception as e:
            logger.error(f"Error collecting queue metrics for {queue_name}: {e}", exc_info=True)
            raise DeepCrawlError(f"Failed to collect queue metrics: {e}", ErrorCode.QUEUE_ERROR)

    async def collect_worker_metrics(self, worker_id: str, worker_info) -> WorkerMetrics:
        """
        Collect metrics for a specific worker.

        Args:
            worker_id: ID of the worker
            worker_info: WorkerInfo object with current worker data

        Returns:
            WorkerMetrics object with current metrics
        """
        try:
            metrics = WorkerMetrics(
                worker_id=worker_id,
                worker_type=worker_info.worker_type,
                current_load=worker_info.current_load,
                max_capacity=worker_info.max_capacity,
                load_percentage=worker_info.load_percentage,
                is_active=worker_info.is_active,
                last_heartbeat=datetime.fromtimestamp(worker_info.last_heartbeat)
            )

            # Get performance metrics from Redis
            performance_data = await self._get_worker_performance_data(worker_id)
            metrics.tasks_completed = performance_data.get("tasks_completed", 0)
            metrics.tasks_failed = performance_data.get("tasks_failed", 0)
            metrics.avg_task_duration_seconds = performance_data.get("avg_duration", 0.0)

            # Calculate uptime percentage
            current_time = datetime.utcnow()
            uptime_window = timedelta(hours=24)  # Look at last 24 hours
            metrics.uptime_percentage = await self._calculate_worker_uptime(worker_id, uptime_window)

            # Store metrics
            await self._store_worker_metrics(metrics)
            self.worker_metrics_history[worker_id].append(metrics)

            return metrics

        except Exception as e:
            logger.error(f"Error collecting worker metrics for {worker_id}: {e}", exc_info=True)
            raise DeepCrawlError(f"Failed to collect worker metrics: {e}", ErrorCode.QUEUE_ERROR)

    async def collect_system_metrics(self, queue_manager, task_distributor) -> SystemMetrics:
        """
        Collect system-wide metrics.

        Args:
            queue_manager: QueueManager instance
            task_distributor: TaskDistributor instance

        Returns:
            SystemMetrics object with current system metrics
        """
        try:
            metrics = SystemMetrics()

            # Worker metrics
            worker_stats = await task_distributor.get_worker_stats()
            metrics.total_workers = len(worker_stats)
            metrics.active_workers = sum(1 for stats in worker_stats.values() if stats["is_active"])

            # Calculate system load
            if metrics.total_workers > 0:
                total_capacity = sum(stats["max_capacity"] for stats in worker_stats.values())
                total_load = sum(stats["current_load"] for stats in worker_stats.values())
                metrics.system_load_percentage = (total_load / total_capacity) * 100.0 if total_capacity > 0 else 0.0

            # Get Redis metrics
            redis_info = await self._get_redis_metrics()
            metrics.redis_memory_usage_mb = redis_info.get("memory_usage_mb", 0.0)
            metrics.redis_connections = redis_info.get("connections", 0)

            # Calculate overall throughput
            metrics.overall_throughput_per_minute = await self._calculate_system_throughput()

            # Store metrics
            await self._store_system_metrics(metrics)
            self.system_metrics_history.append(metrics)

            return metrics

        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}", exc_info=True)
            raise DeepCrawlError(f"Failed to collect system metrics: {e}", ErrorCode.QUEUE_ERROR)

    async def record_task_timing(self, queue_name: str, task_id: str, event_type: str, timestamp: Optional[datetime] = None) -> None:
        """
        Record timing information for task performance analysis.

        Args:
            queue_name: Name of the queue
            task_id: ID of the task
            event_type: Type of event ('enqueued', 'started', 'completed', 'failed')
            timestamp: Timestamp of the event (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        timing_key = f"timing:{queue_name}:{task_id}"

        try:
            await self.redis.hset(
                timing_key,
                event_type,
                timestamp.isoformat()
            )

            # Set expiration for timing data (24 hours)
            await self.redis.expire(timing_key, 86400)

        except Exception as e:
            logger.error(f"Error recording task timing: {e}", exc_info=True)

    async def record_worker_performance(self, worker_id: str, task_duration: float, success: bool) -> None:
        """
        Record worker performance data.

        Args:
            worker_id: ID of the worker
            task_duration: Duration of the task in seconds
            success: Whether the task was successful
        """
        try:
            pipe = self.redis.pipeline()

            # Update counters
            if success:
                pipe.hincrby(f"worker_perf:{worker_id}", "tasks_completed", 1)
            else:
                pipe.hincrby(f"worker_perf:{worker_id}", "tasks_failed", 1)

            # Update average duration (using exponential moving average)
            current_avg_key = f"worker_perf:{worker_id}:avg_duration"
            alpha = 0.1  # Smoothing factor

            # Get current average
            current_avg = await self.redis.get(current_avg_key)
            if current_avg:
                new_avg = (1 - alpha) * float(current_avg) + alpha * task_duration
            else:
                new_avg = task_duration

            pipe.set(current_avg_key, new_avg)

            # Set expiration (7 days)
            pipe.expire(f"worker_perf:{worker_id}", 604800)
            pipe.expire(current_avg_key, 604800)

            await pipe.execute()

        except Exception as e:
            logger.error(f"Error recording worker performance: {e}", exc_info=True)

    async def get_metrics_summary(self, time_range_hours: int = 1) -> Dict[str, Any]:
        """
        Get a summary of metrics for the specified time range.

        Args:
            time_range_hours: Number of hours to look back

        Returns:
            Dictionary with metrics summary
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=time_range_hours)

        summary = {
            "time_range_hours": time_range_hours,
            "generated_at": datetime.utcnow().isoformat(),
            "queues": {},
            "workers": {},
            "system": {}
        }

        try:
            # Aggregate queue metrics
            for queue_name, metrics_list in self.queue_metrics_history.items():
                recent_metrics = [m for m in metrics_list if m.timestamp >= cutoff_time]
                if recent_metrics:
                    summary["queues"][queue_name] = self._aggregate_queue_metrics(recent_metrics)

            # Aggregate worker metrics
            for worker_id, metrics_list in self.worker_metrics_history.items():
                recent_metrics = [m for m in metrics_list if m.timestamp >= cutoff_time]
                if recent_metrics:
                    summary["workers"][worker_id] = self._aggregate_worker_metrics(recent_metrics)

            # Aggregate system metrics
            recent_system_metrics = [m for m in self.system_metrics_history if m.timestamp >= cutoff_time]
            if recent_system_metrics:
                summary["system"] = self._aggregate_system_metrics(recent_system_metrics)

            return summary

        except Exception as e:
            logger.error(f"Error generating metrics summary: {e}", exc_info=True)
            raise DeepCrawlError(f"Failed to generate metrics summary: {e}", ErrorCode.QUEUE_ERROR)

    # Private helper methods

    async def _calculate_throughput(self, queue_name: str, event_type: str) -> float:
        """Calculate throughput for a queue and event type."""
        try:
            # Get count of events in the last minute
            one_minute_ago = datetime.utcnow() - timedelta(minutes=1)

            counter_key = f"throughput:{queue_name}:{event_type}"

            # This is a simplified calculation - in a real implementation,
            # you might want to use Redis sorted sets or time series data
            recent_count = await self.redis.get(counter_key) or 0
            return float(recent_count)

        except Exception:
            return 0.0

    async def _get_timing_metrics(self, queue_name: str) -> Dict[str, float]:
        """Get timing metrics for a queue."""
        try:
            # This would analyze timing data from Redis
            # For now, return default values
            return {
                "avg_processing_time": 0.0,
                "avg_wait_time": 0.0
            }
        except Exception:
            return {"avg_processing_time": 0.0, "avg_wait_time": 0.0}

    async def _get_worker_performance_data(self, worker_id: str) -> Dict[str, Any]:
        """Get performance data for a worker from Redis."""
        try:
            data = await self.redis.hgetall(f"worker_perf:{worker_id}")
            if not data:
                return {}

            # Decode if bytes
            if isinstance(list(data.keys())[0], bytes):
                data = {k.decode(): v.decode() for k, v in data.items()}

            return {
                "tasks_completed": int(data.get("tasks_completed", 0)),
                "tasks_failed": int(data.get("tasks_failed", 0)),
                "avg_duration": float(data.get("avg_duration", 0.0))
            }
        except Exception:
            return {}

    async def _calculate_worker_uptime(self, worker_id: str, time_window: timedelta) -> float:
        """Calculate worker uptime percentage over a time window."""
        # This would analyze worker heartbeat data
        # For now, return a default value
        return 95.0  # Assume 95% uptime

    async def _get_redis_metrics(self) -> Dict[str, Any]:
        """Get Redis server metrics."""
        try:
            info = await self.redis.info()
            memory_usage_bytes = info.get("used_memory", 0)
            memory_usage_mb = memory_usage_bytes / (1024 * 1024)

            return {
                "memory_usage_mb": memory_usage_mb,
                "connections": info.get("connected_clients", 0)
            }
        except Exception:
            return {"memory_usage_mb": 0.0, "connections": 0}

    async def _calculate_system_throughput(self) -> float:
        """Calculate overall system throughput."""
        # This would aggregate throughput across all queues
        return 0.0  # Placeholder

    async def _store_queue_metrics(self, metrics: QueueMetrics) -> None:
        """Store queue metrics in Redis."""
        try:
            key = self.QUEUE_METRICS_KEY.format(queue_name=metrics.queue_name)
            data = {
                "pending_count": metrics.pending_count,
                "processing_count": metrics.processing_count,
                "failed_count": metrics.failed_count,
                "completed_count": metrics.completed_count,
                "throughput_per_minute": metrics.tasks_processed_per_minute,
                "error_rate": metrics.error_rate_percentage,
                "timestamp": metrics.timestamp.isoformat()
            }

            await self.redis.hset(key, mapping=data)
            await self.redis.expire(key, 86400)  # 24 hours

        except Exception as e:
            logger.error(f"Error storing queue metrics: {e}")

    async def _store_worker_metrics(self, metrics: WorkerMetrics) -> None:
        """Store worker metrics in Redis."""
        try:
            key = self.WORKER_METRICS_KEY.format(worker_id=metrics.worker_id)
            data = {
                "current_load": metrics.current_load,
                "load_percentage": metrics.load_percentage,
                "tasks_completed": metrics.tasks_completed,
                "uptime_percentage": metrics.uptime_percentage,
                "timestamp": metrics.timestamp.isoformat()
            }

            await self.redis.hset(key, mapping=data)
            await self.redis.expire(key, 86400)  # 24 hours

        except Exception as e:
            logger.error(f"Error storing worker metrics: {e}")

    async def _store_system_metrics(self, metrics: SystemMetrics) -> None:
        """Store system metrics in Redis."""
        try:
            data = {
                "total_workers": metrics.total_workers,
                "active_workers": metrics.active_workers,
                "system_load_percentage": metrics.system_load_percentage,
                "overall_throughput": metrics.overall_throughput_per_minute,
                "timestamp": metrics.timestamp.isoformat()
            }

            await self.redis.hset(self.SYSTEM_METRICS_KEY, mapping=data)
            await self.redis.expire(self.SYSTEM_METRICS_KEY, 86400)  # 24 hours

        except Exception as e:
            logger.error(f"Error storing system metrics: {e}")

    def _aggregate_queue_metrics(self, metrics_list: List[QueueMetrics]) -> Dict[str, Any]:
        """Aggregate queue metrics over time."""
        if not metrics_list:
            return {}

        return {
            "avg_pending": sum(m.pending_count for m in metrics_list) / len(metrics_list),
            "avg_processing": sum(m.processing_count for m in metrics_list) / len(metrics_list),
            "avg_throughput": sum(m.tasks_processed_per_minute for m in metrics_list) / len(metrics_list),
            "avg_error_rate": sum(m.error_rate_percentage for m in metrics_list) / len(metrics_list),
            "peak_pending": max(m.pending_count for m in metrics_list),
            "samples": len(metrics_list)
        }

    def _aggregate_worker_metrics(self, metrics_list: List[WorkerMetrics]) -> Dict[str, Any]:
        """Aggregate worker metrics over time."""
        if not metrics_list:
            return {}

        return {
            "avg_load_percentage": sum(m.load_percentage for m in metrics_list) / len(metrics_list),
            "total_completed": metrics_list[-1].tasks_completed,  # Latest count
            "avg_uptime": sum(m.uptime_percentage for m in metrics_list) / len(metrics_list),
            "samples": len(metrics_list)
        }

    def _aggregate_system_metrics(self, metrics_list: List[SystemMetrics]) -> Dict[str, Any]:
        """Aggregate system metrics over time."""
        if not metrics_list:
            return {}

        return {
            "avg_workers": sum(m.active_workers for m in metrics_list) / len(metrics_list),
            "avg_system_load": sum(m.system_load_percentage for m in metrics_list) / len(metrics_list),
            "avg_throughput": sum(m.overall_throughput_per_minute for m in metrics_list) / len(metrics_list),
            "peak_system_load": max(m.system_load_percentage for m in metrics_list),
            "samples": len(metrics_list)
        }