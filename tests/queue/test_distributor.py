import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

from src.deepcrawl_chat.core.redis import RedisPool
from src.deepcrawl_chat.queue.manager import QueueManager
from src.deepcrawl_chat.queue.distributor import (
    TaskDistributor, WorkerInfo, DistributionStrategy,
    RoundRobinStrategy, LeastLoadedStrategy, PriorityFirstStrategy, WorkerTypeBasedStrategy
)
from src.deepcrawl_chat.queue.schemas import Task, CrawlURLPayload, ProcessContentPayload, TaskType, TaskPriority
from src.deepcrawl_chat.core.utils import DeepCrawlError

@pytest.fixture
async def redis_pool():
    """Mock Redis pool for testing."""
    pool = MagicMock()
    pool.pool = AsyncMock()
    pool.pool.hset = AsyncMock()
    pool.pool.hdel = AsyncMock()
    return pool

@pytest.fixture
async def queue_manager():
    """Mock QueueManager for testing."""
    qm = AsyncMock()
    qm.dequeue_task = AsyncMock()
    qm.ack_task = AsyncMock()
    qm.fail_task = AsyncMock()
    return qm

@pytest.fixture
async def task_distributor(redis_pool, queue_manager):
    """Create TaskDistributor instance for testing."""
    distributor = TaskDistributor(redis_pool, queue_manager)
    yield distributor
    # Cleanup
    await distributor.stop_worker_cleanup()

# --- WorkerInfo Tests ---

def test_worker_info_creation():
    worker = WorkerInfo(
        worker_id="worker1",
        worker_type="crawler",
        supported_task_types={"CRAWL_URL", "PROCESS_CONTENT"}
    )
    assert worker.worker_id == "worker1"
    assert worker.worker_type == "crawler"
    assert worker.supported_task_types == {"CRAWL_URL", "PROCESS_CONTENT"}
    assert worker.current_load == 0
    assert worker.max_capacity == 10
    assert worker.is_active is True

def test_worker_info_properties():
    worker = WorkerInfo(
        worker_id="worker1",
        worker_type="crawler",
        supported_task_types={"CRAWL_URL"},
        current_load=5,
        max_capacity=10
    )
    assert worker.load_percentage == 50.0
    assert worker.is_available is True

    # Test at max capacity
    worker.current_load = 10
    assert worker.load_percentage == 100.0
    assert worker.is_available is False

    # Test inactive worker
    worker.current_load = 5
    worker.is_active = False
    assert worker.is_available is False

def test_worker_info_heartbeat():
    worker = WorkerInfo(
        worker_id="worker1",
        worker_type="crawler",
        supported_task_types={"CRAWL_URL"}
    )
    initial_heartbeat = worker.last_heartbeat
    time.sleep(0.01)  # Small delay
    worker.update_heartbeat()
    assert worker.last_heartbeat > initial_heartbeat

# --- Distribution Strategy Tests ---

@pytest.mark.asyncio
async def test_round_robin_strategy():
    strategy = RoundRobinStrategy()

    workers = [
        WorkerInfo("worker1", "crawler", {"CRAWL_URL"}, current_load=1),
        WorkerInfo("worker2", "crawler", {"CRAWL_URL"}, current_load=2),
        WorkerInfo("worker3", "crawler", {"CRAWL_URL"}, current_load=3),
    ]

    task = Task(task_type="CRAWL_URL", payload={})

    # Test round-robin selection
    selected1 = await strategy.select_worker(task, workers)
    selected2 = await strategy.select_worker(task, workers)
    selected3 = await strategy.select_worker(task, workers)
    selected4 = await strategy.select_worker(task, workers)  # Should wrap around

    assert selected1.worker_id == "worker1"
    assert selected2.worker_id == "worker2"
    assert selected3.worker_id == "worker3"
    assert selected4.worker_id == "worker1"  # Wrapped around

@pytest.mark.asyncio
async def test_least_loaded_strategy():
    strategy = LeastLoadedStrategy()

    workers = [
        WorkerInfo("worker1", "crawler", {"CRAWL_URL"}, current_load=5, max_capacity=10),  # 50%
        WorkerInfo("worker2", "crawler", {"CRAWL_URL"}, current_load=2, max_capacity=10),  # 20%
        WorkerInfo("worker3", "crawler", {"CRAWL_URL"}, current_load=8, max_capacity=10),  # 80%
    ]

    task = Task(task_type="CRAWL_URL", payload={})

    selected = await strategy.select_worker(task, workers)
    assert selected.worker_id == "worker2"  # Lowest load

@pytest.mark.asyncio
async def test_priority_first_strategy():
    strategy = PriorityFirstStrategy()

    workers = [
        WorkerInfo("worker1", "crawler", {"CRAWL_URL"}, current_load=5, max_capacity=10),
        WorkerInfo("worker2", "crawler", {"CRAWL_URL"}, current_load=2, max_capacity=10),
    ]

    # Test high priority task - should select least loaded
    high_priority_task = Task(task_type="CRAWL_URL", payload={}, priority=TaskPriority.HIGH)
    selected = await strategy.select_worker(high_priority_task, workers)
    assert selected.worker_id == "worker2"  # Least loaded for high priority

    # Test normal priority task - should also select least loaded (current implementation)
    normal_priority_task = Task(task_type="CRAWL_URL", payload={}, priority=TaskPriority.NORMAL)
    selected = await strategy.select_worker(normal_priority_task, workers)
    assert selected.worker_id == "worker2"  # Least loaded

@pytest.mark.asyncio
async def test_worker_type_based_strategy():
    strategy = WorkerTypeBasedStrategy()

    workers = [
        WorkerInfo("crawler1", "crawler", {"CRAWL_URL"}, current_load=5),
        WorkerInfo("processor1", "processor", {"PROCESS_CONTENT"}, current_load=2),
        WorkerInfo("generic1", "generic", {"CRAWL_URL", "PROCESS_CONTENT"}, current_load=1),
    ]

    # Test crawl task - should prefer crawler worker
    crawl_task = Task(task_type="CRAWL_URL", payload={})
    selected = await strategy.select_worker(crawl_task, workers)
    assert selected.worker_id == "crawler1"  # Preferred worker type

    # Test process task - should prefer processor worker
    process_task = Task(task_type="PROCESS_CONTENT", payload={})
    selected = await strategy.select_worker(process_task, workers)
    assert selected.worker_id == "processor1"  # Preferred worker type

@pytest.mark.asyncio
async def test_strategy_no_capable_workers():
    strategy = LeastLoadedStrategy()

    workers = [
        WorkerInfo("worker1", "crawler", {"CRAWL_URL"}, current_load=1),
    ]

    # Task type not supported by any worker
    task = Task(task_type="UNSUPPORTED_TYPE", payload={})
    selected = await strategy.select_worker(task, workers)
    assert selected is None

# --- TaskDistributor Tests ---

@pytest.mark.asyncio
async def test_register_worker(task_distributor):
    await task_distributor.register_worker(
        worker_id="test_worker",
        worker_type="crawler",
        supported_task_types={"CRAWL_URL"},
        max_capacity=5
    )

    assert "test_worker" in task_distributor.workers
    worker = task_distributor.workers["test_worker"]
    assert worker.worker_id == "test_worker"
    assert worker.worker_type == "crawler"
    assert worker.supported_task_types == {"CRAWL_URL"}
    assert worker.max_capacity == 5

@pytest.mark.asyncio
async def test_unregister_worker(task_distributor):
    # First register a worker
    await task_distributor.register_worker("test_worker", "crawler", {"CRAWL_URL"})
    assert "test_worker" in task_distributor.workers

    # Then unregister it
    await task_distributor.unregister_worker("test_worker")
    assert "test_worker" not in task_distributor.workers

@pytest.mark.asyncio
async def test_update_worker_heartbeat(task_distributor):
    await task_distributor.register_worker("test_worker", "crawler", {"CRAWL_URL"})
    worker = task_distributor.workers["test_worker"]
    initial_heartbeat = worker.last_heartbeat

    time.sleep(0.01)
    await task_distributor.update_worker_heartbeat("test_worker")

    assert worker.last_heartbeat > initial_heartbeat

@pytest.mark.asyncio
async def test_update_worker_load(task_distributor):
    await task_distributor.register_worker("test_worker", "crawler", {"CRAWL_URL"})

    await task_distributor.update_worker_load("test_worker", 5)
    worker = task_distributor.workers["test_worker"]
    assert worker.current_load == 5

@pytest.mark.asyncio
async def test_set_distribution_strategy(task_distributor):
    await task_distributor.set_distribution_strategy(DistributionStrategy.ROUND_ROBIN)
    assert task_distributor.current_strategy == DistributionStrategy.ROUND_ROBIN

    # Test invalid strategy
    with pytest.raises(DeepCrawlError):
        await task_distributor.set_distribution_strategy("INVALID_STRATEGY")

@pytest.mark.asyncio
async def test_distribute_task_success(task_distributor):
    # Register a worker
    await task_distributor.register_worker("test_worker", "crawler", {"CRAWL_URL"})

    # Mock queue manager to return a task
    crawl_payload = CrawlURLPayload(url="http://example.com")
    test_task = Task(task_type="CRAWL_URL", payload=crawl_payload.model_dump())
    task_distributor.queue_manager.dequeue_task.return_value = test_task

    # Distribute task
    result = await task_distributor.distribute_task("test_queue")

    assert result is not None
    task, worker = result
    assert task.task_id == test_task.task_id
    assert worker.worker_id == "test_worker"
    assert worker.current_load == 1  # Load should be incremented

@pytest.mark.asyncio
async def test_distribute_task_no_task(task_distributor):
    # Mock queue manager to return no task
    task_distributor.queue_manager.dequeue_task.return_value = None

    result = await task_distributor.distribute_task("test_queue")
    assert result is None

@pytest.mark.asyncio
async def test_distribute_task_no_workers(task_distributor):
    # Mock queue manager to return a task
    test_task = Task(task_type="CRAWL_URL", payload={})
    task_distributor.queue_manager.dequeue_task.return_value = test_task

    # No workers registered
    result = await task_distributor.distribute_task("test_queue")
    assert result is None

    # Verify task was failed back to queue
    task_distributor.queue_manager.fail_task.assert_called_once()

@pytest.mark.asyncio
async def test_distribute_task_no_suitable_worker(task_distributor):
    # Register worker that doesn't support the task type
    await task_distributor.register_worker("test_worker", "processor", {"PROCESS_CONTENT"})

    # Mock queue manager to return a CRAWL_URL task
    test_task = Task(task_type="CRAWL_URL", payload={})
    task_distributor.queue_manager.dequeue_task.return_value = test_task

    result = await task_distributor.distribute_task("test_queue")
    assert result is None

    # Verify task was failed back to queue
    task_distributor.queue_manager.fail_task.assert_called_once()

@pytest.mark.asyncio
async def test_complete_task_success(task_distributor):
    # Register worker and set load
    await task_distributor.register_worker("test_worker", "crawler", {"CRAWL_URL"})
    await task_distributor.update_worker_load("test_worker", 3)

    test_task = Task(task_type="CRAWL_URL", payload={})

    await task_distributor.complete_task("test_worker", test_task, "test_queue", success=True)

    # Verify worker load was decremented
    worker = task_distributor.workers["test_worker"]
    assert worker.current_load == 2

    # Verify task was acknowledged
    task_distributor.queue_manager.ack_task.assert_called_once_with("test_queue", test_task)

@pytest.mark.asyncio
async def test_complete_task_failure(task_distributor):
    # Register worker and set load
    await task_distributor.register_worker("test_worker", "crawler", {"CRAWL_URL"})
    await task_distributor.update_worker_load("test_worker", 3)

    test_task = Task(task_type="CRAWL_URL", payload={})

    await task_distributor.complete_task(
        "test_worker", test_task, "test_queue",
        success=False, error_message="Processing failed"
    )

    # Verify worker load was decremented
    worker = task_distributor.workers["test_worker"]
    assert worker.current_load == 2

    # Verify task was failed
    task_distributor.queue_manager.fail_task.assert_called_once_with(
        "test_queue", test_task, "Processing failed"
    )

@pytest.mark.asyncio
async def test_get_worker_stats(task_distributor):
    # Register a few workers
    await task_distributor.register_worker("worker1", "crawler", {"CRAWL_URL"}, max_capacity=5)
    await task_distributor.register_worker("worker2", "processor", {"PROCESS_CONTENT"}, max_capacity=10)
    await task_distributor.update_worker_load("worker1", 2)

    stats = await task_distributor.get_worker_stats()

    assert "worker1" in stats
    assert "worker2" in stats

    worker1_stats = stats["worker1"]
    assert worker1_stats["worker_type"] == "crawler"
    assert worker1_stats["current_load"] == 2
    assert worker1_stats["max_capacity"] == 5
    assert worker1_stats["load_percentage"] == 40.0
    assert worker1_stats["is_available"] is True

@pytest.mark.asyncio
async def test_worker_cleanup(task_distributor):
    # Register a worker
    await task_distributor.register_worker("test_worker", "crawler", {"CRAWL_URL"})

    # Manually set old heartbeat to simulate inactive worker
    worker = task_distributor.workers["test_worker"]
    worker.last_heartbeat = time.time() - 120  # 2 minutes ago

    # Run cleanup
    await task_distributor._cleanup_inactive_workers()

    # Worker should be removed
    assert "test_worker" not in task_distributor.workers