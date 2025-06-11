import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from src.deepcrawl_chat.queue.metrics import MetricsCollector, QueueMetrics, WorkerMetrics, SystemMetrics
from src.deepcrawl_chat.queue.schemas import TaskStatus
from src.deepcrawl_chat.core.utils import DeepCrawlError, ErrorCode


@pytest.fixture
async def redis_pool():
    """Mock Redis pool."""
    pool = MagicMock()
    redis_mock = AsyncMock()
    pool.pool = redis_mock

    # Mock Redis operations
    redis_mock.get = AsyncMock()
    redis_mock.hset = AsyncMock()
    redis_mock.hgetall = AsyncMock()
    redis_mock.expire = AsyncMock()
    redis_mock.info = AsyncMock()
    redis_mock.pipeline = MagicMock()
    redis_mock.hincrby = AsyncMock()
    redis_mock.set = AsyncMock()

    # Mock pipeline
    pipe_mock = AsyncMock()
    pipe_mock.hincrby = MagicMock()
    pipe_mock.set = MagicMock()
    pipe_mock.expire = MagicMock()
    pipe_mock.execute = AsyncMock(return_value=[])
    redis_mock.pipeline.return_value = pipe_mock

    return pool


@pytest.fixture
async def metrics_collector(redis_pool):
    """Create MetricsCollector instance."""
    return MetricsCollector(redis_pool, collection_interval_seconds=30)


@pytest.fixture
def mock_queue_manager():
    """Mock QueueManager."""
    manager = MagicMock()
    manager.get_all_queue_sizes = AsyncMock(return_value={
        "pending": 5,
        "processing": 2,
        "failed": 1
    })

    # Mock status tracker
    status_tracker = MagicMock()
    status_tracker.get_tasks_by_status = AsyncMock(return_value=["task1", "task2", "task3"])
    manager.status_tracker = status_tracker

    return manager


@pytest.fixture
def mock_worker_info():
    """Mock WorkerInfo object."""
    worker_info = MagicMock()
    worker_info.worker_type = "crawler"
    worker_info.current_load = 3
    worker_info.max_capacity = 10
    worker_info.load_percentage = 30.0
    worker_info.is_active = True
    worker_info.last_heartbeat = datetime.utcnow().timestamp()
    return worker_info


@pytest.fixture
def mock_task_distributor():
    """Mock TaskDistributor."""
    distributor = MagicMock()
    distributor.get_worker_stats = AsyncMock(return_value={
        "worker1": {
            "worker_type": "crawler",
            "current_load": 3,
            "max_capacity": 10,
            "is_active": True
        },
        "worker2": {
            "worker_type": "processor",
            "current_load": 1,
            "max_capacity": 5,
            "is_active": True
        }
    })
    return distributor


@pytest.mark.asyncio
class TestMetricsCollector:

    async def test_collect_queue_metrics_basic(self, metrics_collector, mock_queue_manager):
        """Test basic queue metrics collection."""
        queue_name = "test_queue"

        result = await metrics_collector.collect_queue_metrics(queue_name, mock_queue_manager)

        assert isinstance(result, QueueMetrics)
        assert result.queue_name == queue_name
        assert result.pending_count == 5
        assert result.processing_count == 2
        assert result.failed_count == 1
        assert result.completed_count == 3  # From mock status tracker

        # Verify metrics were stored
        metrics_collector.redis.hset.assert_called()
        metrics_collector.redis.expire.assert_called()

    async def test_collect_queue_metrics_without_status_tracker(self, metrics_collector):
        """Test queue metrics collection when status tracker is not available."""
        queue_manager = MagicMock()
        queue_manager.get_all_queue_sizes = AsyncMock(return_value={
            "pending": 3,
            "processing": 1,
            "failed": 0
        })
        queue_manager.status_tracker = None

        result = await metrics_collector.collect_queue_metrics("test_queue", queue_manager)

        assert result.completed_count == 0
        assert result.cancelled_count == 0

    async def test_collect_worker_metrics(self, metrics_collector, mock_worker_info):
        """Test worker metrics collection."""
        worker_id = "test_worker"

        # Mock performance data
        metrics_collector.redis.hgetall.return_value = {
            "tasks_completed": "10",
            "tasks_failed": "2",
            "avg_duration": "15.5"
        }

        result = await metrics_collector.collect_worker_metrics(worker_id, mock_worker_info)

        assert isinstance(result, WorkerMetrics)
        assert result.worker_id == worker_id
        assert result.worker_type == "crawler"
        assert result.current_load == 3
        assert result.load_percentage == 30.0
        assert result.tasks_completed == 10
        assert result.tasks_failed == 2
        assert result.avg_task_duration_seconds == 15.5

    async def test_collect_worker_metrics_no_performance_data(self, metrics_collector, mock_worker_info):
        """Test worker metrics collection when no performance data exists."""
        worker_id = "new_worker"

        # Mock empty performance data
        metrics_collector.redis.hgetall.return_value = {}

        result = await metrics_collector.collect_worker_metrics(worker_id, mock_worker_info)

        assert result.tasks_completed == 0
        assert result.tasks_failed == 0
        assert result.avg_task_duration_seconds == 0.0

    async def test_collect_system_metrics(self, metrics_collector, mock_queue_manager, mock_task_distributor):
        """Test system metrics collection."""
        # Mock Redis info
        metrics_collector.redis.info.return_value = {
            "used_memory": 1048576,  # 1MB in bytes
            "connected_clients": 5
        }

        result = await metrics_collector.collect_system_metrics(mock_queue_manager, mock_task_distributor)

        assert isinstance(result, SystemMetrics)
        assert result.total_workers == 2
        assert result.active_workers == 2
        assert result.system_load_percentage == (4 / 15) * 100  # (3+1) / (10+5) * 100
        assert result.redis_memory_usage_mb == 1.0
        assert result.redis_connections == 5

    async def test_record_task_timing(self, metrics_collector):
        """Test recording task timing information."""
        queue_name = "test_queue"
        task_id = "task_123"
        event_type = "started"
        timestamp = datetime.utcnow()

        await metrics_collector.record_task_timing(queue_name, task_id, event_type, timestamp)

        # Verify Redis operations
        expected_key = f"timing:{queue_name}:{task_id}"
        metrics_collector.redis.hset.assert_called_with(
            expected_key,
            event_type,
            timestamp.isoformat()
        )
        metrics_collector.redis.expire.assert_called_with(expected_key, 86400)

    async def test_record_worker_performance(self, metrics_collector):
        """Test recording worker performance data."""
        worker_id = "worker_123"
        task_duration = 25.5
        success = True

        # Mock current average
        metrics_collector.redis.get.return_value = "20.0"

        await metrics_collector.record_worker_performance(worker_id, task_duration, success)

        # Verify pipeline operations
        pipe = metrics_collector.redis.pipeline.return_value
        pipe.hincrby.assert_called_with(f"worker_perf:{worker_id}", "tasks_completed", 1)
        pipe.set.assert_called()
        pipe.expire.assert_called()
        pipe.execute.assert_called()

    async def test_record_worker_performance_failure(self, metrics_collector):
        """Test recording worker performance for failed tasks."""
        worker_id = "worker_123"
        task_duration = 10.0
        success = False

        await metrics_collector.record_worker_performance(worker_id, task_duration, success)

        # Verify failure was recorded
        pipe = metrics_collector.redis.pipeline.return_value
        pipe.hincrby.assert_called_with(f"worker_perf:{worker_id}", "tasks_failed", 1)

    async def test_get_metrics_summary(self, metrics_collector):
        """Test generating metrics summary."""
        # Add some test data to metrics history
        now = datetime.utcnow()

        # Add queue metrics
        queue_metrics = QueueMetrics(
            queue_name="test_queue",
            pending_count=5,
            processing_count=2,
            tasks_processed_per_minute=10.0,
            error_rate_percentage=5.0,
            timestamp=now
        )
        metrics_collector.queue_metrics_history["test_queue"].append(queue_metrics)

        # Add worker metrics
        worker_metrics = WorkerMetrics(
            worker_id="worker1",
            worker_type="crawler",
            load_percentage=30.0,
            tasks_completed=100,
            uptime_percentage=98.5,
            timestamp=now
        )
        metrics_collector.worker_metrics_history["worker1"].append(worker_metrics)

        # Add system metrics
        system_metrics = SystemMetrics(
            active_workers=2,
            system_load_percentage=45.0,
            overall_throughput_per_minute=15.0,
            timestamp=now
        )
        metrics_collector.system_metrics_history.append(system_metrics)

        result = await metrics_collector.get_metrics_summary(time_range_hours=1)

        assert "queues" in result
        assert "workers" in result
        assert "system" in result
        assert result["time_range_hours"] == 1

        # Check queue summary
        assert "test_queue" in result["queues"]
        queue_summary = result["queues"]["test_queue"]
        assert queue_summary["avg_pending"] == 5.0
        assert queue_summary["avg_throughput"] == 10.0

        # Check worker summary
        assert "worker1" in result["workers"]
        worker_summary = result["workers"]["worker1"]
        assert worker_summary["avg_load_percentage"] == 30.0
        assert worker_summary["total_completed"] == 100

        # Check system summary
        system_summary = result["system"]
        assert system_summary["avg_workers"] == 2.0
        assert system_summary["avg_system_load"] == 45.0

    async def test_get_metrics_summary_empty_data(self, metrics_collector):
        """Test metrics summary with no historical data."""
        result = await metrics_collector.get_metrics_summary(time_range_hours=1)

        assert result["queues"] == {}
        assert result["workers"] == {}
        assert result["system"] == {}

    async def test_calculate_throughput(self, metrics_collector):
        """Test throughput calculation."""
        queue_name = "test_queue"
        event_type = "processed"

        # Mock Redis counter
        metrics_collector.redis.get.return_value = "15"

        result = await metrics_collector._calculate_throughput(queue_name, event_type)

        assert result == 15.0
        metrics_collector.redis.get.assert_called_with(f"throughput:{queue_name}:{event_type}")

    async def test_get_worker_performance_data(self, metrics_collector):
        """Test getting worker performance data from Redis."""
        worker_id = "worker_123"

        # Mock Redis data with bytes keys/values (common Redis behavior)
        redis_data = {
            b"tasks_completed": b"25",
            b"tasks_failed": b"3",
            b"avg_duration": b"12.5"
        }
        metrics_collector.redis.hgetall.return_value = redis_data

        result = await metrics_collector._get_worker_performance_data(worker_id)

        assert result["tasks_completed"] == 25
        assert result["tasks_failed"] == 3
        assert result["avg_duration"] == 12.5

    async def test_get_redis_metrics(self, metrics_collector):
        """Test getting Redis server metrics."""
        # Mock Redis info
        redis_info = {
            "used_memory": 2097152,  # 2MB
            "connected_clients": 10
        }
        metrics_collector.redis.info.return_value = redis_info

        result = await metrics_collector._get_redis_metrics()

        assert result["memory_usage_mb"] == 2.0
        assert result["connections"] == 10

    async def test_error_handling_in_collect_queue_metrics(self, metrics_collector):
        """Test error handling during queue metrics collection."""
        mock_queue_manager = MagicMock()
        mock_queue_manager.get_all_queue_sizes = AsyncMock(side_effect=Exception("Redis error"))

        with pytest.raises(DeepCrawlError) as exc_info:
            await metrics_collector.collect_queue_metrics("test_queue", mock_queue_manager)

        assert exc_info.value.error_code == ErrorCode.QUEUE_ERROR
        assert "Failed to collect queue metrics" in str(exc_info.value)

    async def test_error_handling_in_record_task_timing(self, metrics_collector):
        """Test error handling during task timing recording."""
        # Mock Redis to raise an exception
        metrics_collector.redis.hset.side_effect = Exception("Redis error")

        # Should not raise exception, just log error
        await metrics_collector.record_task_timing("queue", "task", "started")

        # Verify Redis operation was attempted
        metrics_collector.redis.hset.assert_called_once()

    async def test_metrics_storage_and_expiration(self, metrics_collector):
        """Test that metrics are stored with proper expiration times."""
        # Test queue metrics storage
        queue_metrics = QueueMetrics(queue_name="test", pending_count=5)
        await metrics_collector._store_queue_metrics(queue_metrics)

        metrics_collector.redis.hset.assert_called()
        metrics_collector.redis.expire.assert_called_with(
            "metrics:queue:test", 86400
        )

        # Test worker metrics storage
        worker_metrics = WorkerMetrics(worker_id="worker1", worker_type="crawler")
        await metrics_collector._store_worker_metrics(worker_metrics)

        # Test system metrics storage
        system_metrics = SystemMetrics()
        await metrics_collector._store_system_metrics(system_metrics)

        # All should have been called multiple times
        assert metrics_collector.redis.hset.call_count >= 3

    async def test_metrics_aggregation(self, metrics_collector):
        """Test metrics aggregation functions."""
        # Test queue metrics aggregation
        queue_metrics_list = [
            QueueMetrics("test", pending_count=5, processing_count=2, error_rate_percentage=10.0),
            QueueMetrics("test", pending_count=3, processing_count=4, error_rate_percentage=5.0),
            QueueMetrics("test", pending_count=7, processing_count=1, error_rate_percentage=15.0)
        ]

        result = metrics_collector._aggregate_queue_metrics(queue_metrics_list)

        assert result["avg_pending"] == 5.0  # (5+3+7)/3
        assert result["avg_processing"] == (2+4+1)/3
        assert result["avg_error_rate"] == 10.0  # (10+5+15)/3
        assert result["peak_pending"] == 7
        assert result["samples"] == 3

        # Test worker metrics aggregation
        worker_metrics_list = [
            WorkerMetrics("w1", "crawler", load_percentage=30.0, tasks_completed=10, uptime_percentage=98.0),
            WorkerMetrics("w1", "crawler", load_percentage=50.0, tasks_completed=15, uptime_percentage=97.0)
        ]

        result = metrics_collector._aggregate_worker_metrics(worker_metrics_list)

        assert result["avg_load_percentage"] == 40.0  # (30+50)/2
        assert result["total_completed"] == 15  # Latest count
        assert result["avg_uptime"] == 97.5  # (98+97)/2

        # Test system metrics aggregation
        system_metrics_list = [
            SystemMetrics(active_workers=3, system_load_percentage=60.0),
            SystemMetrics(active_workers=4, system_load_percentage=70.0)
        ]

        result = metrics_collector._aggregate_system_metrics(system_metrics_list)

        assert result["avg_workers"] == 3.5  # (3+4)/2
        assert result["avg_system_load"] == 65.0  # (60+70)/2
        assert result["peak_system_load"] == 70.0


@pytest.mark.asyncio
class TestMetricsDataClasses:

    def test_queue_metrics_creation(self):
        """Test QueueMetrics dataclass creation and defaults."""
        metrics = QueueMetrics(queue_name="test")

        assert metrics.queue_name == "test"
        assert metrics.pending_count == 0
        assert metrics.tasks_enqueued_per_minute == 0.0
        assert isinstance(metrics.timestamp, datetime)

    def test_worker_metrics_creation(self):
        """Test WorkerMetrics dataclass creation and defaults."""
        metrics = WorkerMetrics(worker_id="w1", worker_type="crawler")

        assert metrics.worker_id == "w1"
        assert metrics.worker_type == "crawler"
        assert metrics.current_load == 0
        assert metrics.uptime_percentage == 100.0
        assert metrics.is_active is True

    def test_system_metrics_creation(self):
        """Test SystemMetrics dataclass creation and defaults."""
        metrics = SystemMetrics()

        assert metrics.total_workers == 0
        assert metrics.system_load_percentage == 0.0
        assert metrics.redis_memory_usage_mb == 0.0
        assert isinstance(metrics.timestamp, datetime)