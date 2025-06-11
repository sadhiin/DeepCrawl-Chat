import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from src.deepcrawl_chat.queue.status_tracker import TaskStatusTracker
from src.deepcrawl_chat.queue.schemas import Task, TaskStatus, TaskProgress, TaskStatusHistory, TaskPriority
from src.deepcrawl_chat.core.utils import DeepCrawlError, ErrorCode


@pytest.fixture
async def redis_pool():
    """Mock Redis pool."""
    pool = MagicMock()
    redis_mock = AsyncMock()
    pool.pool = redis_mock

    # Mock Redis operations
    redis_mock.hset = AsyncMock()
    redis_mock.lpush = AsyncMock()
    redis_mock.sadd = AsyncMock()
    redis_mock.srem = AsyncMock()
    redis_mock.zadd = AsyncMock()
    redis_mock.zrem = AsyncMock()
    redis_mock.pipeline = MagicMock()
    redis_mock.hgetall = AsyncMock()
    redis_mock.lrange = AsyncMock()
    redis_mock.smembers = AsyncMock()
    redis_mock.sscan_iter = AsyncMock()
    redis_mock.zrangebyscore = AsyncMock()
    redis_mock.delete = AsyncMock()

    # Mock pipeline
    pipe_mock = AsyncMock()
    pipe_mock.hset = MagicMock()
    pipe_mock.lpush = MagicMock()
    pipe_mock.sadd = MagicMock()
    pipe_mock.srem = MagicMock()
    pipe_mock.zadd = MagicMock()
    pipe_mock.zrem = MagicMock()
    pipe_mock.execute = AsyncMock(return_value=[])
    redis_mock.pipeline.return_value = pipe_mock

    return pool


@pytest.fixture
def sample_task():
    """Create a sample task for testing."""
    return Task(
        task_id="test-task-123",
        task_type="CRAWL_URL",
        payload={"url": "http://example.com"},
        priority=TaskPriority.NORMAL
    )


@pytest.fixture
async def status_tracker(redis_pool):
    """Create TaskStatusTracker instance."""
    return TaskStatusTracker(redis_pool)


@pytest.mark.asyncio
class TestTaskStatusTracker:

    async def test_track_task_status_basic(self, status_tracker, sample_task):
        """Test basic status tracking functionality."""
        await status_tracker.track_task_status(
            sample_task,
            TaskStatus.PROCESSING,
            "Task started processing",
            worker_id="worker-1"
        )

        # Verify Redis operations were called
        pipe = status_tracker.redis.pipeline.return_value
        pipe.hset.assert_called()
        pipe.lpush.assert_called()
        pipe.sadd.assert_called()
        pipe.execute.assert_called_once()

    async def test_track_task_status_with_details(self, status_tracker, sample_task):
        """Test status tracking with additional details."""
        details = {"step": 1, "total_steps": 5}

        await status_tracker.track_task_status(
            sample_task,
            TaskStatus.PROCESSING,
            "Processing step 1",
            worker_id="worker-1",
            details=details
        )

        # Verify pipeline was used
        pipe = status_tracker.redis.pipeline.return_value
        pipe.execute.assert_called_once()

    async def test_track_status_change_index_update(self, status_tracker, sample_task):
        """Test that status indexes are properly updated when status changes."""
        # First status update
        sample_task.status = TaskStatus.PENDING
        await status_tracker.track_task_status(
            sample_task,
            TaskStatus.PROCESSING,
            "Started processing"
        )

        pipe = status_tracker.redis.pipeline.return_value
        # Should remove from old status and add to new status
        pipe.srem.assert_called()
        pipe.sadd.assert_called()

    async def test_update_task_progress(self, status_tracker):
        """Test updating task progress."""
        task_id = "test-task-123"
        progress = TaskProgress(
            percentage=50.0,
            current_step="Processing data",
            current_step_number=3,
            total_steps=5,
            processed_items=100,
            total_items=200
        )

        await status_tracker.update_task_progress(task_id, progress)

        # Verify Redis hset was called with progress data
        status_tracker.redis.hset.assert_called_once()
        args = status_tracker.redis.hset.call_args
        assert "task:progress:test-task-123" in args[0]

    async def test_get_task_status_found(self, status_tracker):
        """Test retrieving existing task status."""
        task_id = "test-task-123"
        status_data = {
            "status": "PROCESSING",
            "updated_at": "2023-01-01T10:00:00",
            "started_at": "2023-01-01T09:00:00",
            "assigned_worker_id": "worker-1",
            "attempts": "1",
            "error_message": ""
        }

        status_tracker.redis.hgetall.return_value = status_data

        result = await status_tracker.get_task_status(task_id)

        assert result is not None
        assert result["task_id"] == task_id
        assert result["status"] == "PROCESSING"
        assert result["assigned_worker_id"] == "worker-1"
        assert result["attempts"] == 1

    async def test_get_task_status_not_found(self, status_tracker):
        """Test retrieving non-existent task status."""
        task_id = "non-existent-task"
        status_tracker.redis.hgetall.return_value = {}

        result = await status_tracker.get_task_status(task_id)

        assert result is None

    async def test_get_task_progress_found(self, status_tracker):
        """Test retrieving existing task progress."""
        task_id = "test-task-123"
        progress_data = {
            "percentage": "75.5",
            "current_step": "Final processing",
            "total_steps": "4",
            "current_step_number": "3",
            "message": "Almost done",
            "processed_items": "150",
            "total_items": "200"
        }

        status_tracker.redis.hgetall.return_value = progress_data

        result = await status_tracker.get_task_progress(task_id)

        assert result is not None
        assert result.percentage == 75.5
        assert result.current_step == "Final processing"
        assert result.processed_items == 150
        assert result.total_items == 200

    async def test_get_task_progress_not_found(self, status_tracker):
        """Test retrieving non-existent task progress."""
        task_id = "non-existent-task"
        status_tracker.redis.hgetall.return_value = {}

        result = await status_tracker.get_task_progress(task_id)

        assert result is None

    async def test_get_task_history(self, status_tracker):
        """Test retrieving task status history."""
        task_id = "test-task-123"
        history_entries = [
            json.dumps({
                "status": "PROCESSING",
                "timestamp": "2023-01-01T10:00:00",
                "message": "Started processing",
                "worker_id": "worker-1",
                "details": {}
            }),
            json.dumps({
                "status": "QUEUED",
                "timestamp": "2023-01-01T09:00:00",
                "message": "Task queued",
                "worker_id": None,
                "details": {}
            })
        ]

        status_tracker.redis.lrange.return_value = history_entries

        result = await status_tracker.get_task_history(task_id, limit=10)

        assert len(result) == 2
        assert result[0].status == TaskStatus.PROCESSING
        assert result[0].worker_id == "worker-1"
        assert result[1].status == TaskStatus.QUEUED

    async def test_get_task_history_malformed_entry(self, status_tracker):
        """Test handling malformed history entries."""
        task_id = "test-task-123"
        history_entries = [
            json.dumps({
                "status": "PROCESSING",
                "timestamp": "2023-01-01T10:00:00",
                "message": "Started processing",
                "worker_id": "worker-1",
                "details": {}
            }),
            "malformed-json-entry",  # This should be skipped
            json.dumps({
                "status": "QUEUED",
                "timestamp": "2023-01-01T09:00:00",
                "message": "Task queued",
                "worker_id": None,
                "details": {}
            })
        ]

        status_tracker.redis.lrange.return_value = history_entries

        result = await status_tracker.get_task_history(task_id)

        # Should skip malformed entry and return only valid ones
        assert len(result) == 2
        assert result[0].status == TaskStatus.PROCESSING
        assert result[1].status == TaskStatus.QUEUED

    async def test_get_tasks_by_status(self, status_tracker):
        """Test retrieving tasks by status."""
        task_ids = [b"task-1", b"task-2", b"task-3"]

        async def mock_sscan_iter(key):
            for task_id in task_ids:
                yield task_id

        status_tracker.redis.sscan_iter = mock_sscan_iter

        result = await status_tracker.get_tasks_by_status(TaskStatus.PROCESSING, limit=5)

        assert len(result) == 3
        assert "task-1" in result
        assert "task-2" in result
        assert "task-3" in result

    async def test_get_tasks_by_status_with_limit(self, status_tracker):
        """Test retrieving tasks by status with limit."""
        task_ids = [b"task-1", b"task-2", b"task-3", b"task-4", b"task-5"]

        async def mock_sscan_iter(key):
            for task_id in task_ids:
                yield task_id

        status_tracker.redis.sscan_iter = mock_sscan_iter

        result = await status_tracker.get_tasks_by_status(TaskStatus.PROCESSING, limit=3)

        assert len(result) == 3

    async def test_get_worker_tasks(self, status_tracker):
        """Test retrieving tasks assigned to a worker."""
        worker_id = "worker-1"
        task_ids = {b"task-1", b"task-2", b"task-3"}

        status_tracker.redis.smembers.return_value = task_ids

        result = await status_tracker.get_worker_tasks(worker_id)

        assert len(result) == 3
        assert "task-1" in result
        assert "task-2" in result
        assert "task-3" in result

    async def test_get_timed_out_tasks(self, status_tracker):
        """Test retrieving timed out tasks."""
        timed_out_tasks = [b"task-1", b"task-2"]

        status_tracker.redis.zrangebyscore.return_value = timed_out_tasks

        result = await status_tracker.get_timed_out_tasks()

        assert len(result) == 2
        assert "task-1" in result
        assert "task-2" in result

        # Verify zrangebyscore was called with correct parameters
        status_tracker.redis.zrangebyscore.assert_called_once()

    async def test_cleanup_completed_tasks(self, status_tracker):
        """Test cleanup of old completed tasks."""
        # Mock get_tasks_by_status to return some task IDs
        task_ids = ["old-task-1", "old-task-2", "recent-task"]

        # Mock get_task_status to return different completion times
        old_time = (datetime.utcnow() - timedelta(hours=48)).isoformat()
        recent_time = (datetime.utcnow() - timedelta(hours=12)).isoformat()

        async def mock_get_task_status(task_id):
            if task_id.startswith("old"):
                return {"completed_at": old_time}
            else:
                return {"completed_at": recent_time}

        async def mock_get_tasks_by_status(status, limit):
            return task_ids

        status_tracker.get_task_status = mock_get_task_status
        status_tracker.get_tasks_by_status = mock_get_tasks_by_status
        status_tracker._cleanup_task_data = AsyncMock()

        result = await status_tracker.cleanup_completed_tasks(older_than_hours=24)

        # Should clean up 2 old tasks but not the recent one
        assert result == 2

        # Verify _cleanup_task_data was called for old tasks
        assert status_tracker._cleanup_task_data.call_count == 2

    async def test_timeout_tracking(self, status_tracker, sample_task):
        """Test task timeout tracking."""
        sample_task.timeout_seconds = 300  # 5 minutes

        await status_tracker.track_task_status(
            sample_task,
            TaskStatus.PROCESSING,
            "Started processing with timeout"
        )

        pipe = status_tracker.redis.pipeline.return_value
        # Should add task to timeout tracking
        pipe.zadd.assert_called()

    async def test_cleanup_task_data(self, status_tracker):
        """Test cleaning up task data."""
        task_id = "test-task"
        status = TaskStatus.COMPLETED

        await status_tracker._cleanup_task_data(task_id, status)

        pipe = status_tracker.redis.pipeline.return_value
        # Should delete all task-related keys
        pipe.delete.assert_called()
        pipe.srem.assert_called()
        pipe.zrem.assert_called()
        pipe.execute.assert_called()

    async def test_error_handling(self, status_tracker, sample_task):
        """Test error handling in status tracking."""
        # Mock Redis to raise an exception
        status_tracker.redis.pipeline.side_effect = Exception("Redis error")

        with pytest.raises(DeepCrawlError) as exc_info:
            await status_tracker.track_task_status(
                sample_task,
                TaskStatus.PROCESSING,
                "This should fail"
            )

        assert exc_info.value.error_code == ErrorCode.QUEUE_ERROR
        assert "Redis error" in str(exc_info.value)