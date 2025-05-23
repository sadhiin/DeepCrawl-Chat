import pytest
import asyncio
from src.deepcrawl_chat.core.redis import RedisPool
from src.deepcrawl_chat.queue.manager import QueueManager
from src.deepcrawl_chat.queue.schemas import Task
from src.deepcrawl_chat.core.utils import DeepCrawlError, ErrorCode

@pytest.fixture(scope="module")
async def redis_pool_instance():
    pool = RedisPool()
    await pool.connect()
    # Clear any pre-existing test queue data
    # Use a unique queue name for testing to avoid conflicts
    test_queue_name = "test_queue_manager_operations"
    await pool.pool.delete(test_queue_name)
    yield pool
    await pool.pool.delete(test_queue_name) # Clean up after tests
    await pool.close()

@pytest.fixture
async def queue_manager(redis_pool_instance: RedisPool):
    return QueueManager(redis_pool_instance)

@pytest.mark.asyncio
async def test_enqueue_dequeue_task(queue_manager: QueueManager):
    queue_name = "test_queue_manager_operations"
    task_payload = {"url": "http://example.com", "depth": 2}
    task_type = "crawl"

    # Enqueue
    task_id = await queue_manager.enqueue_task(queue_name, task_payload, task_type)
    assert isinstance(task_id, str)

    # Check queue size
    size = await queue_manager.get_queue_size(queue_name)
    assert size == 1

    # Dequeue
    dequeued_task = await queue_manager.dequeue_task(queue_name, timeout=1)
    assert dequeued_task is not None
    assert isinstance(dequeued_task, Task)
    assert dequeued_task.task_id == task_id
    assert dequeued_task.task_type == task_type
    assert dequeued_task.payload == task_payload

    # Check queue is empty
    size_after_dequeue = await queue_manager.get_queue_size(queue_name)
    assert size_after_dequeue == 0

@pytest.mark.asyncio
async def test_dequeue_empty_queue(queue_manager: QueueManager):
    queue_name = "test_empty_queue"
    # Ensure queue is empty by deleting it (brpop won't create it if it doesn't exist)
    await queue_manager.redis.delete(queue_name)

    dequeued_task = await queue_manager.dequeue_task(queue_name, timeout=1) # Short timeout
    assert dequeued_task is None

@pytest.mark.asyncio
async def test_get_queue_size(queue_manager: QueueManager):
    queue_name = "test_queue_size_operations"
    # Ensure queue is empty initially
    await queue_manager.redis.delete(queue_name)
    initial_size = await queue_manager.get_queue_size(queue_name)
    assert initial_size == 0

    # Enqueue a few tasks
    await queue_manager.enqueue_task(queue_name, {"data": "task1"})
    await queue_manager.enqueue_task(queue_name, {"data": "task2"})

    current_size = await queue_manager.get_queue_size(queue_name)
    assert current_size == 2

    # Clean up
    await queue_manager.redis.delete(queue_name)

@pytest.mark.asyncio
async def test_enqueue_failure_simulated(queue_manager: QueueManager, mocker):
    queue_name = "test_enqueue_failure"
    mocker.patch.object(queue_manager.redis, 'lpush', side_effect=Exception("Simulated Redis Error"))
    with pytest.raises(DeepCrawlError) as excinfo:
        await queue_manager.enqueue_task(queue_name, {"data": "test"})
    assert excinfo.value.error_code == ErrorCode.QUEUE_ERROR

@pytest.mark.asyncio
async def test_dequeue_failure_simulated(queue_manager: QueueManager, mocker):
    queue_name = "test_dequeue_failure"
    # Enqueue a task first so brpop has something to attempt
    await queue_manager.enqueue_task(queue_name, {"data": "test_for_dequeue_fail"})
    mocker.patch.object(queue_manager.redis, 'brpop', side_effect=Exception("Simulated Redis Error"))
    with pytest.raises(DeepCrawlError) as excinfo:
        await queue_manager.dequeue_task(queue_name)
    assert excinfo.value.error_code == ErrorCode.QUEUE_ERROR
    # Cleanup the enqueued item manually since dequeue failed
    await queue_manager.redis.delete(queue_name)

@pytest.mark.asyncio
async def test_dequeue_json_decode_error(queue_manager: QueueManager, mocker):
    queue_name = "test_json_decode_error"
    # Manually push a malformed JSON string to the queue
    await queue_manager.redis.lpush(queue_name, "not a valid json string")

    # Spy on logger.error to ensure it's called
    spy_logger_error = mocker.spy(logging.getLogger('src.deepcrawl_chat.queue.manager'), 'error')

    task = await queue_manager.dequeue_task(queue_name, timeout=1)
    assert task is None # Should return None for decode errors

    spy_logger_error.assert_called_once()
    args, _ = spy_logger_error.call_args
    assert "JSONDecodeError" in args[0]
    assert "not a valid json string" in args[0]

    # Clean up
    await queue_manager.redis.delete(queue_name)