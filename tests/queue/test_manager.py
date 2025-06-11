import pytest
import asyncio
import json # Added for manipulating task data directly
from src.deepcrawl_chat.core.redis import RedisPool
from src.deepcrawl_chat.queue.manager import QueueManager
from src.deepcrawl_chat.queue.schemas import Task, CrawlURLPayload, TaskPriority
from src.deepcrawl_chat.core.utils import DeepCrawlError, ErrorCode
import logging # Added for spying on logger

# Unique base queue name for all tests in this module to avoid interference
BASE_TEST_QUEUE_NAME = "module_test_queue"

@pytest.fixture(scope="module")
async def redis_pool_instance():
    pool = RedisPool()
    # Ensure Redis is configured for tests (e.g., via environment variables)
    # Example: REDIS_HOST, REDIS_PORT, REDIS_DB could be set
    await pool.connect()
    # Clear any pre-existing test queue data for this module's base name
    pending_q, processing_q, failed_q = f"{BASE_TEST_QUEUE_NAME}{QueueManager.PENDING_SUFFIX}", \
                                        f"{BASE_TEST_QUEUE_NAME}{QueueManager.PROCESSING_SUFFIX}", \
                                        f"{BASE_TEST_QUEUE_NAME}{QueueManager.FAILED_SUFFIX}"
    await pool.pool.delete(pending_q, processing_q, failed_q)
    # Also clean up old style single queues if they exist from previous test versions
    await pool.pool.delete(BASE_TEST_QUEUE_NAME)

    # Clear specific test queues that might be used by name directly
    # These names are derived from how tests were previously structured
    queues_to_clear = [
        "test_queue_manager_operations", "test_empty_queue", "test_queue_size_operations",
        "test_enqueue_failure", "test_dequeue_failure", "test_json_decode_error",
        "test_ack_fail_flow_queue", "test_retry_logic_queue", "test_requeue_processing_queue",
        "test_malformed_requeue_queue"
    ]
    for q_base in queues_to_clear:
        await pool.pool.delete(
            f"{q_base}{QueueManager.PENDING_SUFFIX}",
            f"{q_base}{QueueManager.PROCESSING_SUFFIX}",
            f"{q_base}{QueueManager.FAILED_SUFFIX}",
            q_base # old style
        )

    yield pool

    # Clean up after all tests in the module
    await pool.pool.delete(pending_q, processing_q, failed_q)
    await pool.pool.delete(BASE_TEST_QUEUE_NAME)
    for q_base in queues_to_clear:
        await pool.pool.delete(
            f"{q_base}{QueueManager.PENDING_SUFFIX}",
            f"{q_base}{QueueManager.PROCESSING_SUFFIX}",
            f"{q_base}{QueueManager.FAILED_SUFFIX}",
            q_base
        )
    await pool.close()

@pytest.fixture
async def queue_manager(redis_pool_instance: RedisPool):
    # Each test function will get a new QueueManager, but they share the RedisPool
    qm = QueueManager(redis_pool_instance)
    # Reset max_retries to default for each test, in case a test modifies it
    await qm.set_max_retries(QueueManager.MAX_RETRIES)
    return qm

# Helper to clean up specific queues for a test
async def cleanup_queues(manager: QueueManager, base_name: str):
    pending_q, processing_q, failed_q = manager._get_queue_names(base_name)
    await manager.redis.delete(pending_q, processing_q, failed_q)

@pytest.mark.asyncio
async def test_enqueue_dequeue_task(queue_manager: QueueManager):
    queue_name = f"{BASE_TEST_QUEUE_NAME}_enqueue_dequeue"
    await cleanup_queues(queue_manager, queue_name)

    task_payload_data = {"url": "http://example.com", "depth": 2, "force_recrawl": False}
    crawl_payload = CrawlURLPayload(**task_payload_data)
    task_to_enqueue = Task(task_type="CRAWL_URL", payload=crawl_payload.model_dump(), priority=TaskPriority.HIGH)

    # Enqueue
    task_id = await queue_manager.enqueue_task(queue_name, task_to_enqueue)
    assert isinstance(task_id, str)
    assert task_id == task_to_enqueue.task_id

    # Check pending queue size
    size = await queue_manager.get_queue_size(queue_name, queue_type="pending")
    assert size == 1
    all_sizes = await queue_manager.get_all_queue_sizes(queue_name)
    assert all_sizes == {"pending": 1, "processing": 0, "failed": 0}


    # Dequeue
    dequeued_task = await queue_manager.dequeue_task(queue_name, timeout=1)
    assert dequeued_task is not None
    assert isinstance(dequeued_task, Task)
    assert dequeued_task.task_id == task_id
    assert dequeued_task.task_type == task_to_enqueue.task_type
    assert dequeued_task.payload == task_to_enqueue.payload
    assert dequeued_task.priority == task_to_enqueue.priority
    assert dequeued_task.attempts == 0 # Initial dequeue

    # Check queue sizes after dequeue (task moves to processing)
    all_sizes_after_dequeue = await queue_manager.get_all_queue_sizes(queue_name)
    assert all_sizes_after_dequeue == {"pending": 0, "processing": 1, "failed": 0}

    # Manually clean up the task from processing for this test, or use ack.
    # For this test, let's assume it would be acked.
    await queue_manager.ack_task(queue_name, dequeued_task)
    all_sizes_after_ack = await queue_manager.get_all_queue_sizes(queue_name)
    assert all_sizes_after_ack == {"pending": 0, "processing": 0, "failed": 0}

    await cleanup_queues(queue_manager, queue_name)


@pytest.mark.asyncio
async def test_dequeue_empty_queue(queue_manager: QueueManager):
    queue_name = f"{BASE_TEST_QUEUE_NAME}_empty"
    await cleanup_queues(queue_manager, queue_name)

    dequeued_task = await queue_manager.dequeue_task(queue_name, timeout=1) # Short timeout
    assert dequeued_task is None
    all_sizes = await queue_manager.get_all_queue_sizes(queue_name)
    assert all_sizes == {"pending": 0, "processing": 0, "failed": 0}
    await cleanup_queues(queue_manager, queue_name)

@pytest.mark.asyncio
async def test_get_queue_sizes(queue_manager: QueueManager):
    queue_name = f"{BASE_TEST_QUEUE_NAME}_sizes"
    await cleanup_queues(queue_manager, queue_name)

    initial_sizes = await queue_manager.get_all_queue_sizes(queue_name)
    assert initial_sizes == {"pending": 0, "processing": 0, "failed": 0}
    assert await queue_manager.get_queue_size(queue_name, "pending") == 0
    assert await queue_manager.get_queue_size(queue_name, "processing") == 0
    assert await queue_manager.get_queue_size(queue_name, "failed") == 0

    # Enqueue a few tasks
    task1_payload = CrawlURLPayload(url="http://example.com/1", depth=0)
    task1_obj = Task(task_type="CRAWL_URL", payload=task1_payload.model_dump())
    task1_id = await queue_manager.enqueue_task(queue_name, task1_obj)

    task2_payload = CrawlURLPayload(url="http://example.com/2", depth=0)
    task2_obj = Task(task_type="CRAWL_URL", payload=task2_payload.model_dump())
    await queue_manager.enqueue_task(queue_name, task2_obj)

    current_sizes = await queue_manager.get_all_queue_sizes(queue_name)
    assert current_sizes == {"pending": 2, "processing": 0, "failed": 0}

    # Dequeue one
    task1 = await queue_manager.dequeue_task(queue_name, timeout=1)
    assert task1 and task1.task_id == task1_id
    sizes_after_dequeue = await queue_manager.get_all_queue_sizes(queue_name)
    assert sizes_after_dequeue == {"pending": 1, "processing": 1, "failed": 0}

    # Fail it (will go back to pending if retries left)
    await queue_manager.fail_task(queue_name, task1, "Simulated failure")
    sizes_after_fail = await queue_manager.get_all_queue_sizes(queue_name)
    assert sizes_after_fail == {"pending": 2, "processing": 0, "failed": 0} # task1 back to pending
                                                                          # attempts = 1 for task1

    await cleanup_queues(queue_manager, queue_name)

@pytest.mark.asyncio
async def test_enqueue_failure_simulated(queue_manager: QueueManager, mocker):
    queue_name = f"{BASE_TEST_QUEUE_NAME}_enqueue_fail"
    await cleanup_queues(queue_manager, queue_name)
    mocker.patch.object(queue_manager.redis, 'lpush', side_effect=Exception("Simulated Redis Error"))

    task_payload_data = {"url": "http://example.com/fail", "depth": 0, "force_recrawl": False}
    crawl_payload = CrawlURLPayload(**task_payload_data)
    task_to_enqueue = Task(task_type="CRAWL_URL", payload=crawl_payload.model_dump())

    with pytest.raises(DeepCrawlError) as excinfo:
        await queue_manager.enqueue_task(queue_name, task_to_enqueue)
    assert excinfo.value.error_code == ErrorCode.QUEUE_ERROR
    await cleanup_queues(queue_manager, queue_name)

@pytest.mark.asyncio
async def test_dequeue_failure_simulated(queue_manager: QueueManager, mocker):
    queue_name = f"{BASE_TEST_QUEUE_NAME}_dequeue_fail"
    await cleanup_queues(queue_manager, queue_name)
    # Enqueue a task first so brpoplpush has something to attempt
    pending_q, _, _ = queue_manager._get_queue_names(queue_name)
    task_to_enqueue = Task(task_type="test", payload={"data": "test_for_dequeue_fail"})
    await queue_manager.redis.lpush(pending_q, task_to_enqueue.model_dump_json())

    mocker.patch.object(queue_manager.redis, 'brpoplpush', side_effect=Exception("Simulated Redis Error"))
    with pytest.raises(DeepCrawlError) as excinfo:
        await queue_manager.dequeue_task(queue_name)
    assert excinfo.value.error_code == ErrorCode.QUEUE_ERROR
    # Task should remain in pending queue
    assert await queue_manager.get_queue_size(queue_name, "pending") == 1
    await cleanup_queues(queue_manager, queue_name)


@pytest.mark.asyncio
async def test_dequeue_json_decode_error_moves_to_processing_logs_error(queue_manager: QueueManager, mocker):
    queue_name = f"{BASE_TEST_QUEUE_NAME}_json_decode"
    await cleanup_queues(queue_manager, queue_name)

    pending_q, processing_q, _ = queue_manager._get_queue_names(queue_name)
    malformed_json = "not a valid json string"

    # Manually push a malformed JSON string to the pending queue
    await queue_manager.redis.lpush(pending_q, malformed_json)

    spy_logger_error = mocker.spy(logging.getLogger('src.deepcrawl_chat.queue.manager'), 'error')

    task = await queue_manager.dequeue_task(queue_name, timeout=1)
    assert task is None # Should return None for decode errors as per current implementation

    spy_logger_error.assert_called_once()
    args, _ = spy_logger_error.call_args
    assert "JSONDecodeError" in args[0]
    assert malformed_json in args[0]

    # Check that the malformed task data was moved to the processing queue
    assert await queue_manager.redis.llen(pending_q) == 0
    assert await queue_manager.redis.llen(processing_q) == 1
    retrieved_malformed_task = await queue_manager.redis.lindex(processing_q, 0)
    assert retrieved_malformed_task == malformed_json

    await cleanup_queues(queue_manager, queue_name)


@pytest.mark.asyncio
async def test_ack_task_success_and_not_found(queue_manager: QueueManager):
    queue_name = f"{BASE_TEST_QUEUE_NAME}_ack"
    await cleanup_queues(queue_manager, queue_name)

    task_payload_data = {"url": "http://example.com/ack", "depth": 0}
    ack_payload = CrawlURLPayload(**task_payload_data)
    task_obj_ack = Task(task_type="CRAWL_URL", payload=ack_payload.model_dump())
    task_id = await queue_manager.enqueue_task(queue_name, task_obj_ack)

    task = await queue_manager.dequeue_task(queue_name, timeout=1)
    assert task is not None
    assert await queue_manager.get_queue_size(queue_name, "processing") == 1

    # Ack the task
    await queue_manager.ack_task(queue_name, task)
    assert await queue_manager.get_queue_size(queue_name, "processing") == 0

    # Try to ack again (should log warning but not error)
    await queue_manager.ack_task(queue_name, task)
    assert await queue_manager.get_queue_size(queue_name, "processing") == 0 # Still 0
    await cleanup_queues(queue_manager, queue_name)

@pytest.mark.asyncio
async def test_fail_task_retry_and_max_retries(queue_manager: QueueManager):
    queue_name = f"{BASE_TEST_QUEUE_NAME}_fail_retry"
    await cleanup_queues(queue_manager, queue_name)
    await queue_manager.set_max_retries(2) # Test with 2 retries

    fail_payload = CrawlURLPayload(url="http://example.com/fail_retry", depth=0)
    task_obj_fail = Task(task_type="CRAWL_URL", payload=fail_payload.model_dump())
    task_id = await queue_manager.enqueue_task(queue_name, task_obj_fail)

    # --- Attempt 1 (initial dequeue) ---
    task = await queue_manager.dequeue_task(queue_name, timeout=1)
    assert task is not None and task.task_id == task_id
    assert task.attempts == 0
    assert await queue_manager.get_all_queue_sizes(queue_name) == {"pending": 0, "processing": 1, "failed": 0}

    # Fail task - Attempt 1
    await queue_manager.fail_task(queue_name, task, "Failure 1")
    assert await queue_manager.get_all_queue_sizes(queue_name) == {"pending": 1, "processing": 0, "failed": 0}

    # --- Attempt 2 ---
    task_attempt_2 = await queue_manager.dequeue_task(queue_name, timeout=1)
    assert task_attempt_2 is not None and task_attempt_2.task_id == task_id
    assert task_attempt_2.attempts == 1 # Incremented by fail_task before requeue
    assert task_attempt_2.error_message == "Failure 1"

    # Fail task - Attempt 2
    await queue_manager.fail_task(queue_name, task_attempt_2, "Failure 2")
    assert await queue_manager.get_all_queue_sizes(queue_name) == {"pending": 1, "processing": 0, "failed": 0}

    # --- Attempt 3 (should go to failed queue) ---
    task_attempt_3 = await queue_manager.dequeue_task(queue_name, timeout=1)
    assert task_attempt_3 is not None and task_attempt_3.task_id == task_id
    assert task_attempt_3.attempts == 2
    assert task_attempt_3.error_message == "Failure 2"

    # Fail task - Attempt 3 (max_retries is 2, so this is the 3rd processing attempt)
    await queue_manager.fail_task(queue_name, task_attempt_3, "Failure 3")
    # Now it should be in the failed queue
    assert await queue_manager.get_all_queue_sizes(queue_name) == {"pending": 0, "processing": 0, "failed": 1}

    failed_tasks = await queue_manager.get_failed_tasks(queue_name)
    assert len(failed_tasks) == 1
    failed_task = failed_tasks[0]
    assert failed_task.task_id == task_id
    assert failed_task.attempts == 3
    assert failed_task.error_message == "Failure 3"

    # Test deleting failed tasks
    deleted_count = await queue_manager.delete_failed_tasks(queue_name)
    assert deleted_count == 1 # It contained 1 task
    assert await queue_manager.get_queue_size(queue_name, "failed") == 0

    await cleanup_queues(queue_manager, queue_name)
    await queue_manager.set_max_retries(QueueManager.MAX_RETRIES) # Reset for other tests

@pytest.mark.asyncio
async def test_requeue_all_processing_tasks(queue_manager: QueueManager):
    queue_name = f"{BASE_TEST_QUEUE_NAME}_requeue"
    await cleanup_queues(queue_manager, queue_name)
    await queue_manager.set_max_retries(1) # Max 1 retry (so 2 total processing attempts)

    # Enqueue and dequeue two tasks to simulate them being processed
    requeue1_payload = CrawlURLPayload(url="http://example.com/requeue1", depth=0)
    task1_obj_requeue = Task(task_type="CRAWL_URL", payload=requeue1_payload.model_dump())
    task1_id = await queue_manager.enqueue_task(queue_name, task1_obj_requeue)

    requeue2_payload = CrawlURLPayload(url="http://example.com/requeue2", depth=0)
    task2_obj_requeue = Task(task_type="CRAWL_URL", payload=requeue2_payload.model_dump())
    task2_id = await queue_manager.enqueue_task(queue_name, task2_obj_requeue)

    task1 = await queue_manager.dequeue_task(queue_name, timeout=1) # task1 to processing
    task2 = await queue_manager.dequeue_task(queue_name, timeout=1) # task2 to processing
    assert task1 and task2
    assert await queue_manager.get_all_queue_sizes(queue_name) == {"pending": 0, "processing": 2, "failed": 0}

    # Simulate worker crash - call requeue_all_processing_tasks
    requeued_count = await queue_manager.requeue_all_processing_tasks(queue_name)
    assert requeued_count == 2

    # Both tasks should be back in pending, attempts incremented
    sizes = await queue_manager.get_all_queue_sizes(queue_name)
    assert sizes == {"pending": 2, "processing": 0, "failed": 0}

    # Dequeue them again
    task1_requeued = await queue_manager.dequeue_task(queue_name, timeout=1)
    task2_requeued = await queue_manager.dequeue_task(queue_name, timeout=1)
    assert task1_requeued and task2_requeued

    # Check attempts (original was 0, requeue adds 1)
    # Order might vary, so check by payload or ensure IDs match
    if CrawlURLPayload(**task1_requeued.payload).url == "http://example.com/requeue1":
        assert task1_requeued.attempts == 1
        assert task2_requeued.attempts == 1
    else: # task2 came first
        assert task2_requeued.attempts == 1
        assert task1_requeued.attempts == 1
        # Swap them for consistent checking below
        task1_requeued, task2_requeued = task2_requeued, task1_requeued


    assert await queue_manager.get_all_queue_sizes(queue_name) == {"pending": 0, "processing": 2, "failed": 0}

    # Simulate another crash and requeue. Now they should go to failed queue (max_retries = 1)
    requeued_count_2 = await queue_manager.requeue_all_processing_tasks(queue_name)
    assert requeued_count_2 == 2

    final_sizes = await queue_manager.get_all_queue_sizes(queue_name)
    assert final_sizes == {"pending": 0, "processing": 0, "failed": 2}

    failed_tasks = await queue_manager.get_failed_tasks(queue_name)
    assert len(failed_tasks) == 2
    for ft in failed_tasks:
        assert ft.attempts == 2 # Initial processing + 1 requeue + 1 (failed) requeue = 2 processing attempts, then fails
        assert ft.error_message == "Re-queued due to worker inactivity or shutdown."

    await cleanup_queues(queue_manager, queue_name)
    await queue_manager.set_max_retries(QueueManager.MAX_RETRIES)


@pytest.mark.asyncio
async def test_requeue_malformed_task_in_processing(queue_manager: QueueManager, mocker):
    queue_name = f"{BASE_TEST_QUEUE_NAME}_requeue_malformed"
    await cleanup_queues(queue_manager, queue_name)

    _, processing_q, failed_q = queue_manager._get_queue_names(queue_name)
    malformed_json = "this is definitely not json"

    # Manually put a malformed task in the processing queue
    await queue_manager.redis.lpush(processing_q, malformed_json)
    # Add a valid task as well to ensure it's processed correctly
    valid_task_payload_dict = {"url":"http://example.com/valid_requeue", "depth": 0}
    valid_crawl_payload = CrawlURLPayload(**valid_task_payload_dict)
    valid_task = Task(task_type="CRAWL_URL", payload=valid_crawl_payload.model_dump())
    await queue_manager.redis.lpush(processing_q, valid_task.model_dump_json())

    assert await queue_manager.redis.llen(processing_q) == 2

    spy_logger_error = mocker.spy(logging.getLogger('src.deepcrawl_chat.queue.manager'), 'error')

    requeued_count = await queue_manager.requeue_all_processing_tasks(queue_name)
    assert requeued_count == 2 # Both items were processed from the processing queue

    # Malformed task should go to failed queue
    # Valid task should go to pending (or failed if retries exhausted, but here it's first requeue)

    failed_tasks_json = await queue_manager.redis.lrange(failed_q, 0, -1)
    assert len(failed_tasks_json) == 1

    # The malformed data itself is stored with some context
    # Check that the logger was called for the malformed task
    assert spy_logger_error.call_count >= 1
    # Get the call related to malformed task
    found_malformed_log = False
    for call in spy_logger_error.call_args_list:
        args, _ = call
        if "Found malformed task data" in args[0] and malformed_json in args[0]:
            found_malformed_log = True
            break
    assert found_malformed_log, "Logger was not called for malformed task during requeue"


    # Check the actual content of the failed queue for the malformed task
    # The `requeue_all_processing_tasks` wraps it in a new JSON structure
    malformed_entry_in_failed_q = json.loads(failed_tasks_json[0])
    assert malformed_entry_in_failed_q["task_id"] == "malformed"
    assert malformed_entry_in_failed_q["error"] == "decode_error"
    assert malformed_entry_in_failed_q["data"] == malformed_json


    # Valid task should be in pending
    pending_tasks = await queue_manager.redis.lrange(f"{queue_name}{QueueManager.PENDING_SUFFIX}", 0, -1)
    assert len(pending_tasks) == 1
    pending_task_obj = Task.model_validate_json(pending_tasks[0])
    assert pending_task_obj.payload == valid_crawl_payload.model_dump() # Compare dicts
    assert pending_task_obj.attempts == 1 # Incremented by requeue

    await cleanup_queues(queue_manager, queue_name)

@pytest.mark.asyncio
async def test_invalid_queue_type_for_get_size(queue_manager: QueueManager):
    with pytest.raises(DeepCrawlError) as excinfo:
        await queue_manager.get_queue_size("any_queue", "invalid_type")
    assert "Invalid queue_type" in str(excinfo.value)
    assert excinfo.value.error_code == ErrorCode.QUEUE_ERROR

@pytest.mark.asyncio
async def test_set_max_retries(queue_manager: QueueManager):
    await queue_manager.set_max_retries(5)
    assert queue_manager.max_retries == 5
    await queue_manager.set_max_retries(0)
    assert queue_manager.max_retries == 0
    with pytest.raises(ValueError):
        await queue_manager.set_max_retries(-1)
    # Reset to default for other tests if this wasn't the last one
    await queue_manager.set_max_retries(QueueManager.MAX_RETRIES)