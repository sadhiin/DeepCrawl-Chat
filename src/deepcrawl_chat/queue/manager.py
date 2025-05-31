import json
import logging
from typing import Optional, Dict, Any

from src.deepcrawl_chat.core.redis import RedisPool
from src.deepcrawl_chat.core.utils import DeepCrawlError, ErrorCode
from src.deepcrawl_chat.queue.schemas import Task

logger = logging.getLogger(__name__)

class QueueManager:
    """Manages task queues using Redis with persistence and recovery."""

    PENDING_SUFFIX = ":pending"
    PROCESSING_SUFFIX = ":processing"
    FAILED_SUFFIX = ":failed"
    MAX_RETRIES = 3 # Maximum number of retries for a task

    def __init__(self, redis_pool: RedisPool):
        """
        Initializes the QueueManager with a RedisPool instance.

        Args:
            redis_pool: An instance of RedisPool, assumed to be connected.
        """
        self.redis = redis_pool.pool
        self.max_retries = self.MAX_RETRIES


    def _get_queue_names(self, base_queue_name: str) -> tuple[str, str, str]:
        """Helper to get names for pending, processing, and failed queues."""
        return (
            f"{base_queue_name}{self.PENDING_SUFFIX}",
            f"{base_queue_name}{self.PROCESSING_SUFFIX}",
            f"{base_queue_name}{self.FAILED_SUFFIX}",
        )

    async def enqueue_task(self, queue_name: str, task: Task) -> str:
        """
        Enqueues a pre-constructed Task object to the specified base Redis queue's pending list.
        The Task object should have its payload validated by the caller.

        Args:
            queue_name: The base name of the queue.
            task: The Task object to enqueue.

        Returns:
            The ID of the enqueued task.

        Raises:
            DeepCrawlError: If enqueuing fails.
        """
        pending_queue, _, _ = self._get_queue_names(queue_name)
        # Task object is now passed directly
        try:
            await self.redis.lpush(pending_queue, task.model_dump_json())
            logger.info(f"Enqueued task {task.task_id} of type '{task.task_type}' to queue '{pending_queue}'")
            return task.task_id
        except Exception as e:
            logger.error(f"Error enqueuing task {task.task_id} to '{pending_queue}': {e}", exc_info=True)
            raise DeepCrawlError(f"Failed to enqueue task {task.task_id} to '{pending_queue}': {e}", ErrorCode.QUEUE_ERROR)

    async def dequeue_task(self, queue_name: str, timeout: int = 5) -> Optional[Task]:
        """
        Dequeues a task from the pending queue and moves it to the processing queue.
        Uses BRPOPLPUSH for atomicity.

        Args:
            queue_name: The base name of the queue.
            timeout: Seconds to wait for a task before returning None.

        Returns:
            A Task object if a task is dequeued, otherwise None.

        Raises:
            DeepCrawlError: If dequeuing fails for reasons other than timeout or JSON decode error.
        """
        pending_queue, processing_queue, _ = self._get_queue_names(queue_name)
        task_data_json: Optional[str] = None
        try:
            # Atomically pop from pending and push to processing
            task_data_json = await self.redis.brpoplpush(pending_queue, processing_queue, timeout=timeout)
            if task_data_json:
                task = Task.model_validate_json(task_data_json)
                logger.info(f"Dequeued task {task.task_id} from '{pending_queue}' to '{processing_queue}'")
                return task
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSONDecodeError: Error decoding task from queue '{pending_queue}'. Data: '{task_data_json}'", exc_info=True)
            # If decoding fails, we can't process the task. Consider moving it to a dead-letter queue manually if needed.
            # For now, it remains in the processing_queue. A manual cleanup or a dedicated process might be needed for such cases.
            return None
        except Exception as e:
            logger.error(f"Error dequeuing task from '{pending_queue}': {e}", exc_info=True)
            # If brpoplpush fails, the task is not moved, so it remains in the pending queue.
            raise DeepCrawlError(f"Failed to dequeue task from '{pending_queue}': {e}", ErrorCode.QUEUE_ERROR)

    async def ack_task(self, queue_name: str, task: Task) -> None:
        """
        Acknowledges successful processing of a task by removing it from the processing queue.

        Args:
            queue_name: The base name of the queue.
            task: The Task object that was successfully processed.

        Raises:
            DeepCrawlError: If acknowledging the task fails.
        """
        _, processing_queue, _ = self._get_queue_names(queue_name)
        try:
            # LREM task_id from processing_queue (count 1 to remove only one instance)
            removed_count = await self.redis.lrem(processing_queue, 1, task.model_dump_json())
            if removed_count == 0:
                logger.warning(f"Task {task.task_id} not found in processing queue '{processing_queue}' during ACK.")
            else:
                logger.info(f"Task {task.task_id} acknowledged and removed from '{processing_queue}'")
        except Exception as e:
            logger.error(f"Error acknowledging task {task.task_id} in '{processing_queue}': {e}", exc_info=True)
            raise DeepCrawlError(f"Failed to acknowledge task {task.task_id} in '{processing_queue}': {e}", ErrorCode.QUEUE_ERROR)

    async def fail_task(self, queue_name: str, task: Task, error_message: str) -> None:
        """
        Handles a failed task. It increments the attempt count.
        If retries are exhausted, moves it to the failed queue.
        Otherwise, moves it back to the pending queue for a retry.

        Args:
            queue_name: The base name of the queue.
            task: The Task object that failed. This object will be modified.
            error_message: A message describing the failure.

        Raises:
            DeepCrawlError: If failing the task (moving it) fails.
        """
        pending_queue, processing_queue, failed_queue = self._get_queue_names(queue_name)

        # This is the JSON representation of the task as it was in the processing queue
        original_task_json = task.model_dump_json()

        task.attempts += 1
        task.error_message = error_message

        try:
            pipe = self.redis.pipeline()
            # Remove the original task from the processing queue
            pipe.lrem(processing_queue, 1, original_task_json)

            if task.attempts >= self.max_retries:
                logger.warning(f"Task {task.task_id} failed {task.attempts} times. Moving to failed queue '{failed_queue}'. Error: {error_message}")
                pipe.lpush(failed_queue, task.model_dump_json()) # Push updated task to failed
            else:
                logger.info(f"Task {task.task_id} failed (attempt {task.attempts}/{self.max_retries}). Re-queuing to '{pending_queue}'. Error: {error_message}")
                pipe.lpush(pending_queue, task.model_dump_json()) # Push updated task to pending

            results = await pipe.execute()
            removed_count = results[0]

            if removed_count == 0:
                logger.warning(f"Task {task.task_id} (JSON: {original_task_json}) not found in processing queue '{processing_queue}' during FAIL operation. It might have been acknowledged or moved by another process.")

        except Exception as e:
            logger.error(f"Error processing failure for task {task.task_id} from '{processing_queue}': {e}", exc_info=True)
            raise DeepCrawlError(f"Failed to process failure for task {task.task_id}: {e}", ErrorCode.QUEUE_ERROR)


    async def requeue_all_processing_tasks(self, queue_name: str) -> int:
        """
        Moves all tasks from a queue's processing list back to its pending list.
        This is useful for recovery if workers die and tasks are stuck in processing.
        Increments attempt count and moves to failed queue if max retries exceeded.

        Args:
            queue_name: The base name of the queue.

        Returns:
            The number of tasks re-queued or moved to failed.

        Raises:
            DeepCrawlError: If re-queuing fails.
        """
        pending_queue, processing_queue, failed_queue = self._get_queue_names(queue_name)
        count = 0
        try:
            while True:
                task_data_json = await self.redis.rpop(processing_queue) # Pop from the right (oldest)
                if task_data_json is None:
                    break # Queue is empty

                try:
                    task = Task.model_validate_json(task_data_json)
                except json.JSONDecodeError:
                    logger.error(f"Found malformed task data in '{processing_queue}' during requeue: {task_data_json[:100]}... Moving to failed queue.")
                    await self.redis.lpush(failed_queue, json.dumps({"task_id": "malformed", "error": "decode_error", "data": task_data_json}))
                    count += 1
                    continue

                task.attempts += 1
                task.error_message = "Re-queued due to worker inactivity or shutdown."

                if task.attempts >= self.max_retries:
                    logger.warning(f"Task {task.task_id} from '{processing_queue}' exceeded max retries ({task.attempts}). Moving to '{failed_queue}'.")
                    await self.redis.lpush(failed_queue, task.model_dump_json())
                else:
                    logger.info(f"Re-queueing task {task.task_id} (attempt {task.attempts}) from '{processing_queue}' to '{pending_queue}'.")
                    await self.redis.lpush(pending_queue, task.model_dump_json()) # Push to the left (becomes next to be dequeued)
                count += 1

            logger.info(f"Re-queued/failed {count} tasks from '{processing_queue}' for queue '{queue_name}'.")
            return count
        except Exception as e:
            logger.error(f"Error re-queueing tasks from '{processing_queue}': {e}", exc_info=True)
            raise DeepCrawlError(f"Failed to re-queue tasks from '{processing_queue}': {e}", ErrorCode.QUEUE_ERROR)


    async def get_queue_size(self, queue_name: str, queue_type: str = "pending") -> int:
        """
        Gets the current number of tasks in the specified part of the queue (pending, processing, or failed).

        Args:
            queue_name: The base name of the queue.
            queue_type: Type of queue part: "pending", "processing", or "failed". Default is "pending".

        Returns:
            The number of tasks in the specified queue part.

        Raises:
            DeepCrawlError: If fetching queue size fails or invalid queue_type.
        """
        pending_q, processing_q, failed_q = self._get_queue_names(queue_name)
        target_q_name = ""
        if queue_type == "pending":
            target_q_name = pending_q
        elif queue_type == "processing":
            target_q_name = processing_q
        elif queue_type == "failed":
            target_q_name = failed_q
        else:
            raise DeepCrawlError(f"Invalid queue_type '{queue_type}'. Must be 'pending', 'processing', or 'failed'.", ErrorCode.QUEUE_ERROR)

        try:
            size = await self.redis.llen(target_q_name)
            return size
        except Exception as e:
            logger.error(f"Error getting size of queue '{target_q_name}': {e}", exc_info=True)
            raise DeepCrawlError(f"Failed to get queue size for '{target_q_name}': {e}", ErrorCode.QUEUE_ERROR)

    async def get_all_queue_sizes(self, queue_name: str) -> Dict[str, int]:
        """
        Gets the sizes of pending, processing, and failed queues for a base queue name.

        Args:
            queue_name: The base name of the queue.

        Returns:
            A dictionary with sizes for "pending", "processing", and "failed" queues.
        """
        pending_q, processing_q, failed_q = self._get_queue_names(queue_name)
        try:
            pipe = self.redis.pipeline()
            pipe.llen(pending_q)
            pipe.llen(processing_q)
            pipe.llen(failed_q)
            sizes = await pipe.execute()
            return {
                "pending": sizes[0],
                "processing": sizes[1],
                "failed": sizes[2],
            }
        except Exception as e:
            logger.error(f"Error getting all queue sizes for '{queue_name}': {e}", exc_info=True)
            raise DeepCrawlError(f"Failed to get all queue sizes for '{queue_name}': {e}", ErrorCode.QUEUE_ERROR)

    async def delete_failed_tasks(self, queue_name: str) -> int:
        """
        Deletes all tasks from the failed queue.

        Args:
            queue_name: The base name of the queue.

        Returns:
            Number of tasks that were in the failed queue before deletion.

        Raises:
            DeepCrawlError: If deletion fails.
        """
        _, _, failed_queue = self._get_queue_names(queue_name)
        try:
            # Get the number of tasks before deleting the queue
            num_tasks_in_failed_queue = await self.redis.llen(failed_queue)
            if num_tasks_in_failed_queue > 0:
                await self.redis.delete(failed_queue)
                logger.info(f"Deleted failed queue '{failed_queue}' which contained {num_tasks_in_failed_queue} tasks.")
            else:
                logger.info(f"No tasks to delete in failed queue '{failed_queue}'.")
            return num_tasks_in_failed_queue
        except Exception as e:
            logger.error(f"Error deleting failed tasks from '{failed_queue}': {e}", exc_info=True)
            raise DeepCrawlError(f"Failed to delete failed tasks from '{failed_queue}': {e}", ErrorCode.QUEUE_ERROR)

    async def get_failed_tasks(self, queue_name: str, start: int = 0, end: int = -1) -> list[Task]:
        """
        Retrieves a list of tasks from the failed queue.

        Args:
            queue_name: The base name of the queue.
            start: Start index for lrange.
            end: End index for lrange.

        Returns:
            A list of Task objects from the failed queue.

        Raises:
            DeepCrawlError: If fetching failed tasks fails.
        """
        _, _, failed_queue = self._get_queue_names(queue_name)
        tasks = []
        try:
            task_data_list = await self.redis.lrange(failed_queue, start, end)
            for task_data_json in task_data_list:
                try:
                    tasks.append(Task.model_validate_json(task_data_json))
                except json.JSONDecodeError:
                    logger.error(f"Malformed task data in failed queue '{failed_queue}': {task_data_json[:100]}...")
                    # Create a placeholder task or skip
                    tasks.append(Task(task_id="malformed", task_type="unknown", payload={"error": "decode_failed", "raw_data": task_data_json}))
            return tasks
        except Exception as e:
            logger.error(f"Error retrieving failed tasks from '{failed_queue}': {e}", exc_info=True)
            raise DeepCrawlError(f"Failed to retrieve failed tasks from '{failed_queue}': {e}", ErrorCode.QUEUE_ERROR)

    async def set_max_retries(self, retries: int):
        """Sets the maximum number of retries for tasks."""
        if retries < 0:
            raise ValueError("Max retries cannot be negative.")
        self.max_retries = retries
        logger.info(f"Max task retries set to {self.max_retries}")