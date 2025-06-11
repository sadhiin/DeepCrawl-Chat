import json
import logging
from typing import Optional, Dict, Any

from src.deepcrawl_chat.core.redis import RedisPool
from src.deepcrawl_chat.core.utils import DeepCrawlError, ErrorCode
from src.deepcrawl_chat.queue.schemas import Task, TaskStatus
from src.deepcrawl_chat.queue.status_tracker import TaskStatusTracker

logger = logging.getLogger(__name__)

class QueueManager:
    """Manages task queues using Redis with persistence and recovery."""

    PENDING_SUFFIX = ":pending"
    PROCESSING_SUFFIX = ":processing"
    FAILED_SUFFIX = ":failed"
    MAX_RETRIES = 3 # Maximum number of retries for a task

    def __init__(self, redis_pool: RedisPool, enable_status_tracking: bool = True):
        """
        Initializes the QueueManager with a RedisPool instance.

        Args:
            redis_pool: An instance of RedisPool, assumed to be connected.
            enable_status_tracking: Whether to enable comprehensive status tracking
        """
        self.redis = redis_pool.pool
        self.max_retries = self.MAX_RETRIES
        self.enable_status_tracking = enable_status_tracking

        # Initialize status tracker if enabled
        if self.enable_status_tracking:
            self.status_tracker = TaskStatusTracker(redis_pool)
        else:
            self.status_tracker = None


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
            # Update task status to QUEUED and track if enabled
            if self.status_tracker:
                await self.status_tracker.track_task_status(
                    task,
                    TaskStatus.QUEUED,
                    f"Task enqueued to {queue_name}"
                )

            await self.redis.lpush(pending_queue, task.model_dump_json())
            logger.info(f"Enqueued task {task.task_id} of type '{task.task_type}' to queue '{pending_queue}'")
            return task.task_id
        except Exception as e:
            logger.error(f"Error enqueuing task {task.task_id} to '{pending_queue}': {e}", exc_info=True)
            raise DeepCrawlError(f"Failed to enqueue task {task.task_id} to '{pending_queue}': {e}", ErrorCode.QUEUE_ERROR)

    async def dequeue_task(self, queue_name: str, timeout: int = 5, worker_id: Optional[str] = None) -> Optional[Task]:
        """
        Dequeues a task from the pending queue and moves it to the processing queue.
        Uses BRPOPLPUSH for atomicity.

        Args:
            queue_name: The base name of the queue.
            timeout: Seconds to wait for a task before returning None.
            worker_id: ID of the worker dequeuing the task (for status tracking)

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

                # Update task status to ASSIGNED if we have a worker_id, otherwise PROCESSING
                if self.status_tracker:
                    new_status = TaskStatus.ASSIGNED if worker_id else TaskStatus.PROCESSING
                    await self.status_tracker.track_task_status(
                        task,
                        new_status,
                        f"Task dequeued from {queue_name}" + (f" by worker {worker_id}" if worker_id else ""),
                        worker_id=worker_id
                    )

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

    async def start_task_processing(self, queue_name: str, task: Task, worker_id: str) -> None:
        """
        Mark a task as actively being processed by a worker.

        Args:
            queue_name: The base name of the queue.
            task: The Task object being processed.
            worker_id: ID of the worker processing the task.
        """
        if self.status_tracker:
            await self.status_tracker.track_task_status(
                task,
                TaskStatus.PROCESSING,
                f"Task processing started by worker {worker_id}",
                worker_id=worker_id
            )

    async def ack_task(self, queue_name: str, task: Task, worker_id: Optional[str] = None) -> None:
        """
        Acknowledges successful processing of a task by removing it from the processing queue.

        Args:
            queue_name: The base name of the queue.
            task: The Task object that was successfully processed.
            worker_id: ID of the worker that completed the task (for status tracking)

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
                # Update task status to COMPLETED
                if self.status_tracker:
                    await self.status_tracker.track_task_status(
                        task,
                        TaskStatus.COMPLETED,
                        f"Task completed successfully" + (f" by worker {worker_id}" if worker_id else ""),
                        worker_id=worker_id
                    )

                logger.info(f"Task {task.task_id} acknowledged and removed from '{processing_queue}'")
        except Exception as e:
            logger.error(f"Error acknowledging task {task.task_id} in '{processing_queue}': {e}", exc_info=True)
            raise DeepCrawlError(f"Failed to acknowledge task {task.task_id} in '{processing_queue}': {e}", ErrorCode.QUEUE_ERROR)

    async def fail_task(self, queue_name: str, task: Task, error_message: str, worker_id: Optional[str] = None) -> None:
        """
        Handles a failed task. It increments the attempt count.
        If retries are exhausted, moves it to the failed queue.
        Otherwise, moves it back to the pending queue for a retry.

        Args:
            queue_name: The base name of the queue.
            task: The Task object that failed. This object will be modified.
            error_message: A message describing the failure.
            worker_id: ID of the worker that failed the task (for status tracking)

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

                # Update task status to FAILED
                if self.status_tracker:
                    await self.status_tracker.track_task_status(
                        task,
                        TaskStatus.FAILED,
                        f"Task failed after {task.attempts} attempts: {error_message}",
                        worker_id=worker_id,
                        details={"final_error": error_message, "total_attempts": task.attempts}
                    )
            else:
                logger.info(f"Task {task.task_id} failed (attempt {task.attempts}/{self.max_retries}). Re-queuing to '{pending_queue}'. Error: {error_message}")
                pipe.lpush(pending_queue, task.model_dump_json()) # Push updated task to pending

                # Update task status to RETRYING
                if self.status_tracker:
                    await self.status_tracker.track_task_status(
                        task,
                        TaskStatus.RETRYING,
                        f"Task retry {task.attempts}/{self.max_retries}: {error_message}",
                        worker_id=worker_id,
                        details={"retry_reason": error_message, "attempt": task.attempts, "max_retries": self.max_retries}
                    )

            results = await pipe.execute()
            removed_count = results[0]

            if removed_count == 0:
                logger.warning(f"Task {task.task_id} (JSON: {original_task_json}) not found in processing queue '{processing_queue}' during FAIL operation. It might have been acknowledged or moved by another process.")

        except Exception as e:
            logger.error(f"Error processing failure for task {task.task_id} from '{processing_queue}': {e}", exc_info=True)
            raise DeepCrawlError(f"Failed to process failure for task {task.task_id}: {e}", ErrorCode.QUEUE_ERROR)

    async def cancel_task(self, queue_name: str, task_id: str, reason: str = "", worker_id: Optional[str] = None) -> bool:
        """
        Cancel a task by removing it from queues and marking as cancelled.

        Args:
            queue_name: The base name of the queue.
            task_id: ID of the task to cancel.
            reason: Reason for cancellation.
            worker_id: ID of the worker/user cancelling the task.

        Returns:
            True if task was found and cancelled, False otherwise.
        """
        pending_queue, processing_queue, failed_queue = self._get_queue_names(queue_name)

        try:
            # Try to find and remove the task from all queues
            task_found = False
            task_data = None

            # Check pending queue first
            task_list = await self.redis.lrange(pending_queue, 0, -1)
            for task_json in task_list:
                if isinstance(task_json, bytes):
                    task_json = task_json.decode()

                try:
                    task = Task.model_validate_json(task_json)
                    if task.task_id == task_id:
                        await self.redis.lrem(pending_queue, 1, task_json)
                        task_data = task
                        task_found = True
                        break
                except json.JSONDecodeError:
                    continue

            # Check processing queue if not found in pending
            if not task_found:
                task_list = await self.redis.lrange(processing_queue, 0, -1)
                for task_json in task_list:
                    if isinstance(task_json, bytes):
                        task_json = task_json.decode()

                    try:
                        task = Task.model_validate_json(task_json)
                        if task.task_id == task_id:
                            await self.redis.lrem(processing_queue, 1, task_json)
                            task_data = task
                            task_found = True
                            break
                    except json.JSONDecodeError:
                        continue

            if task_found and task_data:
                # Update task status to CANCELLED
                if self.status_tracker:
                    await self.status_tracker.track_task_status(
                        task_data,
                        TaskStatus.CANCELLED,
                        f"Task cancelled: {reason}" if reason else "Task cancelled",
                        worker_id=worker_id,
                        details={"cancellation_reason": reason}
                    )

                logger.info(f"Task {task_id} cancelled from queue '{queue_name}'. Reason: {reason}")
                return True
            else:
                logger.warning(f"Task {task_id} not found in queue '{queue_name}' for cancellation")
                return False

        except Exception as e:
            logger.error(f"Error cancelling task {task_id} from '{queue_name}': {e}", exc_info=True)
            raise DeepCrawlError(f"Failed to cancel task {task_id}: {e}", ErrorCode.QUEUE_ERROR)


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
                    logger.warning(f"Task {task.task_id} exceeded max retries ({self.max_retries}) during requeue. Moving to failed queue '{failed_queue}'.")
                    await self.redis.lpush(failed_queue, task.model_dump_json())

                    # Update status to FAILED if tracking enabled
                    if self.status_tracker:
                        await self.status_tracker.track_task_status(
                            task,
                            TaskStatus.FAILED,
                            f"Task failed during requeue after {task.attempts} attempts",
                            details={"requeue_failure": True, "total_attempts": task.attempts}
                        )
                else:
                    logger.info(f"Re-queued task {task.task_id} to '{pending_queue}' (attempt {task.attempts}/{self.max_retries})")
                    await self.redis.lpush(pending_queue, task.model_dump_json())

                    # Update status to QUEUED if tracking enabled
                    if self.status_tracker:
                        await self.status_tracker.track_task_status(
                            task,
                            TaskStatus.QUEUED,
                            f"Task re-queued after worker recovery (attempt {task.attempts})",
                            details={"requeue_recovery": True, "attempt": task.attempts}
                        )

                count += 1

            logger.info(f"Re-queued {count} tasks from processing queue '{processing_queue}'")
            return count

        except Exception as e:
            logger.error(f"Error re-queuing tasks from '{processing_queue}': {e}", exc_info=True)
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
            A dictionary with keys "pending", "processing", "failed" and their respective sizes.

        Raises:
            DeepCrawlError: If fetching queue sizes fails.
        """
        try:
            pending_size = await self.get_queue_size(queue_name, "pending")
            processing_size = await self.get_queue_size(queue_name, "processing")
            failed_size = await self.get_queue_size(queue_name, "failed")

            return {
                "pending": pending_size,
                "processing": processing_size,
                "failed": failed_size
            }
        except Exception as e:
            logger.error(f"Error getting all queue sizes for '{queue_name}': {e}", exc_info=True)
            raise DeepCrawlError(f"Failed to get all queue sizes for '{queue_name}': {e}", ErrorCode.QUEUE_ERROR)

    async def delete_failed_tasks(self, queue_name: str) -> int:
        """
        Deletes all tasks from the failed queue of the specified base queue.

        Args:
            queue_name: The base name of the queue.

        Returns:
            The number of tasks deleted.

        Raises:
            DeepCrawlError: If deleting failed tasks fails.
        """
        _, _, failed_queue = self._get_queue_names(queue_name)
        try:
            # Get the current size before deletion for return value
            count = await self.redis.llen(failed_queue)

            # Delete the entire failed queue
            await self.redis.delete(failed_queue)

            logger.info(f"Deleted {count} failed tasks from queue '{failed_queue}'")
            return count
        except Exception as e:
            logger.error(f"Error deleting failed tasks from '{failed_queue}': {e}", exc_info=True)
            raise DeepCrawlError(f"Failed to delete failed tasks from '{failed_queue}': {e}", ErrorCode.QUEUE_ERROR)

    async def get_failed_tasks(self, queue_name: str, start: int = 0, end: int = -1) -> list[Task]:
        """
        Gets a range of failed tasks from the failed queue.

        Args:
            queue_name: The base name of the queue.
            start: Starting index (0-based). Default is 0.
            end: Ending index (inclusive, -1 means to the end). Default is -1.

        Returns:
            List of Task objects representing failed tasks.

        Raises:
            DeepCrawlError: If fetching failed tasks fails.
        """
        _, _, failed_queue = self._get_queue_names(queue_name)
        try:
            failed_tasks_json = await self.redis.lrange(failed_queue, start, end)

            failed_tasks = []
            for task_json in failed_tasks_json:
                if isinstance(task_json, bytes):
                    task_json = task_json.decode()

                try:
                    task = Task.model_validate_json(task_json)
                    failed_tasks.append(task)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to decode task from failed queue '{failed_queue}': {task_json}. Error: {e}")
                    # Skip malformed tasks
                    continue

            return failed_tasks
        except Exception as e:
            logger.error(f"Error getting failed tasks from '{failed_queue}': {e}", exc_info=True)
            raise DeepCrawlError(f"Failed to get failed tasks from '{failed_queue}': {e}", ErrorCode.QUEUE_ERROR)

    async def set_max_retries(self, retries: int):
        """
        Sets the maximum number of retries for tasks in this queue manager.

        Args:
            retries: The maximum number of retries. Must be >= 0.

        Raises:
            DeepCrawlError: If retries is negative.
        """
        if retries < 0:
            raise DeepCrawlError("Max retries must be >= 0", ErrorCode.QUEUE_ERROR)

        self.max_retries = retries
        logger.info(f"Set max retries to {retries}")

    # Status tracking integration methods

    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive status information for a task.

        Args:
            task_id: ID of the task

        Returns:
            Dictionary with status information or None if not found
        """
        if not self.status_tracker:
            logger.warning("Status tracking is disabled")
            return None

        return await self.status_tracker.get_task_status(task_id)

    async def get_task_progress(self, task_id: str):
        """
        Get progress information for a task.

        Args:
            task_id: ID of the task

        Returns:
            TaskProgress object or None if not found
        """
        if not self.status_tracker:
            logger.warning("Status tracking is disabled")
            return None

        return await self.status_tracker.get_task_progress(task_id)

    async def get_task_history(self, task_id: str, limit: int = 50):
        """
        Get status change history for a task.

        Args:
            task_id: ID of the task
            limit: Maximum number of history entries to return

        Returns:
            List of TaskStatusHistory objects
        """
        if not self.status_tracker:
            logger.warning("Status tracking is disabled")
            return []

        return await self.status_tracker.get_task_history(task_id, limit)

    async def get_tasks_by_status(self, status, limit: int = 100):
        """
        Get task IDs with a specific status.

        Args:
            status: Status to filter by
            limit: Maximum number of task IDs to return

        Returns:
            List of task IDs
        """
        if not self.status_tracker:
            logger.warning("Status tracking is disabled")
            return []

        return await self.status_tracker.get_tasks_by_status(status, limit)

    async def update_task_progress(self, task_id: str, progress) -> None:
        """
        Update progress information for a task.

        Args:
            task_id: ID of the task
            progress: TaskProgress object with updated information
        """
        if not self.status_tracker:
            logger.warning("Status tracking is disabled")
            return

        await self.status_tracker.update_task_progress(task_id, progress)