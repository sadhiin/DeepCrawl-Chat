import json
import logging
from typing import Optional, Dict, Any

from src.deepcrawl_chat.core.redis import RedisPool
from src.deepcrawl_chat.core.utils import DeepCrawlError, ErrorCode
from src.deepcrawl_chat.queue.schemas import Task

logger = logging.getLogger(__name__)

class QueueManager:
    """Manages task queues using Redis."""

    def __init__(self, redis_pool: RedisPool):
        """
        Initializes the QueueManager with a RedisPool instance.

        Args:
            redis_pool: An instance of RedisPool, assumed to be connected.
        """
        self.redis = redis_pool.pool  # Direct access to the connected redis client from RedisPool

    async def enqueue_task(self, queue_name: str, task_payload: Dict[str, Any], task_type: str = "generic") -> str:
        """
        Enqueues a task to the specified Redis queue.

        Args:
            queue_name: The name of the queue.
            task_payload: The payload for the task.
            task_type: The type of the task.

        Returns:
            The ID of the enqueued task.

        Raises:
            DeepCrawlError: If enqueuing fails.
        """
        task = Task(task_type=task_type, payload=task_payload)
        try:
            await self.redis.lpush(queue_name, task.model_dump_json())
            logger.info(f"Enqueued task {task.task_id} of type '{task.task_type}' to queue '{queue_name}'")
            return task.task_id
        except Exception as e:
            logger.error(f"Error enqueuing task {task.task_id} to '{queue_name}': {e}", exc_info=True)
            raise DeepCrawlError(f"Failed to enqueue task to '{queue_name}': {e}", ErrorCode.QUEUE_ERROR)

    async def dequeue_task(self, queue_name: str, timeout: int = 5) -> Optional[Task]:
        """
        Dequeues a task from the specified Redis queue with a timeout.

        Args:
            queue_name: The name of the queue.
            timeout: Seconds to wait for a task before returning None.

        Returns:
            A Task object if a task is dequeued, otherwise None.

        Raises:
            DeepCrawlError: If dequeuing fails for reasons other than timeout or JSON decode error.
        """
        task_data_json: Optional[str] = None
        try:
            result = await self.redis.brpop(queue_name, timeout=timeout)
            if result:
                # result is a tuple (queue_name_bytes, value_bytes) if decode_responses=False for redis client
                # if decode_responses=True, result is (queue_name_str, value_str)
                task_data_json = result[1]
                task = Task.model_validate_json(task_data_json)
                logger.info(f"Dequeued task {task.task_id} of type '{task.task_type}' from queue '{queue_name}'")
                return task
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSONDecodeError: Error decoding task from queue '{queue_name}': {e}. Data: '{task_data_json}'", exc_info=True)
            # Consider moving to a dead-letter queue or logging and discarding
            # For now, we'll return None, effectively discarding the malformed message after logging.
            return None
        except Exception as e:
            logger.error(f"Error dequeuing task from '{queue_name}': {e}", exc_info=True)
            raise DeepCrawlError(f"Failed to dequeue task from '{queue_name}': {e}", ErrorCode.QUEUE_ERROR)

    async def get_queue_size(self, queue_name: str) -> int:
        """
        Gets the current number of tasks in the specified queue.

        Args:
            queue_name: The name of the queue.

        Returns:
            The number of tasks in the queue.

        Raises:
            DeepCrawlError: If fetching queue size fails.
        """
        try:
            size = await self.redis.llen(queue_name)
            return size
        except Exception as e:
            logger.error(f"Error getting size of queue '{queue_name}': {e}", exc_info=True)
            raise DeepCrawlError(f"Failed to get queue size for '{queue_name}': {e}", ErrorCode.QUEUE_ERROR)