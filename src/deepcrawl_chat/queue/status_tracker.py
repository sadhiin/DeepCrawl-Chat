import json
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

from src.deepcrawl_chat.core.redis import RedisPool
from src.deepcrawl_chat.core.utils import DeepCrawlError, ErrorCode
from src.deepcrawl_chat.queue.schemas import Task, TaskStatus, TaskStatusHistory, TaskProgress

logger = logging.getLogger(__name__)


class TaskStatusTracker:
    """Manages task status tracking, history, and progress updates using Redis."""

    def __init__(self, redis_pool: RedisPool):
        """Initialize the TaskStatusTracker with a Redis connection pool."""
        self.redis = redis_pool.pool

    # Redis key patterns
    TASK_STATUS_KEY = "task:status:{task_id}"
    TASK_HISTORY_KEY = "task:history:{task_id}"
    TASK_PROGRESS_KEY = "task:progress:{task_id}"
    STATUS_INDEX_KEY = "status_index:{status}"
    WORKER_TASKS_KEY = "worker:tasks:{worker_id}"
    TASK_TIMEOUT_KEY = "task:timeouts"

    async def track_task_status(
        self,
        task: Task,
        new_status: TaskStatus,
        message: str = "",
        worker_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update task status with comprehensive tracking.

        Args:
            task: The task to update
            new_status: New status for the task
            message: Optional message describing the status change
            worker_id: ID of the worker handling the task
            details: Additional details about the status change
        """
        try:
            # Update task status
            old_status = task.status
            task.update_status(new_status, message, worker_id)

            # Create status history entry
            history_entry = TaskStatusHistory(
                status=new_status,
                timestamp=datetime.utcnow(),
                message=message,
                worker_id=worker_id,
                details=details or {}
            )

            # Use Redis pipeline for atomic updates
            pipe = self.redis.pipeline()

            # Store updated task status
            pipe.hset(
                self.TASK_STATUS_KEY.format(task_id=task.task_id),
                mapping={
                    "status": new_status.value,
                    "updated_at": task.updated_at.isoformat(),
                    "started_at": task.started_at.isoformat() if task.started_at else "",
                    "completed_at": task.completed_at.isoformat() if task.completed_at else "",
                    "assigned_worker_id": task.assigned_worker_id or "",
                    "attempts": task.attempts,
                    "error_message": task.error_message or "",
                }
            )

            # Add to status history
            pipe.lpush(
                self.TASK_HISTORY_KEY.format(task_id=task.task_id),
                history_entry.model_dump_json()
            )

            # Update status indexes
            if old_status != new_status:
                # Remove from old status index
                if old_status:
                    pipe.srem(self.STATUS_INDEX_KEY.format(status=old_status.value), task.task_id)

                # Add to new status index
                pipe.sadd(self.STATUS_INDEX_KEY.format(status=new_status.value), task.task_id)

            # Update worker-task mapping
            if worker_id:
                if new_status in [TaskStatus.ASSIGNED, TaskStatus.PROCESSING]:
                    pipe.sadd(self.WORKER_TASKS_KEY.format(worker_id=worker_id), task.task_id)
                elif new_status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                    pipe.srem(self.WORKER_TASKS_KEY.format(worker_id=worker_id), task.task_id)

            # Set task timeout if specified
            if task.timeout_seconds and new_status in [TaskStatus.ASSIGNED, TaskStatus.PROCESSING]:
                timeout_timestamp = datetime.utcnow() + timedelta(seconds=task.timeout_seconds)
                pipe.zadd(
                    self.TASK_TIMEOUT_KEY,
                    {task.task_id: timeout_timestamp.timestamp()}
                )
            elif new_status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                # Remove from timeout tracking if task is completed
                pipe.zrem(self.TASK_TIMEOUT_KEY, task.task_id)

            # Execute all operations atomically
            await pipe.execute()

            logger.info(
                f"Task {task.task_id} status updated: {old_status} -> {new_status} "
                f"(worker: {worker_id}, message: {message})"
            )

        except Exception as e:
            logger.error(f"Error tracking status for task {task.task_id}: {e}", exc_info=True)
            raise DeepCrawlError(
                f"Failed to track status for task {task.task_id}: {e}",
                ErrorCode.QUEUE_ERROR
            )

    async def update_task_progress(
        self,
        task_id: str,
        progress: TaskProgress
    ) -> None:
        """
        Update task progress information.

        Args:
            task_id: ID of the task to update
            progress: Progress information
        """
        try:
            progress_data = {
                "percentage": progress.percentage,
                "current_step": progress.current_step,
                "total_steps": progress.total_steps,
                "current_step_number": progress.current_step_number,
                "message": progress.message,
                "processed_items": progress.processed_items,
                "total_items": progress.total_items or 0,
                "updated_at": datetime.utcnow().isoformat()
            }

            await self.redis.hset(
                self.TASK_PROGRESS_KEY.format(task_id=task_id),
                mapping=progress_data
            )

            logger.debug(f"Updated progress for task {task_id}: {progress.percentage}%")

        except Exception as e:
            logger.error(f"Error updating progress for task {task_id}: {e}", exc_info=True)
            raise DeepCrawlError(
                f"Failed to update progress for task {task_id}: {e}",
                ErrorCode.QUEUE_ERROR
            )

    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current status information for a task.

        Args:
            task_id: ID of the task

        Returns:
            Dictionary with status information or None if not found
        """
        try:
            status_data = await self.redis.hgetall(
                self.TASK_STATUS_KEY.format(task_id=task_id)
            )

            if not status_data:
                return None

            # Decode bytes to strings if necessary
            if isinstance(list(status_data.keys())[0], bytes):
                status_data = {k.decode(): v.decode() for k, v in status_data.items()}

            return {
                "task_id": task_id,
                "status": status_data.get("status"),
                "updated_at": status_data.get("updated_at"),
                "started_at": status_data.get("started_at") or None,
                "completed_at": status_data.get("completed_at") or None,
                "assigned_worker_id": status_data.get("assigned_worker_id") or None,
                "attempts": int(status_data.get("attempts", 0)),
                "error_message": status_data.get("error_message") or None,
            }

        except Exception as e:
            logger.error(f"Error getting status for task {task_id}: {e}", exc_info=True)
            raise DeepCrawlError(
                f"Failed to get status for task {task_id}: {e}",
                ErrorCode.QUEUE_ERROR
            )

    async def get_task_progress(self, task_id: str) -> Optional[TaskProgress]:
        """
        Get current progress information for a task.

        Args:
            task_id: ID of the task

        Returns:
            TaskProgress object or None if not found
        """
        try:
            progress_data = await self.redis.hgetall(
                self.TASK_PROGRESS_KEY.format(task_id=task_id)
            )

            if not progress_data:
                return None

            # Decode bytes to strings if necessary
            if isinstance(list(progress_data.keys())[0], bytes):
                progress_data = {k.decode(): v.decode() for k, v in progress_data.items()}

            return TaskProgress(
                percentage=float(progress_data.get("percentage", 0.0)),
                current_step=progress_data.get("current_step", ""),
                total_steps=int(progress_data.get("total_steps", 1)),
                current_step_number=int(progress_data.get("current_step_number", 1)),
                message=progress_data.get("message", ""),
                processed_items=int(progress_data.get("processed_items", 0)),
                total_items=int(progress_data.get("total_items", 0)) if progress_data.get("total_items") != "0" else None
            )

        except Exception as e:
            logger.error(f"Error getting progress for task {task_id}: {e}", exc_info=True)
            raise DeepCrawlError(
                f"Failed to get progress for task {task_id}: {e}",
                ErrorCode.QUEUE_ERROR
            )

    async def get_task_history(self, task_id: str, limit: int = 50) -> List[TaskStatusHistory]:
        """
        Get status change history for a task.

        Args:
            task_id: ID of the task
            limit: Maximum number of history entries to return

        Returns:
            List of TaskStatusHistory objects in reverse chronological order
        """
        try:
            history_data = await self.redis.lrange(
                self.TASK_HISTORY_KEY.format(task_id=task_id),
                0, limit - 1
            )

            history = []
            for entry_json in history_data:
                if isinstance(entry_json, bytes):
                    entry_json = entry_json.decode()

                try:
                    entry = TaskStatusHistory.model_validate_json(entry_json)
                    history.append(entry)
                except json.JSONDecodeError:
                    logger.warning(f"Malformed history entry for task {task_id}: {entry_json}")
                    continue

            return history

        except Exception as e:
            logger.error(f"Error getting history for task {task_id}: {e}", exc_info=True)
            raise DeepCrawlError(
                f"Failed to get history for task {task_id}: {e}",
                ErrorCode.QUEUE_ERROR
            )

    async def get_tasks_by_status(self, status: TaskStatus, limit: int = 100) -> List[str]:
        """
        Get task IDs with a specific status.

        Args:
            status: Status to filter by
            limit: Maximum number of task IDs to return

        Returns:
            List of task IDs
        """
        try:
            task_ids = await self.redis.sscan_iter(
                self.STATUS_INDEX_KEY.format(status=status.value)
            )

            result = []
            async for task_id in task_ids:
                if isinstance(task_id, bytes):
                    task_id = task_id.decode()
                result.append(task_id)

                if len(result) >= limit:
                    break

            return result

        except Exception as e:
            logger.error(f"Error getting tasks by status {status}: {e}", exc_info=True)
            raise DeepCrawlError(
                f"Failed to get tasks by status {status}: {e}",
                ErrorCode.QUEUE_ERROR
            )

    async def get_worker_tasks(self, worker_id: str) -> List[str]:
        """
        Get task IDs currently assigned to a worker.

        Args:
            worker_id: ID of the worker

        Returns:
            List of task IDs
        """
        try:
            task_ids = await self.redis.smembers(
                self.WORKER_TASKS_KEY.format(worker_id=worker_id)
            )

            return [
                task_id.decode() if isinstance(task_id, bytes) else task_id
                for task_id in task_ids
            ]

        except Exception as e:
            logger.error(f"Error getting tasks for worker {worker_id}: {e}", exc_info=True)
            raise DeepCrawlError(
                f"Failed to get tasks for worker {worker_id}: {e}",
                ErrorCode.QUEUE_ERROR
            )

    async def get_timed_out_tasks(self) -> List[str]:
        """
        Get task IDs that have exceeded their timeout.

        Returns:
            List of timed out task IDs
        """
        try:
            current_timestamp = datetime.utcnow().timestamp()

            # Get tasks with timeout timestamp less than current time
            timed_out_tasks = await self.redis.zrangebyscore(
                self.TASK_TIMEOUT_KEY,
                0,
                current_timestamp
            )

            return [
                task_id.decode() if isinstance(task_id, bytes) else task_id
                for task_id in timed_out_tasks
            ]

        except Exception as e:
            logger.error(f"Error getting timed out tasks: {e}", exc_info=True)
            raise DeepCrawlError(
                f"Failed to get timed out tasks: {e}",
                ErrorCode.QUEUE_ERROR
            )

    async def cleanup_completed_tasks(self, older_than_hours: int = 24) -> int:
        """
        Clean up status tracking data for old completed tasks.

        Args:
            older_than_hours: Remove data for tasks completed longer than this many hours ago

        Returns:
            Number of tasks cleaned up
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=older_than_hours)
            cleaned_count = 0

            # Get completed, failed, and cancelled tasks
            for status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                task_ids = await self.get_tasks_by_status(status, limit=1000)

                for task_id in task_ids:
                    status_info = await self.get_task_status(task_id)
                    if not status_info or not status_info.get("completed_at"):
                        continue

                    try:
                        completed_at = datetime.fromisoformat(status_info["completed_at"])
                        if completed_at < cutoff_time:
                            await self._cleanup_task_data(task_id, status)
                            cleaned_count += 1
                    except (ValueError, TypeError):
                        # If we can't parse the timestamp, skip this task
                        continue

            logger.info(f"Cleaned up tracking data for {cleaned_count} completed tasks")
            return cleaned_count

        except Exception as e:
            logger.error(f"Error during cleanup: {e}", exc_info=True)
            raise DeepCrawlError(f"Failed to cleanup completed tasks: {e}", ErrorCode.QUEUE_ERROR)

    async def _cleanup_task_data(self, task_id: str, status: TaskStatus) -> None:
        """Clean up all tracking data for a specific task."""
        pipe = self.redis.pipeline()

        # Remove status data
        pipe.delete(self.TASK_STATUS_KEY.format(task_id=task_id))
        pipe.delete(self.TASK_HISTORY_KEY.format(task_id=task_id))
        pipe.delete(self.TASK_PROGRESS_KEY.format(task_id=task_id))

        # Remove from status index
        pipe.srem(self.STATUS_INDEX_KEY.format(status=status.value), task_id)

        # Remove from timeout tracking
        pipe.zrem(self.TASK_TIMEOUT_KEY, task_id)

        await pipe.execute()