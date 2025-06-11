from pydantic import BaseModel, Field, HttpUrl
from typing import Dict, Any, Union, Optional, List
from enum import Enum
import uuid
from datetime import datetime

class TaskPriority(str, Enum):
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    URGENT = "URGENT"

class TaskStatus(str, Enum):
    """Comprehensive task status tracking."""
    PENDING = "PENDING"           # Task is waiting to be processed
    QUEUED = "QUEUED"            # Task is in the queue
    ASSIGNED = "ASSIGNED"        # Task has been assigned to a worker
    PROCESSING = "PROCESSING"    # Task is being processed by a worker
    COMPLETED = "COMPLETED"      # Task completed successfully
    FAILED = "FAILED"           # Task failed (after retries if applicable)
    CANCELLED = "CANCELLED"     # Task was cancelled
    RETRYING = "RETRYING"       # Task is being retried after a failure
    PAUSED = "PAUSED"           # Task processing is paused
    TIMEOUT = "TIMEOUT"         # Task timed out

class TaskProgress(BaseModel):
    """Model for tracking task progress and metrics."""
    percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    current_step: str = Field(default="")
    total_steps: int = Field(default=1, ge=1)
    current_step_number: int = Field(default=1, ge=1)
    message: str = Field(default="")
    processed_items: int = Field(default=0, ge=0)
    total_items: Optional[int] = Field(default=None, ge=0)

    @property
    def step_percentage(self) -> float:
        """Calculate percentage based on current step."""
        if self.total_steps <= 0:
            return 0.0
        return (self.current_step_number / self.total_steps) * 100.0

class TaskStatusHistory(BaseModel):
    """Model for tracking task status change history."""
    status: TaskStatus
    timestamp: datetime
    message: str = Field(default="")
    worker_id: Optional[str] = Field(default=None)
    details: Dict[str, Any] = Field(default_factory=dict)

class Task(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_type: str
    payload: Dict[str, Any]
    priority: TaskPriority = Field(default=TaskPriority.NORMAL)
    attempts: int = Field(default=0)
    error_message: str | None = Field(default=None)

    # Status tracking fields
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)
    assigned_worker_id: Optional[str] = Field(default=None)
    progress: TaskProgress = Field(default_factory=TaskProgress)

    # Timeout and retry configuration
    timeout_seconds: Optional[int] = Field(default=None, gt=0)
    max_retries: int = Field(default=3, ge=0)

    def update_status(self, new_status: TaskStatus, message: str = "", worker_id: Optional[str] = None) -> None:
        """Update task status and timestamp."""
        self.status = new_status
        self.updated_at = datetime.utcnow()

        if new_status == TaskStatus.PROCESSING and not self.started_at:
            self.started_at = datetime.utcnow()
        elif new_status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            self.completed_at = datetime.utcnow()

        if worker_id:
            self.assigned_worker_id = worker_id

    def update_progress(
        self,
        percentage: Optional[float] = None,
        current_step: Optional[str] = None,
        current_step_number: Optional[int] = None,
        message: Optional[str] = None,
        processed_items: Optional[int] = None,
        total_items: Optional[int] = None
    ) -> None:
        """Update task progress information."""
        if percentage is not None:
            self.progress.percentage = max(0.0, min(100.0, percentage))
        if current_step is not None:
            self.progress.current_step = current_step
        if current_step_number is not None:
            self.progress.current_step_number = max(1, current_step_number)
        if message is not None:
            self.progress.message = message
        if processed_items is not None:
            self.progress.processed_items = max(0, processed_items)
        if total_items is not None:
            self.progress.total_items = max(0, total_items) if total_items >= 0 else None

        self.updated_at = datetime.utcnow()

    @property
    def is_terminal_status(self) -> bool:
        """Check if the task is in a terminal status."""
        return self.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]

    @property
    def is_active_status(self) -> bool:
        """Check if the task is actively being processed."""
        return self.status in [TaskStatus.ASSIGNED, TaskStatus.PROCESSING]

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate task duration in seconds."""
        if not self.started_at:
            return None
        end_time = self.completed_at or datetime.utcnow()
        return (end_time - self.started_at).total_seconds()

# --- Specific Task Payloads ---

class CrawlURLPayload(BaseModel):
    url: HttpUrl
    depth: int = Field(default=0, ge=0)
    force_recrawl: bool = Field(default=False)
    # Example of further specific fields if needed
    # user_agent: str | None = None
    # metadata: Dict[str, Any] = Field(default_factory=dict)

class ProcessContentPayload(BaseModel):
    document_id: str
    raw_content_path: str
    source_url: HttpUrl
    # Example of further specific fields
    # extraction_rules: List[str] = Field(default_factory=list)

# --- Union type for type hinting if desired, though Task.payload remains Dict[str, Any] for flexibility ---
# This allows for type checking on the client side before creating the generic Task object.
SpecificTaskPayload = Union[CrawlURLPayload, ProcessContentPayload]


# --- Task Type Enum (Optional but recommended for consistency) ---
# from enum import Enum
class TaskType(str, Enum):
    CRAWL_URL = "CRAWL_URL"
    PROCESS_CONTENT = "PROCESS_CONTENT"
    # Add other task types here

# Example of how to use with the Task model:
# validated_crawl_payload = CrawlURLPayload(url="http://example.com", depth=1)
# crawl_task = Task(task_type=TaskType.CRAWL_URL, payload=validated_crawl_payload.model_dump(), priority=TaskPriority.HIGH)

# validated_process_payload = ProcessContentPayload(document_id="doc123", raw_content_path="/path/to/content", source_url="http://example.com")
# process_task = Task(task_type=TaskType.PROCESS_CONTENT, payload=validated_process_payload.model_dump())