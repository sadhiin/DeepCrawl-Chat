from pydantic import BaseModel, Field, HttpUrl
from typing import Dict, Any, Union
import uuid

class Task(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_type: str
    payload: Dict[str, Any]
    attempts: int = Field(default=0)
    error_message: str | None = Field(default=None)

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
# crawl_task = Task(task_type=TaskType.CRAWL_URL, payload=validated_crawl_payload.model_dump())

# validated_process_payload = ProcessContentPayload(document_id="doc123", raw_content_path="/path/to/content", source_url="http://example.com")
# process_task = Task(task_type=TaskType.PROCESS_CONTENT, payload=validated_process_payload.model_dump())