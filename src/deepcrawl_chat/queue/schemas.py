from pydantic import BaseModel, Field
from typing import Dict, Any
import uuid

class Task(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_type: str
    payload: Dict[str, Any]