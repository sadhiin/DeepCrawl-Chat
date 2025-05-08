# models/schema.py
from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Literal
from datetime import datetime

class CrawlRequest(BaseModel):
    url: HttpUrl
    max_depth: int = 3
    chat_id: str

class ChatSession(BaseModel):
    chat_id: str
    website_url: HttpUrl
    created_at: datetime
    status: Literal['crawling', 'processing', 'ready', 'error']

class ChatMessage(BaseModel):
    chat_id: str
    message: str
    timestamp: datetime
    role: Literal['user', 'assistant']

class ProcessingStatus(BaseModel):
    chat_id: str
    pages_crawled: int
    pages_processed: int
    status: str
    estimated_time: Optional[float]