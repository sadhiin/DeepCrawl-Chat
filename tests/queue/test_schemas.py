import pytest
from pydantic import ValidationError

from src.deepcrawl_chat.queue.schemas import Task, CrawlURLPayload, ProcessContentPayload, TaskPriority

def test_task_creation_default_id_attempts():
    payload_data = {"key": "value"}
    task = Task(task_type="TEST_TYPE", payload=payload_data)
    assert isinstance(task.task_id, str)
    assert len(task.task_id) > 0 # Basic check for UUID-like string
    assert task.task_type == "TEST_TYPE"
    assert task.payload == payload_data
    assert task.priority == TaskPriority.NORMAL  # Default priority
    assert task.attempts == 0
    assert task.error_message is None

def test_task_creation_custom_values():
    task = Task(
        task_id="custom_id_123",
        task_type="CUSTOM_TYPE",
        payload={"info": "details"},
        priority=TaskPriority.HIGH,
        attempts=2,
        error_message="Previous error"
    )
    assert task.task_id == "custom_id_123"
    assert task.task_type == "CUSTOM_TYPE"
    assert task.payload == {"info": "details"}
    assert task.priority == TaskPriority.HIGH
    assert task.attempts == 2
    assert task.error_message == "Previous error"

def test_task_priority_enum():
    """Test that TaskPriority enum values work correctly."""
    for priority in TaskPriority:
        task = Task(task_type="TEST", payload={}, priority=priority)
        assert task.priority == priority

# --- Tests for CrawlURLPayload ---

def test_crawl_url_payload_valid():
    payload = CrawlURLPayload(url="http://example.com", depth=1, force_recrawl=True)
    assert str(payload.url) == "http://example.com/"
    assert payload.depth == 1
    assert payload.force_recrawl is True

def test_crawl_url_payload_default_values():
    payload = CrawlURLPayload(url="https://another.example.com/path?query=test")
    assert str(payload.url) == "https://another.example.com/path?query=test"
    assert payload.depth == 0 # Default
    assert payload.force_recrawl is False # Default

def test_crawl_url_payload_invalid_url():
    with pytest.raises(ValidationError) as excinfo:
        CrawlURLPayload(url="not_a_valid_url", depth=0)
    assert "url" in str(excinfo.value).lower() # Check that 'url' field is mentioned in error
    # Example of how to check specific error messages if needed:
    # errors = excinfo.value.errors()
    # assert any(e['type'] == 'url_scheme' for e in errors)

def test_crawl_url_payload_invalid_depth():
    with pytest.raises(ValidationError) as excinfo:
        CrawlURLPayload(url="http://example.com", depth=-1)
    assert "depth" in str(excinfo.value).lower()
    # errors = excinfo.value.errors()
    # assert any(e['type'] == 'greater_than_equal' and e['loc'][0] == 'depth' for e in errors)


# --- Tests for ProcessContentPayload ---

def test_process_content_payload_valid():
    payload = ProcessContentPayload(
        document_id="doc_789",
        raw_content_path="/data/files/doc_789.txt",
        source_url="http://crawl.example.com/page1"
    )
    assert payload.document_id == "doc_789"
    assert payload.raw_content_path == "/data/files/doc_789.txt"
    assert str(payload.source_url) == "http://crawl.example.com/page1"

def test_process_content_payload_missing_fields():
    with pytest.raises(ValidationError) as excinfo:
        ProcessContentPayload(document_id="doc1") # Missing raw_content_path and source_url
    errors = excinfo.value.errors()
    assert any(e['loc'] == ('raw_content_path',) and e['type'] == 'missing' for e in errors)
    assert any(e['loc'] == ('source_url',) and e['type'] == 'missing' for e in errors)

def test_process_content_payload_invalid_url():
    with pytest.raises(ValidationError) as excinfo:
        ProcessContentPayload(
            document_id="doc_failed_url",
            raw_content_path="/path/to/it",
            source_url="ftp://invalid.url.scheme"
        )
    errors = excinfo.value.errors()
    assert any(e['loc'] == ('source_url',) and 'url_scheme' in e['type'] for e in errors)


# --- Example of how these payloads integrate with Task model ---

def test_task_with_crawl_payload():
    crawl_payload_data = CrawlURLPayload(url="http://test.com", depth=3)
    task = Task(task_type="CRAWL", payload=crawl_payload_data.model_dump(), priority=TaskPriority.HIGH)
    assert task.task_type == "CRAWL"
    assert task.priority == TaskPriority.HIGH
    assert task.payload["url"] == str(crawl_payload_data.url)
    assert task.payload["depth"] == 3
    # Re-validate payload from task to ensure it's still valid
    validated_payload_from_task = CrawlURLPayload(**task.payload)
    assert validated_payload_from_task.url == crawl_payload_data.url

def test_task_with_process_payload():
    process_payload_data = ProcessContentPayload(
        document_id="proc_id_001",
        raw_content_path="s3://bucket/item.dat",
        source_url="http://origin.site/article/001"
    )
    task = Task(task_type="PROCESS", payload=process_payload_data.model_dump(), priority=TaskPriority.URGENT)
    assert task.task_type == "PROCESS"
    assert task.priority == TaskPriority.URGENT
    assert task.payload["document_id"] == "proc_id_001"
    # Re-validate
    validated_payload_from_task = ProcessContentPayload(**task.payload)
    assert validated_payload_from_task.document_id == process_payload_data.document_id
    assert validated_payload_from_task.source_url == process_payload_data.source_url