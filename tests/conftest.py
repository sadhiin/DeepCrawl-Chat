import os
import pytest
import tempfile
import pandas as pd
from pathlib import Path

@pytest.fixture
def test_csv_data():
    """Create a sample CSV file for testing."""
    data = {
        'Type': ['page', 'page', 'image', 'document', 'page'],
        'URL': [
            'https://python.langchain.com/docs/integrations/chat/',
            'https://python.langchain.com/docs/integrations/retrievers/',
  
        ]
    }
    return data

@pytest.fixture
def test_csv_file(test_csv_data):
    """Create a temporary CSV file with test data."""
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
        pd.DataFrame(test_csv_data).to_csv(temp_file.name, index=False)
        temp_path = temp_file.name

    yield temp_path

    # Clean up
    if os.path.exists(temp_path):
        os.remove(temp_path)

@pytest.fixture
def test_vector_store_dir():
    """Create a temporary directory for vector stores."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture
def mock_document():
    """Create a mock document for testing."""
    from langchain.docstore.document import Document
    return Document(
        page_content="This is a test document for DeepCrawl-Chat.",
        metadata={"source": "https://python.langchain.com/docs/integrations/retrievers"}
    )