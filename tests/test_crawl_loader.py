import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import tempfile
import os

from src.data.crawl_loader import CrawlResultLoader

class TestCrawlResultLoader:

    def test_init(self):
        """Test the initialization of CrawlResultLoader."""
        loader = CrawlResultLoader(
            csv_path="test.csv",
            url_type="page",
            max_urls=10,
            max_workers=3,
            max_retries=2
        )

        assert loader.csv_path == "test.csv"
        assert loader.url_type == "page"
        assert loader.max_urls == 10
        assert loader.max_workers == 3
        assert loader.max_retries == 2

    def test_load_with_valid_csv(self, test_csv_file):
        """Test loading with a valid CSV file."""
        with patch('src.data.crawl_loader.WebBaseLoader') as mock_loader:
            # Mock the WebBaseLoader to return a document
            mock_instance = MagicMock()
            mock_document = MagicMock()
            mock_document.metadata = {}
            mock_instance.load.return_value = [mock_document]
            mock_loader.return_value = mock_instance

            # Create loader and load documents
            loader = CrawlResultLoader(csv_path=test_csv_file, max_urls=2)
            documents = loader.load()

            # Assertions
            assert len(documents) == 2
            assert mock_loader.call_count == 2
            assert mock_instance.load.call_count == 2

            # Check that metadata was enhanced
            for doc in documents:
                assert doc.metadata["source_type"] == "page"
                assert doc.metadata["crawler_source"] is True

    def test_load_empty_csv(self):
        """Test loading with an empty CSV file."""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
            pd.DataFrame({'Type': [], 'URL': []}).to_csv(temp_file.name, index=False)
            temp_path = temp_file.name

        loader = CrawlResultLoader(csv_path=temp_path)
        documents = loader.load()

        assert documents == []

        # Clean up
        os.remove(temp_path)

    def test_load_nonexistent_csv(self):
        """Test loading with a non-existent CSV file."""
        loader = CrawlResultLoader(csv_path="nonexistent.csv")

        with pytest.raises(Exception):
            loader.load()