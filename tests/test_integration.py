import pytest
from unittest.mock import patch, MagicMock
import os
import tempfile

from src.deep_crawler.integration import CrawlRAGPipeline

class TestCrawlRAGPipeline:

    def test_init(self):
        """Test the initialization of CrawlRAGPipeline."""
        pipeline = CrawlRAGPipeline(
            output_dir="test_output",
            vector_store_dir="test_vectors",
            chunk_size=1000,
            chunk_overlap=50
        )

        assert pipeline.output_dir == "test_output"
        assert pipeline.vector_store_dir == "test_vectors"
        assert pipeline.chunk_size == 1000
        assert pipeline.chunk_overlap == 50

        # Clean up
        os.rmdir("test_output")
        os.rmdir("test_vectors")

    @patch('src.deep_crawler.integration.CrawlResultLoader')
    @patch('src.deep_crawler.integration.TextSplitter')
    @patch('src.deep_crawler.integration.get_embeddings_model')
    @patch('src.deep_crawler.integration.get_or_create_vectorstore')
    def test_index_crawl_results(self, mock_vectorstore, mock_embeddings,
                                 mock_splitter, mock_loader, test_csv_file):
        """Test indexing crawl results."""
        # Configure mocks
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = ["doc1", "doc2"]
        mock_loader.return_value = mock_loader_instance

        mock_splitter_instance = MagicMock()
        mock_splitter_instance.split_documents.return_value = ["chunk1", "chunk2", "chunk3"]
        mock_splitter.return_value = mock_splitter_instance

        mock_embeddings.return_value = "embeddings_model"

        mock_vectorstore.return_value = "vectorstore_instance"

        # Create pipeline and test indexing
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = CrawlRAGPipeline(vector_store_dir=temp_dir)
            result = pipeline.index_crawl_results(test_csv_file)

            # Check that correct methods were called
            mock_loader.assert_called_once_with(test_csv_file, url_type="page")
            mock_loader_instance.load.assert_called_once()
            mock_splitter.assert_called_once_with(chunk_size=5000, chunk_overlap=100)
            mock_splitter_instance.split_documents.assert_called_once_with(["doc1", "doc2"])
            mock_embeddings.assert_called_once()
            mock_vectorstore.assert_called_once()

            assert result is not None
