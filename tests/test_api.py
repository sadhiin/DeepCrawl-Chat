import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from api.main import app

client = TestClient(app)

class TestChatAPI:

    @patch('api.routes.chat.process_urls_and_index')
    def test_start_crawling_async(self, mock_process):
        """Test the /crawl endpoint with async processing."""
        response = client.post(
            "/crawl",
            json={"urls": ["https://python.langchain.com/docs/integrations/retrievers/"], "max_depth": 2, "async_process": True}
        )

        assert response.status_code == 200
        assert response.json()["status"] == "processing"
        assert "crawl_id" in response.json()
        mock_process.assert_not_called()  # Should be called as background task

    @patch('api.routes.chat.CrawlRAGPipeline')
    def test_start_crawling_sync(self, mock_pipeline):
        """Test the /crawl endpoint with synchronous processing."""
        # Configure mock
        mock_instance = MagicMock()
        mock_instance.crawl_and_index.return_value = "test_store_path"
        mock_pipeline.return_value = mock_instance

        response = client.post(
            "/crawl",
            json={"urls": ["https://langchain.com/"], "max_depth": 2, "async_process": False}
        )

        assert response.status_code == 200
        assert response.json()["status"] == "completed"
        assert response.json()["crawl_id"] == "test_store_path"
        mock_instance.crawl_and_index.assert_called_once_with(
            "https://langchain.com/ ", max_depth=2
        )

    def test_crawl_no_urls(self):
        """Test the /crawl endpoint with no URLs."""
        response = client.post(
            "/crawl",
            json={"urls": [], "max_depth": 2}
        )

        assert response.status_code == 400
        assert "No URLs provided" in response.json()["detail"]

    @patch('api.routes.chat.get_embeddings_model')
    @patch('api.routes.chat.load_vectorstore')
    @patch('api.routes.chat.create_chat_chain')
    def test_chat_with_crawl_id(self, mock_chain, mock_load, mock_embeddings):
        """Test the /chat endpoint with a crawl_id."""
        # Configure mocks
        mock_embeddings.return_value = "embeddings_model"
        mock_load.return_value = "vectorstore"

        mock_chain_instance = MagicMock()
        mock_chain_instance.invoke.return_value = {
            "answer": "This is a test answer",
            "source_documents": [
                MagicMock(metadata={"source": "https://example.com/page1"})
            ]
        }
        mock_chain.return_value = mock_chain_instance

        response = client.post(
            "/chat",
            json={"query": "What is DeepCrawl?", "crawl_id": "test_crawl_id"}
        )

        assert response.status_code == 200
        assert response.json()["answer"] == "This is a test answer"
        assert response.json()["sources"] == ["https://example.com/page1"]

        mock_embeddings.assert_called_once()
        mock_load.assert_called_once_with("test_crawl_id", "embeddings_model")
        mock_chain.assert_called_once_with("vectorstore")
        mock_chain_instance.invoke.assert_called_once_with({"input": "What is DeepCrawl?"})

    def test_chat_no_source(self):
        """Test the /chat endpoint with no source specified."""
        response = client.post(
            "/chat",
            json={"query": "What is DeepCrawl?"}
        )

        assert response.status_code == 400
        assert "Either crawl_id or urls must be provided" in response.json()["detail"] 