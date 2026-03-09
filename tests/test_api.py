import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from src.deepcrawl_chat.api.main import app
import src.deepcrawl_chat.api.v1.endpoints.chat as chat_endpoints

client = TestClient(app)

class TestChatAPI:

    def test_chat_no_urls(self):
        """Test the /chat endpoint with no URLs."""
        response = client.post(
            "/chat",
            json={"query": "What is DeepCrawl?", "urls": []}
        )

        assert response.status_code == 400, response.json()
        assert "URLs must be provided" in response.json()["detail"]

    @patch('src.deepcrawl_chat.api.main.DocumentLoader')
    @patch('src.deepcrawl_chat.api.main.DeepCrawlTextSplitter', create=True)
    @patch('src.deepcrawl_chat.api.main.get_embeddings_model')
    @patch('src.deepcrawl_chat.api.main.get_or_create_vectorstore')
    @patch('src.deepcrawl_chat.api.main.create_chat_chain')
    def test_chat_with_urls(self, mock_chain, mock_vectorstore, mock_embeddings, mock_splitter, mock_loader):
        """Test the /chat endpoint with URLs."""
        from langchain_core.documents import Document

        # Configure mocks
        mock_loader_instance = MagicMock()
        doc_mock = Document(page_content="doc1", metadata={"source": "url"})
        mock_loader_instance.load_from_urls.return_value = [doc_mock]
        mock_loader.return_value = mock_loader_instance

        mock_splitter_instance = MagicMock()
        chunk_mock = Document(page_content="chunk1", metadata={"source": "url"})
        mock_splitter_instance.split_documents.return_value = [chunk_mock]
        mock_splitter.return_value = mock_splitter_instance

        mock_embeddings.return_value = "embeddings"
        mock_vectorstore.return_value = "vectorstore"

        mock_chain_instance = MagicMock()
        mock_chain_instance.invoke.return_value = {
            "answer": "This is a test answer",
            "context": [
                Document(page_content="chunk1", metadata={"source": "https://example.com/page1"})
            ]
        }
        mock_chain.return_value = mock_chain_instance

        response = client.post(
            "/chat",
            json={"query": "What is DeepCrawl?", "urls": ["https://example.com"]}
        )

        assert response.status_code == 200, response.json()
        assert response.json()["answer"] == "This is a test answer"
        assert response.json()["sources"] == ["https://example.com/page1"]

        mock_loader_instance.load_from_urls.assert_called_once_with(["https://example.com"])
        mock_splitter_instance.split_documents.assert_called_once()
        mock_vectorstore.assert_called_once_with([chunk_mock], "embeddings")
        mock_chain.assert_called_once_with("vectorstore")
        mock_chain_instance.invoke.assert_called_once_with({"input": "What is DeepCrawl?"}) 