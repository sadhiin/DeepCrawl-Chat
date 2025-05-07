import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import aiohttp
from bs4 import BeautifulSoup
import asyncio

from deep_crawler import (
    CrawlConfig,
    WebCrawler,
    UrlUtils,
    RobotsTxtManager,
    LinkCategory
)

class TestUrlUtils:

    def test_is_valid_url(self):
        """Test URL validation."""
        assert UrlUtils.is_valid_url("https://example.com")
        assert UrlUtils.is_valid_url("http://example.com/path?query=1")
        assert not UrlUtils.is_valid_url("not a url")
        assert not UrlUtils.is_valid_url("example.com")  # No scheme

    def test_get_base_domain(self):
        """Test base domain extraction."""
        assert UrlUtils.get_base_domain("https://langchain.com") == "langchain.com"
        assert UrlUtils.get_base_domain("python.langchain.com") == "python.langchain.com"

    def test_normalize_url(self):
        """Test URL normalization."""
        assert UrlUtils.normalize_url("https://example.com/") == "https://example.com"
        assert UrlUtils.normalize_url("https://example.com/path/") == "https://example.com/path"
        assert UrlUtils.normalize_url("https://example.com/path#fragment") == "https://example.com/path"

    def test_categorize_url(self):
        """Test URL categorization."""
        assert UrlUtils.categorize_url("https://example.com/doc.pdf") == LinkCategory.PDF
        assert UrlUtils.categorize_url("https://example.com/image.jpg") == LinkCategory.IMAGE
        assert UrlUtils.categorize_url("https://example.com/archive.zip") == LinkCategory.ARCHIVE
        assert UrlUtils.categorize_url("https://example.com/page.html") == LinkCategory.PAGE
        assert UrlUtils.categorize_url("https://example.com/") == LinkCategory.PAGE

class TestWebCrawler:

    def test_init(self):
        """Test crawler initialization."""
        config = CrawlConfig(
            start_url="https://example.com",
            output_file="test.csv",
            max_depth=3
        )
        crawler = WebCrawler(config)

        assert crawler.config == config
        assert crawler.base_domain == "example.com"
        assert len(crawler.visited_urls) == 0
        assert len(crawler.urls_to_visit) == 1
        assert crawler.urls_to_visit[0] == ("https://example.com", 0)

    @pytest.mark.asyncio
    @patch('deep_crawler.aiohttp.ClientSession')
    async def test_process_url(self, mock_session):
        """Test URL processing."""
        # Setup mocks
        mock_session_instance = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.text = AsyncMock(return_value="<html><body><a href='page1.html'>Link</a></body></html>")
        mock_session_instance.get.return_value.__aenter__.return_value = mock_response
        mock_session.return_value = mock_session_instance

        # Create crawler
        config = CrawlConfig(start_url="https://example.com")
        crawler = WebCrawler(config)

        # Call method
        semaphore = asyncio.Semaphore(1)
        await crawler.process_url(
            "https://example.com/test",
            0,
            mock_session_instance,
            semaphore
        )

        # Assertions
        assert "https://example.com/test" in crawler.visited_urls
        assert LinkCategory.PAGE in crawler.discovered_links
        assert "https://example.com/test" in crawler.discovered_links[LinkCategory.PAGE]
        mock_session_instance.get.assert_called_once_with("https://example.com/test") 