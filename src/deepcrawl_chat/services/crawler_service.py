# services/crawler_service.py
from calendar import c
from typing import AsyncIterator
import asyncio
from redis import Redis
from deepcrawl_chat.crawler.web_crawler import WebCrawler
from deepcrawl_chat.schemas.crawling_schema import CrawlConfig
from deepcrawl_chat.schemas.service_schema import CrawlRequest
from deepcrawl_chat.crawler.integration import CrawlRAGPipeline

class CrawlerService:
    def __init__(self):
        self.redis = Redis()
        
        

    async def start_crawl(self, crawl_request: CrawlRequest) -> str:
        """Starts a crawl and returns a chat_id"""
        # Initialize crawl status
        self.redis.hset(
            f"crawl:{crawl_request.chat_id}",
            mapping={"status": "crawling", "pages_crawled": 0}
        )

        # Start background crawl
        asyncio.create_task(self._crawl_and_process(crawl_request))

        return crawl_request.chat_id

    async def _crawl_and_process(self, crawl_request: CrawlRequest):
        """Background crawling and processing"""
        try:
            # Use existing WebCrawler but modify to yield results
            crawler_pipeline = CrawlRAGPipeline()
            await crawler_pipeline.crawl_and_index(crawl_request.url, crawl_request.max_depth)

            # Mark as complete
            self.redis.hset(
                f"crawl:{crawl_request.chat_id}",
                "status",
                "ready"
            )

        except Exception as e:
            self.redis.hset(
                f"crawl:{crawl_request.chat_id}",
                "status",
                f"error: {str(e)}"
            )
