# services/crawler_service.py
from typing import AsyncIterator
import asyncio
from redis import Redis
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings

class CrawlerService:
    def __init__(self):
        self.redis = Redis()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = Chroma(
            embedding_function=self.embeddings,
            persist_directory="./data/vectorstore"
        )

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
            crawler = WebCrawler(CrawlConfig(
                start_url=str(crawl_request.url),
                max_depth=crawl_request.max_depth
            ))

            async for page in crawler.crawl_streaming():
                # Process page immediately
                await self._process_page(crawl_request.chat_id, page)

                # Update status
                self.redis.hincrby(
                    f"crawl:{crawl_request.chat_id}",
                    "pages_crawled",
                
                )

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

    async def _process_page(self, chat_id: str, page_content: str):
        """Process a single page and add to vector store"""
        # Split text into chunks
        chunks = self.text_splitter.split_text(page_content)

        # Add to vector store with metadata
        self.vector_store.add_texts(
            texts=chunks,
            metadatas=[{"chat_id": chat_id} for _ in chunks]
        )