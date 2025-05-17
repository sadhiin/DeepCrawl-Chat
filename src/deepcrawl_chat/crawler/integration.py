import os
import asyncio
import time
from typing import List, Optional
from pathlib import Path

# from src.data.crawl_loader import CrawlResultLoader
from deepcrawl_chat.data_processing import DeepCrawlTextSplitter, CrawlResultLoader
from deepcrawl_chat.embeddings import EmbeddingModel
from deepcrawl_chat.vectorstores import FAISSVectorStore
from .web_crawler import WebCrawler
from deepcrawl_chat.schemas.crawling_schema import CrawlConfig

from deepcrawl_chat.utils import create_logger
logger = create_logger(__name__)

class CrawlRAGPipeline:
    """Pipeline for crawling websites and setting up RAG."""

    def __init__(self,
                output_dir: str = "data/crawls",
                vector_store_dir: str = "data/vector_stores",
                chunk_size: int = 5000,
                chunk_overlap: int = 100):
        """
        Initialize the pipeline.

        Args:
            output_dir: Directory for crawl results
            vector_store_dir: Directory for vector stores
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
        """
        self.output_dir = output_dir
        self.vector_store_dir = vector_store_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Create directories if they don't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(vector_store_dir).mkdir(parents=True, exist_ok=True)

    async def async_crawl_and_index(self, url: str, max_depth: int = 3):
        """
        Crawl a website and index the results.

        Args:
            url: URL to crawl
            max_depth: Maximum crawl depth

        Returns:
            Path to the created vector store
        """
        # Generate a filename based on the URL
        from urllib.parse import urlparse
        domain = urlparse(url).netloc.replace(".", "_")
        timestamp = int(time.time())
        crawl_filename = f"{domain}_{timestamp}_crawl_results.csv"
        crawl_path = os.path.join(self.output_dir, crawl_filename)

        # Import and run the crawler
        

        config = CrawlConfig(
            start_url=url,
            output_file=crawl_path,
            max_depth=max_depth
        )
        # TODO: add the crawling and indexing + save in to the vector store and database
        
        crawler = WebCrawler(config)
        await crawler.crawl()
        # crawler.export_links_to_csv()

        # # Index the crawled results
        # return self.index_crawl_results(crawl_path)

    def index_crawl_results(self, csv_path: str, url_type: str = "page") -> str:
        """
        Index crawl results from a CSV file.

        Args:
            csv_path: Path to the crawl results CSV
            url_type: Type of URLs to index

        Returns:
            Path to the created vector store
        """
        logger.info(f"Indexing crawl results from {csv_path}")

        # Load documents from the crawl results
        loader = CrawlResultLoader(csv_path, url_type=url_type)
        documents = loader.load()

        if not documents:
            logger.warning("No documents loaded from crawl results")
            return None

        # Split the documents into chunks
        text_splitter = DeepCrawlTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = text_splitter.split_documents(documents)

        # Get the embedding model
        embeddings_model = EmbeddingModel().get_embeddings_model()
        vectore_store = FAISSVectorStore(embeddings_model, cache_dir=self.vector_store_dir)
        # Generate store identifier
        store_name = os.path.basename(csv_path).replace(".csv", "")
        store_path = os.path.join(self.vector_store_dir, store_name)

        # Create and save the vector store
        vectorstore = vectore_store.load_or_create(chunks)
        vectorstore.save_local(store_path)

        logger.info(f"Created vector store at {store_path} with {len(chunks)} chunks")
        return store_path


async def process_urls_and_index(urls: List[str], max_depth: int = 2):
    """
    Process a list of URLs, crawl them, and create a vector store.

    Args:
        urls: List of URLs to crawl and index
        max_depth: Maximum crawl depth
    """
    pipeline = CrawlRAGPipeline()

    for url in urls:
        try:
            store_path = await pipeline.crawl_and_index(url, max_depth=max_depth)
            logger.info(f"Completed indexing for {url}, vector store at {store_path}")
        except Exception as e:
            logger.error(f"Error processing {url}: {e}")