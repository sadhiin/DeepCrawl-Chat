from tqdm import tqdm
import pandas as pd
from typing import List, Optional, Dict, Any
from langchain.docstore.document import Document
from langchain_community.document_loaders import WebBaseLoader
from concurrent.futures import ThreadPoolExecutor, as_completed
from deepcrawl_chat.utils import create_logger

logger = create_logger(__name__)

class CrawlResultLoader:
    """Load documents from crawl results CSV file."""

    def __init__(self,
                csv_path: str,
                url_type: str = "page",
                max_urls: Optional[int] = None,
                max_workers: int = 5,
                max_retries: int = 2):
        """
        Initialize the loader.

        Args:
            csv_path: Path to the crawl results CSV file
            url_type: Type of URLs to load (default: "page")
            max_urls: Maximum number of URLs to load (None for all)
            max_workers: Maximum number of worker threads for parallel loading
            max_retries: Maximum number of retries for failed URL loads
        """
        self.csv_path = csv_path
        self.url_type = url_type
        self.max_urls = max_urls
        self.max_workers = max_workers
        self.max_retries = max_retries

    def load(self) -> List[Document]:
        """Load documents from the crawl results."""
        # Load the CSV file
        try:
            df = pd.read_csv(self.csv_path)
            logger.info(f"Loaded {len(df)} entries from {self.csv_path}")
        except Exception as e:
            logger.error(f"Error loading CSV file {self.csv_path}: {e}")
            return []

        # Filter for the desired URL type
        filtered_df = df[df['Type'] == self.url_type]
        logger.info(f"Found {len(filtered_df)} URLs of type '{self.url_type}'")

        # Get the list of URLs to process
        urls = filtered_df['URL'].tolist()
        if self.max_urls:
            urls = urls[:self.max_urls]
            logger.info(f"Limited to {len(urls)} URLs")

        # Load documents in parallel
        documents = self._load_urls_parallel(urls)
        logger.info(f"Successfully loaded {len(documents)} documents")

        return documents

    def _load_urls_parallel(self, urls: List[str]) -> List[Document]:
        """Load URLs in parallel using thread pool."""
        documents = []
        failed_urls = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all URL loading tasks
            future_to_url = {executor.submit(self._load_single_url, url): url for url in urls}

            # Process results as they complete
            for future in tqdm(as_completed(future_to_url), total=len(urls), desc="Loading documents"):
                url = future_to_url[future]
                try:
                    result = future.result()
                    if result:
                        documents.extend(result)
                    else:
                        failed_urls.append(url)
                except Exception as e:
                    logger.warning(f"Error processing {url}: {e}")
                    failed_urls.append(url)

        # Handle retries for failed URLs
        if failed_urls and self.max_retries > 0:
            logger.info(f"Retrying {len(failed_urls)} failed URLs")
            retry_loader = CrawlResultLoader(
                self.csv_path,
                self.url_type,
                max_urls=None,
                max_workers=self.max_workers,
                max_retries=self.max_retries - 1
            )
            retry_docs = retry_loader._load_urls_parallel(failed_urls)
            documents.extend(retry_docs)

        return documents

    def _load_single_url(self, url: str) -> List[Document]:
        """Load a single URL and return documents."""
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()

            # Enhance metadata
            for doc in docs:
                doc.metadata["source_type"] = self.url_type
                doc.metadata["crawler_source"] = True

            return docs
        except Exception as e:
            logger.warning(f"Failed to load {url}: {e}")
            return []