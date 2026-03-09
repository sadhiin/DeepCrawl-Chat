from src.deepcrawl_chat.utils.logging import logger
from typing import List, Optional, Dict, Any
from langchain_community.document_loaders import WebBaseLoader
from src.deepcrawl_chat.config.config import settings

class DocumentLoader:
    def __init__(self, loader_type=settings.DOCUMENT_LOADER):
        self.loader_type = loader_type
        
    def load_from_urls(self, urls: List[str]) -> List[Any]:
        """Load documents from a list of URLs"""
        logger.info(f"Loading documents from {len(urls)} URLs using {self.loader_type}")
        
        try:
            loader = WebBaseLoader(urls)
            return loader.load()
        except Exception as e:
            # Log the error with proper context
            logger.error(f"Error loading documents: {str(e)}")
            # Potentially return partial results or retry logic
            raise DocumentLoadingError(f"Failed to load documents: {str(e)}")