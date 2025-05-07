from src.utils.logging import logger
from langchain_community.document_loaders import UnstructuredURLLoader, WebBaseLoader
from config.settings import settings

class DocumentLoader:
    def __init__(self, loader_type=settings.DOCUMENT_LOADER):
        self.loader_type = loader_type

    def load_from_urls(self, urls, max_urls=None):
        if not urls:
            raise ValueError("No URLs provided")

        urls_to_process = urls[:max_urls] if max_urls else urls

        try:
            if self.loader_type == "unstructured":
                loader = UnstructuredURLLoader(urls_to_process)
            elif self.loader_type == "web":
                loader = WebBaseLoader(urls_to_process)
            else:
                raise ValueError(f"Unsupported loader type: {self.loader_type}")

            return loader.load()
        except Exception as e:
            # Log the error with proper context
            logger.error(f"Error loading documents: {str(e)}")
            # Potentially return partial results or retry logic
            raise DocumentLoadingError(f"Failed to load documents: {str(e)}")