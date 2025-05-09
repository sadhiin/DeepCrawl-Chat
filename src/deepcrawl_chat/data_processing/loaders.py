from deepcrawl_chat.utils import create_logger
from langchain_community.document_loaders import UnstructuredURLLoader, WebBaseLoader
from deepcrawl_chat.utils import DocumentLoadingError

logger = create_logger()

class DocumentLoader:
    def __init__(self, loader_type="unstructured"):
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

            raise DocumentLoadingError(f"Failed to load documents: {str(e)}")