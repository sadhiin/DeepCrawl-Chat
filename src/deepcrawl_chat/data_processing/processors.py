from langchain_text_splitters import RecursiveCharacterTextSplitter
from .loaders import DocumentLoader

class DeepCrawlTextSplitter:
    def __init__(self, chunk_size=5000, chunk_overlap=100):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def split_documents(self, documents):
        return self.splitter.split_documents(documents)


class DeepCrawlDocumentProcessor:
    def __init__(self):
        pass

    @staticmethod
    def process_document(url:str):
        loader = DocumentLoader()
        documents = loader.load_from_urls([url])
        splitter = DeepCrawlTextSplitter()
        chunks = splitter.split_documents(documents)
        return chunks

    @staticmethod
    def process_documents(urls:list[str]):
        loader = DocumentLoader()
        documents = loader.load_from_urls(urls)
        splitter = DeepCrawlTextSplitter()
        chunks = splitter.split_documents(documents)
        return chunks