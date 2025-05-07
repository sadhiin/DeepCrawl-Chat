from langchain_text_splitters import RecursiveCharacterTextSplitter


class DeepCrawlTextSplitter:
    def __init__(self, chunk_size=5000, chunk_overlap=100):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def split_documents(self, documents):
        return self.splitter.split_documents(documents)


