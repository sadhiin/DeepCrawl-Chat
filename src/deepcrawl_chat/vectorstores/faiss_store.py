import os
import hashlib
import asyncio
from typing import List, Optional, Dict, Any, Union, Tuple, Callable
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_core.embeddings import Embeddings
import logging

logger = logging.getLogger(__name__)

class FAISSVectorStore:
    """Class to manage FAISS vector stores with support for updates and async operations.
    source: https://python.langchain.com/docs/integrations/vectorstores/faiss_async/
    
    """
    
    def __init__(self, embeddings_model: Embeddings, cache_dir: str = "vector_stores"):
        """Initialize the FAISS vector store manager.
        
        Args:
            embeddings_model: The embedding model to use
            cache_dir: Directory for storing vector indices
        """
        self.embeddings_model = embeddings_model
        self.cache_dir = cache_dir
        self.vectorstore = None
        os.makedirs(cache_dir, exist_ok=True)
    
    def _generate_store_path(self, documents: List[Document]) -> str:
        """Generate a unique path based on document content."""
        content_hash = hashlib.md5(str([doc.page_content for doc in documents]).encode()).hexdigest()
        return os.path.join(self.cache_dir, content_hash)
    
    def load_or_create(self, documents: List[Document]) -> FAISS:
        """Load vector store from cache or create a new one."""
        store_path = self._generate_store_path(documents)
        
        # Try to load from cache first
        if os.path.exists(store_path):
            try:
                self.vectorstore = FAISS.load_local(store_path, self.embeddings_model)
                logger.info(f"Loaded vector store from {store_path}")
                return self.vectorstore
            except Exception as e:
                logger.warning(f"Error loading cached vector store: {e}")
        
        # Create new vector store
        self.vectorstore = FAISS.from_documents(documents, self.embeddings_model)
        
        # Save to cache
        self.vectorstore.save_local(store_path)
        logger.info(f"Created and saved new vector store at {store_path}")
        
        return self.vectorstore
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add new documents to the existing vector store."""
        if not self.vectorstore:
            self.load_or_create(documents)
            return
        
        self.vectorstore.add_documents(documents)
        
        # Update cache
        store_path = self._generate_store_path(documents)
        self.vectorstore.save_local(store_path)
        logger.info(f"Updated vector store at {store_path}")
    
    def add_document(self, document: Document) -> None:
        """Add a single document to the vector store."""
        self.add_documents([document])
    
    async def load_or_create_async(self, documents: List[Document]) -> FAISS:
        """Async version of load_or_create."""
        store_path = self._generate_store_path(documents)
        
        # Try to load from cache first
        if os.path.exists(store_path):
            try:
                self.vectorstore = await FAISS.load_local(store_path, self.embeddings_model, asynchronous=True)
                logger.info(f"Loaded vector store async from {store_path}")
                return self.vectorstore
            except Exception as e:
                logger.warning(f"Error loading cached vector store async: {e}")
        
        # Create new vector store asynchronously
        self.vectorstore = await FAISS.afrom_documents(documents, self.embeddings_model)
        
        # Save to cache
        await asyncio.to_thread(self.vectorstore.save_local, store_path)
        logger.info(f"Created and saved new async vector store at {store_path}")
        
        return self.vectorstore
    
    async def add_documents_async(self, documents: List[Document]) -> None:
        """Async version of add_documents."""
        if not self.vectorstore:
            await self.load_or_create_async(documents)
            return
        
        await asyncio.to_thread(self.vectorstore.add_documents, documents)
        
        # Update cache
        store_path = self._generate_store_path(documents)
        await asyncio.to_thread(self.vectorstore.save_local, store_path)
        logger.info(f"Updated vector store async at {store_path}")
    
    async def add_document_async(self, document: Document) -> None:
        """Async version of add_document."""
        await self.add_documents_async([document])
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Perform a similarity search on the vector store."""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Call load_or_create first.")
        
        return self.vectorstore.similarity_search(query, k=k)
    
    async def similarity_search_async(self, query: str, k: int = 4) -> List[Document]:
        """Async version of similarity search."""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Call load_or_create_async first.")
        
        return await self.vectorstore.asimilarity_search(query, k=k)
    
    def similarity_search_with_score(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        """Perform similarity search with scores."""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Call load_or_create first.")
            
        return self.vectorstore.similarity_search_with_score(query, k=k)
    
    async def similarity_search_with_score_async(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        """Async version of similarity search with scores."""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Call load_or_create_async first.")
            
        return await self.vectorstore.asimilarity_search_with_score(query, k=k)
    
    async def similarity_search_by_vector_async(self, embedding_vector, k: int = 4) -> List[Document]:
        """Async version of similarity search by vector."""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Call load_or_create_async first.")
            
        return await self.vectorstore.asimilarity_search_by_vector(embedding_vector, k=k)
    
    def merge_from(self, other_store: "FAISSVectorStore") -> None:
        """Merge another vector store into this one."""
        if not self.vectorstore:
            self.vectorstore = other_store.vectorstore
            return
            
        if other_store.vectorstore:
            self.vectorstore.merge_from(other_store.vectorstore)
    
    @classmethod
    async def from_texts_async(cls, texts: List[str], embeddings_model: Embeddings, 
                               cache_dir: str = "vector_stores", metadatas: Optional[List[Dict[str, Any]]] = None) -> "FAISSVectorStore":
        """Create a vector store from texts asynchronously."""
        store = cls(embeddings_model, cache_dir)
        store.vectorstore = await FAISS.afrom_texts(texts, embeddings_model, metadatas=metadatas)
        return store


async def get_or_create_vectorstore_async(documents: List[Document], 
                                         embeddings_model: Embeddings, 
                                         cache_dir: str = "vector_stores") -> FAISS:
    """Async version to get vector store from cache or create a new one.
    
    Args:
        documents: List of documents to embed
        embeddings_model: The embedding model to use
        cache_dir: Directory for storing vector indices
        
    Returns:
        The FAISS vector store
    """
    store = FAISSVectorStore(embeddings_model, cache_dir)
    return await store.load_or_create_async(documents)


def get_or_create_vectorstore(documents: List[Document], 
                             embeddings_model: Embeddings, 
                             cache_dir: str = "vector_stores") -> FAISS:
    """Get vector store from cache or create a new one (legacy function).
    
    Args:
        documents: List of documents to embed
        embeddings_model: The embedding model to use
        cache_dir: Directory for storing vector indices
        
    Returns:
        The FAISS vector store
    """
    store = FAISSVectorStore(embeddings_model, cache_dir)
    return store.load_or_create(documents)