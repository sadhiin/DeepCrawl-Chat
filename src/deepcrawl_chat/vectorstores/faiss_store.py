import os
import hashlib
from langchain_community.vectorstores import FAISS

def get_or_create_vectorstore(documents, embeddings_model, cache_dir="vector_stores"):
    """Get vector store from cache or create a new one"""

    # Create a hash of the document content for cache identification
    content_hash = hashlib.md5(str([doc.page_content for doc in documents]).encode()).hexdigest()
    store_path = os.path.join(cache_dir, content_hash)

    # Try to load from cache first
    if os.path.exists(store_path):
        try:
            return FAISS.load_local(store_path, embeddings_model)
        except Exception as e:
            print(f"Error loading cached vector store: {e}")

    # Create new vector store
    vectorstore = FAISS.from_documents(documents, embeddings_model)

    # Save to cache
    os.makedirs(cache_dir, exist_ok=True)
    vectorstore.save_local(store_path)

    return vectorstore