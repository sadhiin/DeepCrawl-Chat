from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List

from src.deepcrawl_chat.data_processing.loaders import DocumentLoader
from src.deepcrawl_chat.embeddings.models import get_embeddings_model
from src.deepcrawl_chat.vectorstores.faiss_store import get_or_create_vectorstore
from src.deepcrawl_chat.chains.retrieval import create_chat_chain

app = FastAPI(title="DeepCrawl Chat API")

class ChatRequest(BaseModel):
    query: str
    urls: List[str] = []

class ChatResponse(BaseModel):
    answer: str
    sources: List[str] = []

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        from src.deepcrawl_chat.embeddings.models import get_embeddings_model
        from src.deepcrawl_chat.chains.retrieval import create_chat_chain
        from src.deepcrawl_chat.vectorstores.faiss_store import get_or_create_vectorstore
        from src.deepcrawl_chat.data_processing.processors import DeepCrawlTextSplitter

        vectorstore = None

        if request.urls:
            loader = DocumentLoader()
            documents = loader.load_from_urls(request.urls)
            
            # Process documents
            text_splitter = DeepCrawlTextSplitter(chunk_size=5000, chunk_overlap=100)
            chunks = text_splitter.split_documents(documents)
            
            embeddings_model = get_embeddings_model()
            vectorstore = get_or_create_vectorstore(chunks, embeddings_model)
        else:
            raise HTTPException(status_code=400, detail="URLs must be provided for this endpoint")

        chain = create_chat_chain(vectorstore)
        response = chain.invoke({"input": request.query})

        return ChatResponse(
            answer=response['answer'],
            sources=[doc.metadata.get('source', '') for doc in response.get('context', [])]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))