from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List

from src.data.loaders import DocumentLoader
from src.embeddings.models import get_embeddings_model
from src.vectorstores.faiss_store import get_or_create_vectorstore
from src.chains.retrieval import create_chat_chain

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
        # If URLs are provided, create a new vector store
        if request.urls:
            loader = DocumentLoader()
            documents = loader.load_from_urls(request.urls)

            # Process documents
            # ...

        # Get the retrieval chain
        chain = create_chat_chain()

        # Get response
        response = chain.invoke({"input": request.query})

        return ChatResponse(
            answer=response['answer'],
            sources=[doc.metadata.get('source', '') for doc in response.get('source_documents', [])]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))