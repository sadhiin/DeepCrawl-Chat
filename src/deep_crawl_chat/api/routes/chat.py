from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends
from typing import List, Optional
from pydantic import BaseModel
import logging

from src.deep_crawler.integration import process_urls_and_index
from src.chains.retrieval import create_chat_chain
from src.vectorstores.faiss_store import load_vectorstore
from src.embeddings.models import get_embeddings_model

router = APIRouter()
logger = logging.getLogger(__name__)

class CrawlRequest(BaseModel):
    urls: List[str]
    max_depth: int = 2
    async_process: bool = True

class CrawlResponse(BaseModel):
    status: str
    message: str
    crawl_id: Optional[str] = None

class ChatRequest(BaseModel):
    query: str
    crawl_id: Optional[str] = None
    urls: Optional[List[str]] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[str] = []

@router.post("/crawl", response_model=CrawlResponse)
async def start_crawling(request: CrawlRequest, background_tasks: BackgroundTasks):
    """Start crawling and indexing process for one or more URLs."""
    if not request.urls:
        raise HTTPException(status_code=400, detail="No URLs provided")

    # Generate a unique ID for this crawl
    import uuid
    import time
    crawl_id = f"crawl_{int(time.time())}_{uuid.uuid4().hex[:8]}"

    if request.async_process:
        # Process asynchronously
        background_tasks.add_task(process_urls_and_index, request.urls, request.max_depth)
        return CrawlResponse(
            status="processing",
            message=f"Crawling and indexing started for {len(request.urls)} URLs",
            crawl_id=crawl_id
        )
    else:
        # Process synchronously
        try:
            # For simplicity in this example, we just process the first URL synchronously
            pipeline = CrawlRAGPipeline()
            store_path = await pipeline.crawl_and_index(request.urls[0], max_depth=request.max_depth)

            return CrawlResponse(
                status="completed",
                message=f"Crawling and indexing completed for {request.urls[0]}",
                crawl_id=store_path
            )
        except Exception as e:
            logger.error(f"Error during synchronous crawling: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Chat using the RAG system with either existing crawl results or new URLs."""
    try:
        # Determine which vector store to use
        if request.crawl_id:
            # Use an existing vector store
            embeddings = get_embeddings_model()
            vectorstore = load_vectorstore(request.crawl_id, embeddings)
        elif request.urls:
            # Create a temporary vector store for these URLs
            # This is a simplified version - would need more implementation
            raise HTTPException(status_code=501, detail="Direct URL processing not implemented yet")
        else:
            raise HTTPException(status_code=400, detail="Either crawl_id or urls must be provided")

        # Create the retrieval chain
        chain = create_chat_chain(vectorstore)

        # Get response
        response = chain.invoke({"input": request.query})

        return ChatResponse(
            answer=response["answer"],
            sources=[doc.metadata.get("source", "") for doc in response.get("source_documents", [])]
        )
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))