from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Dict

from deepcrawl_chat.crawler.WebCrawler import WebCrawler
from deepcrawl_chat.schemas.crawling_schema import CrawlConfig
from deepcrawl_chat.data_processing.loaders import DocumentLoader
from deepcrawl_chat.data_processing.processors import DeepCrawlTextSplitter
from deepcrawl_chat.vectorstores.faiss_store import FaissVectorStore
from deepcrawl_chat.chains.retrieval import create_chat_chain
from deepcrawl_chat.utils import create_logger

logger = create_logger()

app = FastAPI(title="DeepCrawl Chat API")

# Store crawling tasks status
crawling_tasks: Dict[str, str] = {}

class CrawlRequest(BaseModel):
    url: HttpUrl
    max_depth: int = 2
    concurrency: int = 5
    respect_robots_txt: bool = True
    user_agent: str = "DeepCrawlBot/1.0"

class CrawlResponse(BaseModel):
    task_id: str
    status: str
    message: str

class ChatRequest(BaseModel):
    query: str
    task_id: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[str] = []

async def crawl_website(task_id: str, config: CrawlConfig):
    try:
        crawling_tasks[task_id] = "in_progress"

        # Initialize and run crawler
        crawler = WebCrawler(config)
        discovered_links = await crawler.crawl()

        # Load and process documents
        loader = DocumentLoader()
        documents = loader.load_from_urls(list(discovered_links.get("page", set())))

        # Split documents
        splitter = DeepCrawlTextSplitter()
        chunks = splitter.split_documents(documents)

        # Store in vector store
        vector_store = FaissVectorStore(collection_name=task_id)
        vector_store.add_documents(chunks)

        crawling_tasks[task_id] = "completed"
        logger.info(f"Crawling task {task_id} completed successfully")

    except Exception as e:
        crawling_tasks[task_id] = "failed"
        logger.error(f"Crawling task {task_id} failed: {str(e)}")
        raise

@app.post("/crawl", response_model=CrawlResponse)
async def crawl_endpoint(request: CrawlRequest, background_tasks: BackgroundTasks):
    try:
        # Generate unique task ID
        import uuid
        task_id = str(uuid.uuid4())

        # Create crawl config
        config = CrawlConfig(
            start_url=str(request.url),
            max_depth=request.max_depth,
            concurrency=request.concurrency,
            respect_robots_txt=request.respect_robots_txt,
            user_agent=request.user_agent
        )

        # Start crawling in background
        background_tasks.add_task(crawl_website, task_id, config)

        return CrawlResponse(
            task_id=task_id,
            status="started",
            message="Crawling task started successfully"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/crawl/{task_id}/status")
async def get_crawl_status(task_id: str):
    if task_id not in crawling_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    return {
        "task_id": task_id,
        "status": crawling_tasks[task_id]
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        if request.task_id not in crawling_tasks:
            raise HTTPException(status_code=404, detail="Task not found")

        if crawling_tasks[request.task_id] != "completed":
            raise HTTPException(
                status_code=400,
                detail=f"Crawling task is not completed. Current status: {crawling_tasks[request.task_id]}"
            )

        # Get vector store for this task
        vector_store = FaissVectorStore(collection_name=request.task_id)

        # Create chat chain
        chain = create_chat_chain()

        # Get response using RAG
        context = vector_store.similarity_search(request.query)
        response = chain.invoke({
            "input": request.query,
            "context": "\n".join(doc.page_content for doc in context)
        })

        return ChatResponse(
            answer=response['answer'],
            sources=[doc.metadata.get('source', '') for doc in context]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))