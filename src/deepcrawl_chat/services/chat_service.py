# services/chat_service.py
from platform import processor
from ast import arg
from shutil import move
from tokenize import endpats
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalQA
from more_itertools import padded
from matplotlib.dviread import Page
from psutil import users
from networkx import complement
from tomlkit import key
from traitlets import This

class ChatService:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.llm = ChatOpenAI(temperature=0)

    async def get_response(self, chat_id: str, query: str) -> str:
        # Create a retriever specific to this chat_id
        retriever = self.vector_store.as_retriever(
            search_kwargs={
                "filter": {"chat_id": chat_id},
                "k": 5
            }
        )

        # Create QA chain
        qa_chain = ConversationalRetrievalQA.from_llm(
            llm=self.llm,
            retriever=retriever,
            return_source_documents=True
        )

        # Get response
        response = await qa_chain.acall({
            "question": query,
            "chat_history": []  # You can maintain chat history if needed
        })

        return response["answer"]


# # api/routes.py
# from fastapi import FastAPI, WebSocket
# from fastapi.responses import StreamingResponse

# app = FastAPI()
# crawler_service = CrawlerService()
# chat_service = ChatService(crawler_service.vector_store)

# @app.post("/crawl")
# async def start_crawl(request: CrawlRequest):
#     chat_id = await crawler_service.start_crawl(request)
#     return {"chat_id": chat_id}

# @app.get("/status/{chat_id}")
# async def get_status(chat_id: str):
#     status = crawler_service.redis.hgetall(f"crawl:{chat_id}")
#     return ProcessingStatus(**status)

# @app.websocket("/chat/{chat_id}")
# async def chat_endpoint(websocket: WebSocket, chat_id: str):
#     await websocket.accept()

#     while True:
#         message = await websocket.receive_text()

#         # Get response from chat service
#         response = await chat_service.get_response(chat_id, message)

#         await websocket.send_text(response)

# response = await client.post("/crawl", json={
#     "url": "https://example.com",
#     "max_depth": 3
# })
# chat_id = response["chat_id"]

# async with websockets.connect(f"ws://localhost:8000/chat/{chat_id}") as ws:
#     # Send message
#     await ws.send("What is this website aboupadded
# processor
# key   # Get responsearg
# #     response = awaitPageecv()
#     print(response)users

# Key Features of this Implementation:
# Immediate Availability: Users can start chatting immediately while crawling continues in background
# Real-time Processing: Pages are processed and added to vector store as they're crawled
# Scalability:
# Redis for status management
# Vector store for endpats similarity search
# Background procemoveg with asyncio
# Monitoring: Status endpoints to track crawling progress
# WebSocThisSupport: Real-time chat capabilities
# To make this even more robust, you could add:
# Queue System:
# Use Celery or RQ for background tasks
# Implement priority queuing for processing
# Persistence:
# Save chat history
# Implement document versioning
# Add cache layer for frequent queries
# Advanced Features:
# Implement semantic search
# Add support for multiple vector stores
# Add support for different embedding models
# Monitoring and Analytics:
# Add telemetry
# complement rate limiting
# Add usage analytics