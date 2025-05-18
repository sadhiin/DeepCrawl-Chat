import asyncio
import time
import requests
from typing import Optional
import json

def start_crawl(url: str, max_depth: int = 2) -> str:
    """Start crawling a website and return the task ID."""
    response = requests.post(
        "http://localhost:8000/crawl",
        json={
            "url": url,
            "max_depth": max_depth
        }
    )
    response.raise_for_status()
    return response.json()["task_id"]

def get_crawl_status(task_id: str) -> dict:
    """Get the status of a crawling task."""
    response = requests.get(f"http://localhost:8000/crawl/{task_id}/status")
    response.raise_for_status()
    return response.json()

def chat(task_id: str, query: str) -> dict:
    """Send a chat message and get response."""
    response = requests.post(
        "http://localhost:8000/chat",
        json={
            "task_id": task_id,
            "query": query
        }
    )
    response.raise_for_status()
    return response.json()

async def main():
    # Start crawling
    url = "https://www.langchain.com/"
    print(f"Starting crawl for {url}...")
    task_id = start_crawl(url, max_depth=1)
    print(f"Crawl started with task ID: {task_id}")

    # Start interactive chat loop
    print("\nYou can start chatting now! The system will use available data as it's being crawled.")
    print("Type 'exit' to quit or 'status' to check crawl progress.\n")

    while True:
        query = input("You: ").strip()

        if query.lower() == 'exit':
            break
        elif query.lower() == 'status':
            status = get_crawl_status(task_id)
            print(f"\nCrawl Status: {status['status']}")
            print(f"Processed URLs: {status['processed_urls']}/{status['total_urls']}\n")
            continue

        try:
            response = chat(task_id, query)
            print(f"\nAssistant: {response['answer']}")
            if response['sources']:
                print("\nSources:")
                for source in response['sources']:
                    print(f"- {source}")
            if response['crawl_status'] == 'in_progress':
                print("\nNote: Crawling is still in progress. More information may become available.")
            print()
        except Exception as e:
            print(f"\nError: {str(e)}\n")

if __name__ == "__main__":
    asyncio.run(main())

