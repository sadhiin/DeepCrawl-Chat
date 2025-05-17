import asyncio
import time
from src.deepcrawl_chat.config import get_config
from src.deepcrawl_chat.crawler.integration import CrawlRAGPipeline

async def main():
    # print(get_config())
    pipeline = CrawlRAGPipeline()
    url = "https://www.langchain.com/"
    
    await pipeline.async_crawl_and_index(url=url, max_depth=1)
    
if __name__ == "__main__":
    asyncio.run(main())

