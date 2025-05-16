from src.deepcrawl_chat.config import get_config
from src.deepcrawl_chat.crawler.integration import CrawlRAGPipeline

def main():
    # print(get_config())
    pipeline = CrawlRAGPipeline()
    url = "https://google.com"
    
    await pipeline.async_crawl_and_index(url=url, max_depth=1)
    
if __name__ == "__main__":
    main()

