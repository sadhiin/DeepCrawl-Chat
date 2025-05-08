from pydantic.dataclasses import dataclass

@dataclass
class CrawlConfig:
    start_url: str
    output_file: str = "crawl_results.csv"
    delay: float = 0.5
    max_depth: int = 5
    timeout: int = 10
    max_retries: int = 3
    concurrency: int = 5
    respect_robots_txt: bool = True
    user_agent: str = "DeepCrawl-Chat/1.0"
    log_level: str = "INFO"


@dataclass
class LinkCategory:
    PAGE:str = "page"
    IMAGE:str = "image"
    PDF:str = "pdf"
    ARCHIVE:str = "archive"
    VIDEO:str = "video"
    AUDIO:str = "audio"
    DOCUMENT:str = "document"
    OTHER:str = "other"