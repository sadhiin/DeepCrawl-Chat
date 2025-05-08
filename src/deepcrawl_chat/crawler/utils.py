import re
from urllib.parse import urlparse, urlunparse
from typing import Dict, Set
import aiohttp
from aiohttp import ClientTimeout
import logging
from urllib.robotparser import RobotFileParser

from deepcrawl_chat.crawler.schema import LinkCategory

class UrlUtils:
    """URL utility functions."""

    @staticmethod
    def is_valid_url(url: str) -> bool:
        """Check if URL is valid."""
        parsed = urlparse(url)
        return bool(parsed.netloc) and bool(parsed.scheme)

    @staticmethod
    def get_base_domain(url: str) -> str:
        """Extract base domain from URL."""
        parsed = urlparse(url)
        return ".".join(parsed.netloc.split(".")[-2:])

    @staticmethod
    def normalize_url(url: str) -> str:
        """Normalize URL to avoid duplicates."""
        parsed = urlparse(url)
        path = parsed.path.rstrip('/') or '/'
        return urlunparse((
            parsed.scheme,
            parsed.netloc,
            path,
            parsed.params,
            parsed.query,
            ''  # Remove fragments
        ))

    @staticmethod
    def categorize_url(url: str) -> str:
        """Categorize URL by its file type."""
        lower_url = url.lower()
        if re.search(r"\.pdf$", lower_url):
            return LinkCategory.PDF
        elif re.search(r"\.(jpg|jpeg|png|gif|svg|webp|bmp|ico)$", lower_url):
            return LinkCategory.IMAGE
        elif re.search(r"\.(zip|gz|tar|rar|7z)$", lower_url):
            return LinkCategory.ARCHIVE
        elif re.search(r"\.(mp4|avi|mov|wmv|flv|mkv)$", lower_url):
            return LinkCategory.VIDEO
        elif re.search(r"\.(mp3|wav|ogg|flac|aac)$", lower_url):
            return LinkCategory.AUDIO
        elif re.search(r"\.(doc|docx|xls|xlsx|ppt|pptx)$", lower_url):
            return LinkCategory.DOCUMENT
        else:
            return LinkCategory.PAGE


class RobotsTxtManager:
    """Manages robots.txt parsing and checking."""

    def __init__(self, user_agent: str):
        self.user_agent = user_agent
        self.parsers: Dict[str, RobotFileParser] = {}
        self.checked_domains: Set[str] = set()

    async def can_fetch(self, session: aiohttp.ClientSession, url: str) -> bool:
        """Check if a URL can be fetched according to robots.txt rules."""
        parsed_url = urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"

        if base_url not in self.parsers:
            robots_url = f"{base_url}/robots.txt"
            parser = RobotFileParser(robots_url)

            try:
                async with session.get(robots_url, timeout=ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        content = await response.text()
                        parser.parse(content.splitlines())
                    else:
                        # No robots.txt or can't access it, assume all is allowed
                        parser.allow_all = True
            except Exception as e:
                logging.debug(f"Error fetching robots.txt from {base_url}: {e}")
                parser.allow_all = True

            self.parsers[base_url] = parser

        return self.parsers[base_url].can_fetch(self.user_agent, url)
