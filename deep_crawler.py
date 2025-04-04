#!/usr/bin/env python3
import asyncio
import csv
import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Union
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import aiohttp
from aiohttp import ClientTimeout
from bs4 import BeautifulSoup
import argparse


@dataclass
class CrawlConfig:
    """Configuration for the crawler."""
    start_url: str
    output_file: str = "crawl_results.csv"
    delay: float = 0.5
    max_depth: int = 5
    timeout: int = 10
    max_retries: int = 3
    concurrency: int = 5
    respect_robots_txt: bool = True
    user_agent: str = "DeepCrawler/1.0"
    log_level: str = "INFO"


class LinkCategory:
    """Link categories enum."""
    PAGE = "page"
    IMAGE = "image"
    PDF = "pdf"
    ARCHIVE = "archive"
    VIDEO = "video"
    AUDIO = "audio"
    DOCUMENT = "document"
    OTHER = "other"


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


class WebCrawler:
    """Asynchronous web crawler with improved features."""

    def __init__(self, config: CrawlConfig):
        self.config = config
        self.base_domain = UrlUtils.get_base_domain(config.start_url)
        self.visited_urls: Set[str] = set()
        self.urls_to_visit: List[tuple] = [(UrlUtils.normalize_url(config.start_url), 0)]  # (url, depth)
        self.discovered_links: Dict[str, Set[str]] = defaultdict(set)
        self.robots_manager = RobotsTxtManager(config.user_agent)

        # Configure logging
        logging.basicConfig(
            level=getattr(logging, config.log_level),
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    async def crawl(self) -> Dict[str, Set[str]]:
        """Main crawling method."""
        logging.info(f"Starting crawl at {self.config.start_url} with max depth {self.config.max_depth}")

        async with aiohttp.ClientSession(
            headers={"User-Agent": self.config.user_agent},
            timeout=ClientTimeout(total=self.config.timeout)
        ) as session:
            semaphore = asyncio.Semaphore(self.config.concurrency)
            tasks = []

            # Process URLs until we've visited them all or reached limits
            while self.urls_to_visit:
                url, depth = self.urls_to_visit.pop(0)

                if url in self.visited_urls or depth > self.config.max_depth:
                    continue

                if self.config.respect_robots_txt:
                    can_fetch = await self.robots_manager.can_fetch(session, url)
                    if not can_fetch:
                        logging.info(f"Skipping {url} (disallowed by robots.txt)")
                        continue

                # Add task to process the URL
                task = asyncio.create_task(self.process_url(url, depth, session, semaphore))
                tasks.append(task)

                # Add a small delay between starting tasks
                if self.config.delay > 0:
                    await asyncio.sleep(self.config.delay)

            # Wait for all tasks to complete
            if tasks:
                await asyncio.gather(*tasks)

        return dict(self.discovered_links)

    async def process_url(self, url: str, depth: int, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore):
        """Process a single URL."""
        if url in self.visited_urls:
            return

        self.visited_urls.add(url)

        for attempt in range(self.config.max_retries):
            try:
                async with semaphore:
                    logging.debug(f"Crawling: {url} (depth: {depth}, attempt: {attempt+1})")

                    async with session.get(url) as response:
                        if response.status != 200:
                            logging.warning(f"Failed to fetch {url} - Status {response.status}")
                            continue

                        # Add URL to appropriate category
                        category = UrlUtils.categorize_url(url)
                        self.discovered_links[category].add(url)

                        # Only parse HTML content for extraction
                        content_type = response.headers.get('Content-Type', '').lower()
                        if 'text/html' not in content_type:
                            logging.debug(f"Skipping non-HTML content: {url}")
                            return

                        html_content = await response.text()
                        await self.extract_links(url, html_content, depth)
                        return

            except asyncio.TimeoutError:
                logging.warning(f"Timeout while fetching {url} (attempt {attempt+1})")
            except Exception as e:
                logging.error(f"Error processing {url}: {str(e)}")

        logging.error(f"Failed to process {url} after {self.config.max_retries} attempts")

    async def extract_links(self, url: str, html_content: str, depth: int):
        """Extract links from HTML content."""
        soup = BeautifulSoup(html_content, 'html.parser')
        logging.info(f"Processing: {url} (Found {len(soup.find_all('a'))} links)")

        # Extract regular links
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            self.process_extracted_url(url, href, depth)

        # Extract image links
        for img_tag in soup.find_all('img', src=True):
            src = img_tag['src']
            if src and not src.startswith('data:'):
                absolute_url = urljoin(url, src)
                normalized_url = UrlUtils.normalize_url(absolute_url)
                if UrlUtils.is_valid_url(normalized_url):
                    self.discovered_links[LinkCategory.IMAGE].add(normalized_url)

        # Extract other embedded content
        for tag in soup.find_all(['source', 'video', 'audio', 'iframe', 'embed'], src=True):
            src = tag['src']
            if src and not src.startswith('data:'):
                absolute_url = urljoin(url, src)
                normalized_url = UrlUtils.normalize_url(absolute_url)
                if UrlUtils.is_valid_url(normalized_url):
                    category = UrlUtils.categorize_url(normalized_url)
                    self.discovered_links[category].add(normalized_url)

    def process_extracted_url(self, source_url: str, href: str, depth: int):
        """Process an extracted URL."""
        if not href or href.startswith(('javascript:', 'mailto:', 'tel:')):
            return

        absolute_url = urljoin(source_url, href)
        normalized_url = UrlUtils.normalize_url(absolute_url)

        if not UrlUtils.is_valid_url(normalized_url) or re.match(r".*\s.*", normalized_url):
            return

        # Categorize the link
        category = UrlUtils.categorize_url(normalized_url)
        self.discovered_links[category].add(normalized_url)

        # Queue HTML pages from same domain for crawling
        next_depth = depth + 1
        if (UrlUtils.get_base_domain(normalized_url) == self.base_domain and
            category == LinkCategory.PAGE and
            next_depth <= self.config.max_depth and
            normalized_url not in self.visited_urls and
            (normalized_url, next_depth) not in self.urls_to_visit):

            self.urls_to_visit.append((normalized_url, next_depth))

    def export_links_to_csv(self, filename: Optional[str] = None):
        """Export discovered links to a CSV file."""
        output_path = filename or self.config.output_file

        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Type', 'URL'])

            for link_type, urls in self.discovered_links.items():
                for url in urls:
                    writer.writerow([link_type, url])

        logging.info(f"Links exported to {output_path}")

    def print_summary(self):
        """Print a summary of the crawl results."""
        total_links = sum(len(links) for links in self.discovered_links.values())

        logging.info("\nCrawl Summary:")
        logging.info(f"Total links discovered: {total_links}")
        for link_type, urls in self.discovered_links.items():
            logging.info(f"  {link_type}: {len(urls)}")


async def main(args):
    """Main entry point for the crawler."""
    config = CrawlConfig(
        start_url=args.url,
        output_file=os.path.join("data", args.output),
        delay=args.delay,
        max_depth=args.depth,
        timeout=args.timeout,
        max_retries=args.retries,
        concurrency=args.concurrency,
        respect_robots_txt=not args.ignore_robots,
        user_agent=args.user_agent,
        log_level=args.log_level
    )

    crawler = WebCrawler(config)
    await crawler.crawl()
    crawler.export_links_to_csv()
    crawler.print_summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep Web Crawler')
    parser.add_argument('url', default="https://www.langchain.com/",help='Starting URL to crawl')
    parser.add_argument('-o', '--output', default='crawl_results.csv', help='Output CSV file')
    parser.add_argument('-d', '--delay', type=float, default=0.5, help='Delay between requests')
    parser.add_argument('--depth', type=int, default=5, help='Maximum crawl depth')
    parser.add_argument('--timeout', type=int, default=10, help='Request timeout in seconds')
    parser.add_argument('--retries', type=int, default=3, help='Maximum number of retries per URL')
    parser.add_argument('--concurrency', type=int, default=5, help='Number of concurrent requests')
    parser.add_argument('--ignore-robots', action='store_true', help='Ignore robots.txt restrictions')
    parser.add_argument('--user-agent', default='DeepCrawler/1.0', help='User-Agent string')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')

    args = parser.parse_args()

    asyncio.run(main(args))
