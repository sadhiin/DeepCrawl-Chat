import asyncio
import aiofiles
import aiohttp
from typing import Dict, Set, List, Optional
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import re
import csv
from collections import defaultdict
from aiohttp import ClientTimeout

from deepcrawl_chat.schemas.crawling_schema import CrawlConfig, LinkCategory
from deepcrawl_chat.crawler.utils import UrlUtils, RobotsTxtManager
from deepcrawl_chat.utils import create_logger

logger = create_logger()


class WebCrawler:
    """Asynchronous web crawler with improved features."""

    def __init__(self, config: CrawlConfig):
        self.config = config
        self.base_domain = UrlUtils.get_base_domain(config.start_url)
        self.visited_urls: Set[str] = set()
        self.urls_to_visit: List[tuple] = [(UrlUtils.normalize_url(config.start_url), 0)]  # (url, depth)
        self.discovered_links: Dict[str, Set[str]] = defaultdict(set)
        self.robots_manager = RobotsTxtManager(config.user_agent)

    async def crawl(self) -> Dict[str, Set[str]]:
        """Main crawling method."""
        logger.info(f"Starting crawl at {self.config.start_url} with max depth {self.config.max_depth}")

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
                        logger.info(f"Skipping {url} (disallowed by robots.txt)")
                        continue

                # Add task to process the URL
                task = asyncio.create_task(self.process_url(url, depth, session, semaphore))
                tasks.append(task)

                # Add a small delay between starting task    s
                if self.config.delay > 0:
                    await asyncio.sleep(self.config.delay)

            # Wait for all tasks to complete
            if tasks:
                await asyncio.gather(*tasks)

        return dict(self.discovered_links)

    async def process_url(self, url: str, depth: int, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore):
        """Process a single URL."""
        if url in self.visited_urls:
            # TODO: Handle already visited URLs
            # we can add the document loader here.
            return

        self.visited_urls.add(url)

        for attempt in range(self.config.max_retries):
            try:
                async with semaphore:
                    logger.debug(f"Crawling: {url} (depth: {depth}, attempt: {attempt+1})")

                    async with session.get(url) as response:
                        if response.status != 200:
                            logger.warning(f"Failed to fetch {url} - Status {response.status}")
                            continue

                        # Add URL to appropriate category
                        category = UrlUtils.categorize_url(url)
                        self.discovered_links[category].add(url)

                        # Only parse HTML content for extraction
                        content_type = response.headers.get('Content-Type', '').lower()
                        if 'text/html' not in content_type:
                            logger.debug(f"Skipping non-HTML content: {url}")
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



