#!/usr/bin/env python3
import asyncio
import os
import argparse
from src.deepcrawl_chat.crawler.core import WebCrawler, CrawlConfig

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
    parser.add_argument('url', default="https://sadhiin.github.io",help='Starting URL to crawl')
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
