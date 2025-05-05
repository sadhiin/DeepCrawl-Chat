#!/usr/bin/env python3
import argparse
import sys
import os
import logging

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.deep_crawl_chat.core.crawler.integration import CrawlRAGPipeline
from src.utils.logging import setup_logging

def main():
    """Load existing crawl results into the RAG system."""
    parser = argparse.ArgumentParser(description='Load existing crawl results into RAG')
    parser.add_argument('csv_path', help='Path to the crawl results CSV file')
    parser.add_argument('--output-name', help='Name for the output vector store')

    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    csv_path = args.csv_path
    if not os.path.exists(csv_path):
        logger.error(f"CSV file does not exist: {csv_path}")
        sys.exit(1)

    try:
        pipeline = CrawlRAGPipeline()
        store_path = pipeline.index_crawl_results(csv_path)

        if store_path:
            logger.info(f"Successfully created vector store at: {store_path}")
            logger.info(f"\nCreated vector store: {store_path}")
            logger.info("You can now use this with the RAG system")
        else:
            logger.error("Failed to create vector store")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 