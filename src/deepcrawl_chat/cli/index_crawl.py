# #!/usr/bin/env python3
# import argparse
# import sys
# import os

# # Add the project root to sys.path if needed
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# from deepcrawl_chat.crawler.WebCrawler import WebCrawler
# from deepcrawl_chat.utils import create_logger

# def main():
#     """Main entry point for indexing crawl results."""
#     parser = argparse.ArgumentParser(description='Index crawl results for RAG')
#     parser.add_argument('csv_path', help='Path to the crawl results CSV file')
#     parser.add_argument('--url-type', default='page', help='Type of URLs to index (default: page)')
#     parser.add_argument('--chunk-size', type=int, default=5000, help='Size of text chunks')
#     parser.add_argument('--chunk-overlap', type=int, default=100, help='Overlap between chunks')
#     parser.add_argument('--vector-store-dir', default='data/vector_stores',
#                         help='Directory for vector stores')
#     parser.add_argument('--log_level', default='INFO',
#                         choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
#                         help='Logging level')

#     args = parser.parse_args()

#     # Setup logging
#     logger = create_logger(args.log_level)

#     try:
#         # Create and run the pipeline
#         pipeline = CrawlRAGPipeline(
#             vector_store_dir=args.vector_store_dir,
#             chunk_size=args.chunk_size,
#             chunk_overlap=args.chunk_overlap
#         )

#         store_path = pipeline.index_crawl_results(args.csv_path, url_type=args.url_type)

#         if store_path:
#             logger.info(f"Successfully created vector store at: {store_path}")
#             print(f"\nVector store created: {store_path}")
#             print(f"You can now use this with the RAG system by specifying this path.")
#         else:
#             logger.error("Failed to create vector store")
#             sys.exit(1)

#     except Exception as e:
#         logger.error(f"Error indexing crawl results: {e}")
#         sys.exit(1)

# if __name__ == "__main__":
#     main()