# DeepCrawl-Chat

<!-- <p align="center">
  <img src="docs/images/deepcrawl-logo.png" alt="DeepCrawl-Chat Logo" width="200"/>
</p> -->

DeepCrawl-Chat is an intelligent web crawling and retrieval augmented generation (RAG) system that allows you to crawl websites, extract information, and ask questions about the crawled content using state-of-the-art AI language models.

## 🌟 Features

- **🕸️ Advanced Web Crawling**: Efficiently crawl websites with configurable depth, concurrency, and filters
- **📊 Data Extraction**: Extract text, links, and media from various content types
- **🤖 RAG System**: Ask questions about crawled content using retrieval augmented generation
- **⚡ High Performance**: Asynchronous crawling and parallel document processing
- **🔄 Flexible Integration**: Use as a standalone tool or integrate with other systems
- **🔍 Categorized Results**: Automatically categorize discovered links (pages, documents, media)
- **🛡️ Respectful Crawling**: Built-in rate limiting and robots.txt compliance

## 📋 Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Usage Examples](#-usage-examples)
- [Configuration](#-configuration)
- [API Reference](#-api-reference)
- [Development](#-development)
- [Contributing](#-contributing)
- [License](#-license)

## 🔧 Installation

### Prerequisites

- Python 3.12+ (managed by `uv`)
- `uv` (Fast Python package and project manager)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/sadhiin/DeepCrawl-Chat.git
cd DeepCrawl-Chat
```

2. Sync the environment and install dependencies using `uv`:
```bash
uv sync
```

3. Set up environment variables:
```bash
cp .env.example .env
```

Edit the `.env` file to add your API keys:
```
NVIDIA_API_KEY=your_nvidia_api_key
HF_TOKEN=your_huggingface_token
LANGSMITH_API_KEY=your_langsmith_api_key  # Optional
```

## 🚀 Quick Start

### Crawl a Website

```bash
uv run python deep_crawler.py https://www.example.com --output data/crawls/example_crawl.csv --depth 3
```

### Index the Crawled Content

```bash
uv run python -m src.deepcrawl_chat.cli.index_crawl data/crawls/example_crawl.csv
```

### Start the Chat API

```bash
uv run uvicorn src.deepcrawl_chat.api.main:app --reload
```

Open your browser and navigate to `http://localhost:8000/docs` to access the API documentation and chat with your crawled data.

## 📁 Project Structure

```
DeepCrawl-Chat/
├── api/                      # FastAPI web application
├── config/                   # Configuration management
├── data/                     # Data directory
│   ├── crawls/               # Stored crawl results
│   └── vector_stores/        # FAISS vector stores
├── notebooks/                # Jupyter notebooks
├── scripts/                  # Utility scripts
├── src/                      # Source code
│   ├── chains/               # LangChain chains
│   ├── cli/                  # Command-line tools
│   ├── data/                 # Data handling
│   ├── deep_crawler/         # Crawler implementation
│   ├── embeddings/           # Embedding models
│   └── vectorstores/         # Vector store implementations
├── tests/                    # Test suite
├── deep_crawler.py           # Standalone crawler script
├── Dockerfile                # Docker configuration
├── .env.example              # Example environment variables
└── README.md                 # Project documentation
```

## 📝 Usage Examples

### Crawl and Chat in One Step (API)

```python
import requests

# Chat with the crawled content directly
chat_response = requests.post(
    "http://localhost:8000/chat",
    json={
        "query": "What products does the company offer?",
        "urls": ["https://www.example.com"]
    }
)

print(chat_response.json()["answer"])
```

### Using Existing Crawl Results

```bash
uv run scripts/load_existing_crawl.py path/to/your/crawl_results.csv
```

### Crawl Options

```bash
uv run deep_crawler.py --help
```

This will display all available options for the crawler:

```
usage: deep_crawler.py [-h] [-o OUTPUT] [-d DELAY] [--depth DEPTH] [--timeout TIMEOUT]
                      [--retries RETRIES] [--concurrency CONCURRENCY] [--ignore-robots]
                      [--user-agent USER_AGENT] [--log-level {DEBUG,INFO,WARNING,ERROR}]
                      url

Deep Web Crawler

positional arguments:
  url                   Starting URL to crawl

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Output CSV file
  -d DELAY, --delay DELAY
                        Delay between requests
  --depth DEPTH         Maximum crawl depth
  --timeout TIMEOUT     Request timeout in seconds
  --retries RETRIES     Maximum number of retries per URL
  --concurrency CONCURRENCY
                        Number of concurrent requests
  --ignore-robots       Ignore robots.txt restrictions
  --user-agent USER_AGENT
                        User-Agent string
  --log-level {DEBUG,INFO,WARNING,ERROR}
                        Logging level
```

## ⚙️ Configuration

DeepCrawl-Chat can be configured using environment variables or a .env file:

| Environment Variable | Description | Default |
|----------------------|-------------|---------|
| NVIDIA_API_KEY | NVIDIA API key for embeddings | (Required) |
| HF_TOKEN | HuggingFace token | (Required) |
| LANGSMITH_API_KEY | LangSmith API key for tracing | (Optional) |
| LANGSMITH_TRACING | Enable LangSmith tracing | "false" |
| EMBEDDING_MODEL | Embedding model name | "nvidia/llama-3.2-nv-embedqa-1b-v2" |
| LLM_MODEL | LLM model name | "deepseek-ai/deepseek-r1-distill-llama-8b" |
| CHUNK_SIZE | Text chunk size | 5000 |
| CHUNK_OVERLAP | Chunk overlap size | 100 |

## 🔌 API Reference

The DeepCrawl-Chat API provides the following endpoints:

### `/chat` (POST)

Dynamically crawl websites and chat with their content in a single operation using Retrieval Augmented Generation.

**Request Body:**
```json
{
  "query": "What does this website offer?",
  "urls": ["https://example.com"]
}
```

**Response:**
```json
{
  "answer": "Based on the crawled content, the website offers...",
  "sources": ["https://example.com/page1", "https://example.com/page2"]
}
```

## 🛠️ Development

### Setting Up Development Environment

```bash
# Sync all development dependencies
uv sync --dev

# Setup pre-commit hooks
uv run pre-commit install
```

### Running Tests

```bash
uv run pytest
```

### Docker Development

```bash
# Build the Docker image
docker build -t deepcrawl-chat .

# Run the container
docker run -p 8000:8000 --env-file .env deepcrawl-chat
```

## 🤝 Contributing

We welcome contributions to DeepCrawl-Chat! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

Please ensure your code follows our coding standards and includes appropriate tests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Built with ❤️ by [@sadhiin](https://github.com/sadhiin)
