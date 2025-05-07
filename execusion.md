<think>
Let me review the folder structure and files you've shared to understand the organization of your DeepCrawl-Chat project. Then I'll provide instructions on how to run the project.

From the code snippets you've shared, I can see:

1. Project Structure:
   - DeepCrawl-Chat (root directory)
   - api/ (FastAPI application)
     - main.py (main FastAPI app)
     - routes/chat.py (API routes)
   - config/settings.py (configuration using pydantic)
   - deep_crawler.py (your web crawler script)
   - src/ (presumably contains source code modules)
   - notebooks/chat_app.ipynb (prototype notebook)
   - Dockerfile (for containerization)
   - .env.example (example environment variables)
   - .gitignore
   - LICENSE (MIT)
   - README.md

2. Configuration:
   - Uses environment variables for API keys (NVIDIA, HuggingFace, LangSmith)
   - Configures model choices, vector store paths, and chunking parameters

3. Functionality:
   - Web crawler (deep_crawler.py) to crawl websites and extract links
   - RAG (Retrieval Augmented Generation) system using LangChain
   - Vector storage with FAISS
   - API endpoints for chat and crawling functionality
   - Integration between crawled content and RAG system

4. New components we've added:
   - src/data/crawl_loader.py (to load documents from crawl results)
   - src/deep_crawler/integration.py (to integrate crawler with RAG)
   - src/cli/index_crawl.py (CLI tool for indexing)
   - scripts/load_existing_crawl.py (script to load existing crawl results)

Now I'll create instructions on how to set up and run the project.
</think>

# DeepCrawl-Chat: Setup and Running Instructions

Based on your project structure and the code you've shared, here's a comprehensive guide to set up and run your DeepCrawl-Chat project.

## 1. Project Setup

### Clone the Repository
```bash
git clone <your-repository-url>
cd DeepCrawl-Chat
```

### Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Set Up Environment Variables
```bash
cp .env.example .env
```

Then edit the `.env` file to add your API keys:
- `NVIDIA_API_KEY`: Your NVIDIA API key for embeddings
- `HF_TOKEN`: Your HuggingFace token
- `LANGSMITH_API_KEY`: Your LangSmith API key (optional for tracing)

## 2. Project Structure

The project is organized as follows:

```
DeepCrawl-Chat/
├── api/                      # FastAPI web application
│   ├── main.py               # Main application entry point
│   └── routes/               # API route handlers
│       └── chat.py           # Chat endpoint handlers
├── config/                   # Configuration
│   └── settings.py           # Settings using pydantic
├── data/                     # Data directory (will be created)
│   ├── crawls/               # Stored crawl results
│   └── vector_stores/        # FAISS vector stores
├── notebooks/                # Jupyter notebooks
│   └── chat_app.ipynb        # Prototype notebook
├── scripts/                  # Utility scripts
│   └── load_existing_crawl.py # Load existing crawl data
├── src/                      # Source code
│   ├── chains/               # LangChain chains
│   ├── cli/                  # Command-line tools
│   │   └── index_crawl.py    # Tool to index crawl results
│   ├── data/                 # Data handling
│   │   └── crawl_loader.py   # Load documents from crawls
│   ├── deep_crawler/         # Crawler implementation
│   │   └── integration.py    # Integration with RAG
│   ├── embeddings/           # Embedding models
│   └── vectorstores/         # Vector store implementations
├── deep_crawler.py           # Standalone crawler script
├── Dockerfile                # Docker configuration
├── .env.example              # Example environment variables
└── README.md                 # Project documentation
```

## 3. Running the Project

### Option 1: Run the Crawler First, Then Index the Results

1. **Run the crawler to gather website data**:
   ```bash
   python deep_crawler.py https://www.example.com --output data/crawls/example_crawl.csv --depth 3
   ```

2. **Index the crawl results**:
   ```bash
   python -m src.cli.index_crawl data/crawls/example_crawl.csv --url-type page
   ```
   This will create a vector store that can be used by the RAG system.

3. **Start the API server**:
   ```bash
   uvicorn api.main:app --reload
   ```

4. **Access the API**:
   - Open your browser and go to `http://localhost:8000/docs` to see the Swagger UI.
   - Use the `/chat` endpoint with your crawl results.

### Option 2: Use the Integrated API (Crawl and Chat)

1. **Start the API server**:
   ```bash
   uvicorn api.main:app --reload
   ```

2. **Use the API to crawl and index**:
   - Make a POST request to `/crawl` with a list of URLs to crawl.
   - The API will return a `crawl_id` that you can use for chatting.

3. **Chat using the crawled data**:
   - Make a POST request to `/chat` with your query and the `crawl_id`.

### Option 3: Use Existing Crawl Results (CSV)

If you already have crawl results in a CSV file:

```bash
python scripts/load_existing_crawl.py path/to/your/langchain_crawl_results.csv
```

This will create a vector store from your existing crawl data.

## 4. Docker Deployment

To run the project using Docker:

```bash
# Build the Docker image
docker build -t deepcrawl-chat .

# Run the container
docker run -p 8000:8000 --env-file .env deepcrawl-chat
```

## 5. Development Workflow

1. **Create structure for new modules**:
   ```bash
   mkdir -p src/some_new_module
   touch src/some_new_module/__init__.py
   ```

2. **Update environment variables** when adding new dependencies:
   ```
   NEW_API_KEY=your_key_here
   ```

3. **Run tests** (if implemented):
   ```bash
   pytest
   ```

## 6. Folder Descriptions

- **api/**: Contains the FastAPI application that serves as the web interface.
- **config/**: Configuration management using pydantic.
- **data/**: Stores crawled data and vector stores.
- **src/**: Contains the core implementation of different components.
  - **chains/**: LangChain chains for the RAG system.
  - **data/**: Handles loading and processing documents.
  - **deep_crawler/**: Contains the crawler and integration code.
  - **embeddings/**: Manages embedding models.
  - **vectorstores/**: Handles vector storage with FAISS.
- **scripts/**: Utility scripts for different tasks.

## 7. Common Operations

### Index Existing Crawl Results
```bash
python -m src.cli.index_crawl path/to/your/crawl_results.csv
```

### Load a Specific URL for Testing
```bash
python scripts/load_existing_crawl.py path/to/your/langchain_crawl_results.csv
```

### Run a Quick Crawl
```bash
python deep_crawler.py https://example.com --output data/crawls/quick_test.csv --depth 1
```

## 8. Project Improvements

As you continue developing:

1. Add more comprehensive tests
2. Implement caching mechanisms
3. Set up monitoring and logging
4. Create a simple frontend for demonstration
5. Add rate limiting and authentication

Would you like more specific details on any particular aspect of running or developing the project?
