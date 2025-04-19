import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    NVIDIA_API_KEY: str = os.getenv("NVIDIA_API_KEY", "")
    LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY", "")
    LANGSMITH_TRACING: str = os.getenv("LANGSMITH_TRACING", "false")
    LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT", "DeepCrawl-Chat")

    EMBEDDING_MODEL: str = "nvidia/llama-3.2-nv-embedqa-1b-v2"
    LLM_MODEL: str = "deepseek-ai/deepseek-r1-distill-llama-8b"
    # LLM_MODEL: str = "meta/llama3-8b-instruct"

    VECTOR_STORE_PATH: str = "data/vector_stores"

    CHUNK_SIZE: int = 5000
    CHUNK_OVERLAP: int = 100

    DOCUMENT_LOADER: str = "unstructured"
    class Config:
        env_file = ".env"

settings = Settings()