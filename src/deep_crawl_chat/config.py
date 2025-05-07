import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", env_file_encoding="utf-8"
    )

    NVIDIA_API_KEY: str = os.getenv("NVIDIA_API_KEY", "")
    LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY", "")
    LANGSMITH_TRACING: str = os.getenv("LANGSMITH_TRACING", "false")
    LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT", "DeepCrawl-Chat")

    EMBEDDING_MODEL: str = "nvidia/llama-3.2-nv-embedqa-1b-v2"
    LLM_MODEL: str = "deepseek-ai/deepseek-r1-distill-llama-8b"

    VECTOR_STORE_PATH: str = "data/vector_stores"

    CHUNK_SIZE: int = 5000
    CHUNK_OVERLAP: int = int(0.15 * CHUNK_SIZE)

    DOCUMENT_LOADER: str = "unstructured"


settings = Settings()