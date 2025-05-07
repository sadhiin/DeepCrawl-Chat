from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Dict
import json

class DatabaseSettings(BaseSettings):
    model_config: SettingsConfigDict = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        env_file_encoding="utf-8"
    )

    DATABASE_URL: str = "sqlite:///:memory:"
    DATABASE_USERNAME: str = "user"
    DATABASE_PASSWORD: str = "password"
    DATABASE_NAME: str = "my_database"
    DATABASE_PORT: int = 5432


class VectorStoreSettings(BaseSettings):
    model_config: SettingsConfigDict = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        env_file_encoding="utf-8"
    )

    VECTOR_STORE_PATH: str = "data/vector_stores"
    EMBEDDING_MODEL: str = "nvidia/llama-3.2-nv-embedqa-1b-v2"
    CHUNK_SIZE: int = 5000
    CHUNK_OVERLAP: int = None  # Will be set via validator
    DOCUMENT_LOADER: str = "unstructured"

    @property
    def default_chunk_overlap(self) -> int:
        return int(0.15 * self.CHUNK_SIZE)

    @validator("CHUNK_OVERLAP", pre=True, always=True)
    def set_chunk_overlap(cls, v, values):
        if v is not None:
            return v
        chunk_size = values.get("CHUNK_SIZE", 5000)
        return int(0.15 * chunk_size)


class LLMSettings(BaseSettings):
    model_config: SettingsConfigDict = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        env_file_encoding="utf-8"
    )

    LLM_MODEL: str = "deepseek-ai/deepseek-r1-distill-llama-8b"
    NVIDIA_API_KEY: str = ""
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_TRACING: bool = False
    LANGSMITH_PROJECT: str = "DeepCrawl-Chat"


class RetrieverSettings(BaseSettings):
    model_config: SettingsConfigDict = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        env_file_encoding="utf-8"
    )

    RETRIEVER_TYPE: str = "default"
    RETRIEVER_PARAMS: Dict = {}

    @validator("RETRIEVER_PARAMS", pre=True, always=True)
    def parse_retriever_params(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return {}
        return v


class AppConfig(BaseSettings):
    model_config: SettingsConfigDict = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        env_file_encoding="utf-8"
    )

    database: DatabaseSettings = DatabaseSettings()
    vector_store: VectorStoreSettings = VectorStoreSettings()
    llm: LLMSettings = LLMSettings()
    retriever: RetrieverSettings = RetrieverSettings()


settings = AppConfig()

