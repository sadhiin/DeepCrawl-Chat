from numpy import append
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Union, Literal, Annotated
from dotenv import load_dotenv
import os
load_dotenv()
## --------------------------------------- Database configs --------------------------------------- ##
# Base class for database configs
class DatabaseConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        env_file_encoding="utf-8"
    )
    type: str

    def get_connection_string(self) -> str:
        raise NotImplementedError()

class SQLiteConfig(DatabaseConfig):
    type: Literal["sqlite"]
    filepath: str = "place_holder_value.db"

    def get_connection_string(self) -> str:
        return f"sqlite:///{self.filepath}"

class PostgresConfig(DatabaseConfig):
    type: Literal["postgres"]
    host: str = "localhost"
    port: int = 5432
    username: str = "user"
    password: str = "password"
    db_name: str = "mydatabase"

    def get_connection_string(self) -> str:
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.db_name}"

# Discriminated union for database config
DatabaseConfigUnion = Annotated[
    Union[SQLiteConfig, PostgresConfig],
    Field(discriminator="type")
]

## --------------------------------------- Vector store configs --------------------------------------- ##
# Base class for vector store configs
class VectorStoreConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        env_file_encoding="utf-8"
    )
    type: str

    def get_store(self):
        raise NotImplementedError()

class FaissConfig(VectorStoreConfig):
    type: Literal["faiss"]
    index_path: str = "faiss.index"

    def get_store(self):
        return f"Faiss index at {self.index_path}"

class ChromaConfig(VectorStoreConfig):
    type: Literal["chroma"]
    collection_name: str = "my_collection"

    def get_store(self):
        return f"Chroma collection {self.collection_name}"

VectorStoreConfigUnion = Annotated[
    Union[FaissConfig, ChromaConfig],
    Field(discriminator="type")
]



## ------------------------------------ Base class for LLM Configs ------------------------------------ ##

class LLMConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        env_file_encoding="utf-8"
    )

    type: str
    api_endpoint: str
    api_key: str
    model_name: str
    temperature: float
    max_tokens: int
    top_p: float = 0.7

class GeminiLLMConfig(LLMConfig):
    type: Literal["google"]
    api_endpoint: str = Field(default_factory=lambda: os.getenv("GEMINI_API_ENDPOINT", "https://googleapis.com/endpoint/uri/"))
    api_key: str = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY", "your_api_key"))
    api_provider: str = Field(default_factory=lambda: os.getenv("GEMINI_API_PROVIDER", "google"))
    model_name: str = Field(default_factory=lambda: os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash"))
    temperature: float = Field(default_factory=lambda: float(os.getenv("GEMINI_TEMPERATURE", 0.7)))
    max_tokens: int = Field(default_factory=lambda: int(os.getenv("GEMINI_MAX_TOKENS", 1024)))
    top_p: float = Field(default_factory=lambda: float(os.getenv("GEMINI_TOP_P", 0.7)))

class NvidiaLLMConfig(LLMConfig):
    type : Literal["nvidia"]
    api_endpoint: str = Field(default_factory=lambda: os.getenv("NVIDIA_API_ENDPOINT", "https://api.nvidia.com/endpoint/uri/"))
    api_key: str = Field(default_factory=lambda: os.getenv("NVIDIA_API_KEY", "your_api_key"))
    api_provider: str = Field(default_factory=lambda: os.getenv("NVIDIA_API_PROVIDER", "nvidia"))
    model_name: str = Field(default_factory=lambda: os.getenv("NVIDIA_MODEL_NAME", "nvidia-llm"))
    temperature: float = Field(default_factory=lambda: float(os.getenv("NVIDIA_TEMPERATURE", 0.7)))
    max_tokens: int = Field(default_factory=lambda: int(os.getenv("NVIDIA_MAX_TOKENS", 1024)))
    top_p: float = Field(default_factory=lambda: float(os.getenv("NVIDIA_TOP_P", 0.7)))

class GroqLLMConfig(LLMConfig):
    type: Literal["groq"]
    api_endpoint: str = Field(default_factory=lambda: os.getenv("GROQ_API_ENDPOINT", "https://api.groq.com/endpoint/uri/"))
    api_key: str = Field(default_factory=lambda: os.getenv("GROQ_API_KEY", "your_api_key"))
    api_provider: str = Field(default_factory=lambda: os.getenv("GROQ_API_PROVIDER", "groq"))
    model_name: str = Field(default_factory=lambda: os.getenv("GROQ_MODEL_NAME", "groq-llm"))
    temperature: float = Field(default_factory=lambda: float(os.getenv("GROQ_TEMPERATURE", 0.7)))
    max_tokens: int = Field(default_factory=lambda: int(os.getenv("GROQ_MAX_TOKENS", 1024)))
    top_p: float = Field(default_factory=lambda: float(os.getenv("GROQ_TOP_P", 0.7)))

    class Config:
        arbitrary_types_allowed = True


LLMConfigUnion = Annotated[
    Union[GeminiLLMConfig, NvidiaLLMConfig, GroqLLMConfig],
    Field(discriminator="type")
]

## ------------------------------------ Base config class for embedding ------------------------------------ ##


class EmbeddingConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        env_file_encoding="utf-8"
    )
    type: str


class NvidiaEmbeddingConfig(EmbeddingConfig):
    type: Literal["nvidia"]
    api_provider: str = 'nividia'
    api_key: str = Field(default_factory=lambda: os.getenv("NVIDIA_API_KEY", "your_api_key"))
    api_endpoint: str = Field(default_factory=lambda: os.getenv("NVIDIA_API_ENDPOINT", "https://api.nvidia.com/endpoint/uri/"))
    embedding_model: str = Field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "default_embedding_model"))
    chunk_size: int = 5000
    chunk_overlap: int = int(0.15*5000)
    similarity_threshold: float = Field(default_factory=lambda: float(os.getenv("SIMILARITY_THRESHOLD", 0.7)))

class RedisConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        env_file_encoding="utf-8"
    )
    host: str = Field(default_factory=lambda: os.getenv("REDIS_HOST", "localhost"))
    port: int = Field(default_factory=lambda: int(os.getenv("REDIS_PORT", 6379)))
    db: int = Field(default_factory=lambda: int(os.getenv("REDIS_DB", 0)))
    password: str | None = Field(default_factory=lambda: os.getenv("REDIS_PASSWORD", None))
    pool_size: int = Field(default_factory=lambda: int(os.getenv("REDIS_POOL_SIZE", 10)))
