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
    api_endpoint: str = Field(default_factory=lambda: os.getenv("GEMINI_API_ENDPOINT", "https://generativelanguage.googleapis.com/v1beta/openai/"))
    api_key: str = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY", "your_api_key"))
    model_name: str = Field(default_factory=lambda: os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash"))
    temperature: float = Field(default_factory=lambda: float(os.getenv("GEMINI_TEMPERATURE", 0.7)))
    max_tokens: int = Field(default_factory=lambda: int(os.getenv("GEMINI_MAX_TOKENS", 1024)))
    top_p: float = Field(default_factory=lambda: float(os.getenv("GEMINI_TOP_P", 0.7)))
    
class NvidiaLLMConfig(LLMConfig):
    type : Literal["nvidia"]
    api_endpoint: str = Field(default_factory=lambda: os.getenv("NVIDIA_API_ENDPOINT", "https://api.nvidia.com/llm"))
    api_key: str = Field(default_factory=lambda: os.getenv("NVIDIA_API_KEY", "your_api_key"))
    model_name: str = Field(default_factory=lambda: os.getenv("NVIDIA_MODEL_NAME", "nvidia-llm"))
    temperature: float = Field(default_factory=lambda: float(os.getenv("NVIDIA_TEMPERATURE", 0.7)))
    max_tokens: int = Field(default_factory=lambda: int(os.getenv("NVIDIA_MAX_TOKENS", 1024)))
    top_p: float = Field(default_factory=lambda: float(os.getenv("NVIDIA_TOP_P", 0.7)))
    
class GroqLLMCofig():
    type: Literal["groq"]
    api_endpoint: str = Field(default_factory=lambda: os.getenv("GROQ_API_ENDPOINT", "https://api.groq.com/llm"))
    api_key: str = Field(default_factory=lambda: os.getenv("GROQ_API_KEY", "your_api_key"))
    model_name: str = Field(default_factory=lambda: os.getenv("GROQ_MODEL_NAME", "groq-llm"))
    temperature: float = Field(default_factory=lambda: float(os.getenv("GROQ_TEMPERATURE", 0.7)))
    max_tokens: int = Field(default_factory=lambda: int(os.getenv("GROQ_MAX_TOKENS", 1024)))
    top_p: float = Field(default_factory=lambda: float(os.getenv("GROQ_TOP_P", 0.7)))
    
LLMConfigUnion = Annotated[
    Union[GeminiLLMConfig, NvidiaLLMConfig, GroqLLMCofig],
    Field(discriminator="type")
]
