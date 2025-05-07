import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Union, Literal

from dotenv import load_dotenv
load_dotenv()

# Base class for database configs
class DatabaseConfig(BaseSettings):
    model_config: SettingsConfigDict = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        env_file_encoding="utf-8"
    )
    type: str
    def get_connection_string(self) -> str:
        raise NotImplementedError()

# SQLite configuration
class SQLiteConfig(DatabaseConfig):
    type: Literal["sqlite"]
    filepath: str = "sqlite.db"

    def get_connection_string(self) -> str:
        return f"sqlite:///{self.filepath}"

# Postgres configuration
class PostgresConfig(DatabaseConfig):
    type: Literal["postgres"]
    host: str = "localhost"
    port: int = 5432
    username: str = "user"
    password: str = "password"
    db_name: str = "mydb"

    def get_connection_string(self) -> str:
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.db_name}"

# Factory function
def get_database_config(config: dict) -> DatabaseConfig:
    db_type = config.get("type")
    if db_type == "sqlite":
        return SQLiteConfig(**config)
    elif db_type == "postgres":
        return PostgresConfig(**config)
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

# Similar for vector store
class VectorStoreConfig(BaseSettings):
    model_config: SettingsConfigDict = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        env_file_encoding="utf-8"
    )
    type: str  # discriminator

    def get_store(self):
        raise NotImplementedError()

# Faiss vector store
class FaissConfig(VectorStoreConfig):
    type: Literal["faiss"]
    index_path: str = "faiss.index"

    def get_store(self):
        # Initialize Faiss index here
        return f"Faiss index at {self.index_path}"

# Chroma vector store
class ChromaConfig(VectorStoreConfig):
    type: Literal["chroma"]
    collection_name: str = "my_collection"

    def get_store(self):
        # Initialize Chroma store here
        return f"Chroma collection {self.collection_name}"

# Factory for vector store
def get_vector_store_config(config: dict) -> VectorStoreConfig:
    store_type = config.get("type")
    if store_type == "faiss":
        return FaissConfig(**config)
    elif store_type == "chroma":
        return ChromaConfig(**config)
    else:
        raise ValueError(f"Unsupported vector store type: {store_type}")
