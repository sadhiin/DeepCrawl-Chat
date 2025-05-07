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
