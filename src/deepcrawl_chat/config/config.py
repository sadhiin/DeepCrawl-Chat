import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Union, Literal
from pydantic import field_validator

from .schemas import (
    DatabaseConfig, 
    VectorStoreConfig, 
    
    get_database_config, 
    get_vector_store_config
)

class AppConfig(BaseSettings):
    model_config: SettingsConfigDict = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        env_file_encoding="utf-8"
    )

    # Instantiate specific configs
    database: DatabaseConfig = None
    vector_store: VectorStoreConfig = None


    # Load the dynamic configurations from environment variables
    
    @field_validator("database", mode='before')
    def load_database(cls, v, values):
        # Load raw dict from environment or file
        config_data = {
            "type": os.getenv("DB_TYPE", "sqlite"),
            "filepath": os.getenv("SQLITE_FILEPATH", "sqlite.db"),
        }
        return get_database_config(config_data)

    @field_validator("vector_store", mode='before')
    def load_vector_store(cls, v, values):
        config_data = {
            "type": os.getenv("VSTORE_TYPE", "faiss"),
            "index_path": os.getenv("FAISS_INDEX_PATH", "faiss.index"),
            "collection_name": os.getenv("CHROMA_COLLECTION", "my_collection"),
        }
        return get_vector_store_config(config_data)

# Instantiate
settings = AppConfig()
