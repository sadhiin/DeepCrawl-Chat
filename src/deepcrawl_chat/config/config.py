from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from src.deepcrawl_chat.config.schemas import PostgresConfig, FaissConfig

class AppConfig(BaseSettings):
    # This uses the environment variables (e.g. POSTGRES_URL) but falls back to default schema config 
    database: PostgresConfig = Field(default_factory=lambda: PostgresConfig(type="postgres"))
    vector_store: FaissConfig = Field(default_factory=lambda: FaissConfig(type="faiss"))
    DOCUMENT_LOADER: str = "WebBaseLoader"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

def config_setting() -> AppConfig:
    app_config = AppConfig()
    print("Database connection string:", app_config.database.get_connection_string())
    print("Vector store info:", app_config.vector_store.get_store())
    return app_config

settings = config_setting()
