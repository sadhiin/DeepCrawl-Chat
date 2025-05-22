import os
from typing import Any, Dict
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from hydra import initialize, compose
from omegaconf import OmegaConf
from src.deepcrawl_chat.config.schemas import (
    DatabaseConfigUnion,
    VectorStoreConfigUnion,
    LLMConfigUnion,
    NvidiaEmbeddingConfig,
    RedisConfig
)

import threading

class AppConfig(BaseModel):
    app: Dict[str, Any] = Field(default_factory=dict)
    database: DatabaseConfigUnion
    vectorstore: VectorStoreConfigUnion
    llm: LLMConfigUnion
    embedding: NvidiaEmbeddingConfig
    redis: RedisConfig

# Singleton cache and lock
_config_instance = None
_config_lock = threading.Lock()

def get_config():
    global _config_instance
    if _config_instance is not None:
        return _config_instance

    with _config_lock:
        if _config_instance is None:
            # Initialize Hydra and compose config
            with initialize(config_path="../../../configs", version_base=None):
                cfg = compose(config_name="config")
                cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
                # Validate and create AppConfig instance
                _config_instance = AppConfig(**cfg_dict)
                print("Configuration loaded successfully.")
    return _config_instance

# Optional: if you want to test loading directly
if __name__ == "__main__":
    config = get_config()
    print(config)
