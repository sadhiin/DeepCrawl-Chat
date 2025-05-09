import os
from dataclasses import dataclass
from typing import Any, Dict
from pydantic import TypeAdapter

from pydantic_settings import BaseSettings

from hydra import initialize, compose
from hydra.core.config_store import ConfigStore

from src.deepcrawl_chat.config.schemas import DatabaseConfigUnion, VectorStoreConfigUnion

@dataclass
class AppConfigHydra:
    database: Dict[str, Any]
    vector_store: Dict[str, Any]

cs = ConfigStore.instance()
cs.store(name="app_config", node=AppConfigHydra)

def config_setting():
    with initialize(config_path="../../../configs", version_base='1.2'):
        cfg = compose(config_name="config")

        db_config = TypeAdapter(DatabaseConfigUnion).validate_python(cfg.database)
        vs_config = TypeAdapter(VectorStoreConfigUnion).validate_python(cfg.vectorstore)

        class AppConfig(BaseSettings):
            database: DatabaseConfigUnion
            vector_store: VectorStoreConfigUnion

        app_config = AppConfig(database=db_config, vector_store=vs_config)

        print("Database connection string:", app_config.database.get_connection_string())
        print("Vector store info:", app_config.vector_store.get_store())

        return app_config

setting = config_setting()

