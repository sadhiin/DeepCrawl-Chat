import os
from dataclasses import dataclass
from typing import Any, Dict
from pydantic import TypeAdapter

from pydantic_settings import BaseSettings

from hydra import initialize, compose
from hydra.core.config_store import ConfigStore

from src.deepcrawl_chat.config.schemas import DatabaseConfigUnion, VectorStoreConfigUnion

# @dataclass
# class AppConfigHydra:
#     database: Dict[str, Any]
#     vector_store: Dict[str, Any]

# cs = ConfigStore.instance()
# cs.store(name="app_config", node=AppConfigHydra)

# def config_setting():
#     with initialize(config_path="../../../configs", version_base='1.2'):
#         cfg = compose(config_name="config")

#         db_config = TypeAdapter(DatabaseConfigUnion).validate_python(cfg.database)  # Use TypeAdapter
#         vs_config = TypeAdapter(VectorStoreConfigUnion).validate_python(cfg.vectorstore)  # Use TypeAdapter

#         class AppConfig(BaseSettings):
#             database: DatabaseConfigUnion
#             vector_store: VectorStoreConfigUnion

#         app_config = AppConfig(database=db_config, vector_store=vs_config)

#         print("Database connection string:", app_config.database.get_connection_string())
#         print("Vector store info:", app_config.vector_store.get_store())

#         return app_config

# setting = config_setting()
class AppConfig(BaseSettings):
    """
    Root config for DeepCrawl Chat.
    Hydra will load all of `configs/…/*.yaml` into a DictConfig,
    then we hand that entire DictConfig to Pydantic for validation.
    """
    database: DatabaseConfigUnion
    vectorstore: VectorStoreConfigUnion

    class Config:
        # allow unused keys in the YAML (e.g. hydra.run.dir, app.name, etc.)
        extra = "ignore"
        # case-insensitive environment-var parsing, if you like:
        env_file = ".env"
        env_file_encoding = "utf-8"

def load_config() -> AppConfig:
    # point at your top-level `configs/` directory:
    config_path = "../../../configs"
    with initialize(config_path=config_path, version_base="1.2"):
        hydra_cfg = compose(config_name="config")
        # directly hand the Hydra DictConfig to Pydantic
        return TypeAdapter(AppConfig).validate_python(hydra_cfg)

def main():
    cfg = load_config()
    print("→ DB connection:", cfg.database.get_connection_string())
    print("→ Vector store:", cfg.vectorstore.get_store())

setting = main()