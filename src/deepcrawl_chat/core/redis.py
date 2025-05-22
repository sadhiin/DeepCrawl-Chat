import redis.asyncio as aioredis
from src.deepcrawl_chat.config.config import get_config
from src.deepcrawl_chat.core.utils import DeepCrawlError, ErrorCode
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class RedisPool:
    """
    Async Redis connection pool utility for DeepCrawl-Chat.
    """
    def __init__(self):
        self._pool: Optional[aioredis.Redis] = None
        self._config = get_config().redis

    async def connect(self):
        if self._pool is not None:
            logger.info("Redis pool already initialized.")
            return self._pool
        try:
            self._pool = aioredis.Redis(
                host=self._config.host,
                port=self._config.port,
                db=self._config.db,
                password=self._config.password,
                max_connections=self._config.pool_size,
                decode_responses=True,
            )
            # Test connection
            await self._pool.ping()
            logger.info(f"Connected to Redis at {self._config.host}:{self._config.port}, db {self._config.db}")
            return self._pool
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise DeepCrawlError(
                f"Failed to connect to Redis: {e}",
                ErrorCode.CONNECTION_ERROR
            )

    async def close(self):
        if self._pool is not None:
            await self._pool.close()
            logger.info("Redis pool closed.")
            self._pool = None

    @property
    def pool(self) -> aioredis.Redis:
        if self._pool is None:
            raise DeepCrawlError(
                "Redis pool is not initialized. Call connect() first.",
                ErrorCode.CONNECTION_ERROR
            )
        return self._pool