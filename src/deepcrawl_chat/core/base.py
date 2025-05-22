from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import logging
from .interfaces import Worker, Service, Storage

logger = logging.getLogger(__name__)

class BaseWorker(ABC):
    """Base implementation of the Worker interface."""

    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize the worker.

        Args:
            name: The name of the worker
            config: Configuration for the worker
        """
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{name}")

    @abstractmethod
    async def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task and return the result.

        Args:
            task: The task to process

        Returns:
            The result of processing the task
        """
        pass

    async def health_check(self) -> bool:
        """Check if the worker is healthy.

        Returns:
            True if the worker is healthy, False otherwise
        """
        try:
            # Basic health check - can be overridden by subclasses
            return True
        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            return False

class BaseService(ABC):
    """Base implementation of the Service interface."""

    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize the service.

        Args:
            name: The name of the service
            config: Configuration for the service
        """
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self._is_running = False

    @abstractmethod
    async def start(self) -> None:
        """Start the service."""
        self._is_running = True
        self.logger.info(f"Service {self.name} started")

    @abstractmethod
    async def stop(self) -> None:
        """Stop the service."""
        self._is_running = False
        self.logger.info(f"Service {self.name} stopped")

    async def health_check(self) -> bool:
        """Check if the service is healthy.

        Returns:
            True if the service is healthy, False otherwise
        """
        return self._is_running

class BaseStorage(ABC):
    """Base implementation of the Storage interface."""

    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize the storage.

        Args:
            name: The name of the storage
            config: Configuration for the storage
        """
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{name}")

    @abstractmethod
    async def store(self, key: str, value: Any) -> None:
        """Store a value with the given key.

        Args:
            key: The key to store the value under
            value: The value to store
        """
        pass

    @abstractmethod
    async def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a value by key.

        Args:
            key: The key to retrieve

        Returns:
            The stored value, or None if not found
        """
        pass

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete a value by key.

        Args:
            key: The key to delete
        """
        pass

    @abstractmethod
    async def list_keys(self, pattern: str = "*") -> List[str]:
        """List all keys matching the pattern.

        Args:
            pattern: The pattern to match keys against

        Returns:
            List of matching keys
        """
        pass