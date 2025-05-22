from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, TypeVar

T = TypeVar('T')

class Worker(Protocol):
    """Interface for worker implementations."""

    @abstractmethod
    async def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task and return the result.

        Args:
            task: The task to process

        Returns:
            The result of processing the task
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the worker is healthy.

        Returns:
            True if the worker is healthy, False otherwise
        """
        pass

class Service(Protocol):
    """Interface for service implementations."""

    @abstractmethod
    async def start(self) -> None:
        """Start the service."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the service."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the service is healthy.

        Returns:
            True if the service is healthy, False otherwise
        """
        pass

class Storage(Protocol):
    """Interface for storage implementations."""

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