import logging
import sys
from enum import Enum
from typing import Any, Dict, Optional, TypeVar, Union
from pathlib import Path
import json
from datetime import datetime

# Type definitions
T = TypeVar('T')
JSONValue = Union[str, int, float, bool, None, Dict[str, Any], list[Any]]
ConfigDict = Dict[str, Any]

class LogLevel(str, Enum):
    """Log levels supported by the application."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class ErrorCode(str, Enum):
    """Error codes for the application."""
    # General errors
    UNKNOWN_ERROR = "UNKNOWN_ERROR"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"

    # Worker errors
    WORKER_ERROR = "WORKER_ERROR"
    TASK_ERROR = "TASK_ERROR"
    PROCESSING_ERROR = "PROCESSING_ERROR"

    # Service errors
    SERVICE_ERROR = "SERVICE_ERROR"
    STARTUP_ERROR = "STARTUP_ERROR"
    SHUTDOWN_ERROR = "SHUTDOWN_ERROR"

    # Storage errors
    STORAGE_ERROR = "STORAGE_ERROR"
    CONNECTION_ERROR = "CONNECTION_ERROR"
    QUERY_ERROR = "QUERY_ERROR"

class DeepCrawlError(Exception):
    """Base exception class for DeepCrawl application."""

    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize the error.

        Args:
            message: Error message
            error_code: Error code from ErrorCode enum
            details: Additional error details
        """
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = datetime.utcnow()
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary.

        Returns:
            Dictionary representation of the error
        """
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }

    def __str__(self) -> str:
        """String representation of the error.

        Returns:
            Formatted error string
        """
        return f"{self.error_code}: {self.message}"

def setup_logging(
    log_level: Union[str, LogLevel] = LogLevel.INFO,
    log_file: Optional[Path] = None,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> None:
    """Set up logging configuration.

    Args:
        log_level: Logging level
        log_file: Optional log file path
        log_format: Log message format
    """
    # Convert string log level to LogLevel enum if needed
    if isinstance(log_level, str):
        log_level = LogLevel(log_level.upper())

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level.value)

    # Clear existing handlers
    root_logger.handlers = []

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Add file handler if log file is specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

def load_config(config_path: Path) -> ConfigDict:
    """Load configuration from a JSON file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Configuration dictionary

    Raises:
        DeepCrawlError: If configuration file cannot be loaded or is invalid
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        raise DeepCrawlError(
            f"Configuration file not found: {config_path}",
            ErrorCode.CONFIGURATION_ERROR
        )
    except json.JSONDecodeError:
        raise DeepCrawlError(
            f"Invalid JSON in configuration file: {config_path}",
            ErrorCode.CONFIGURATION_ERROR
        )
    except Exception as e:
        raise DeepCrawlError(
            f"Error loading configuration: {str(e)}",
            ErrorCode.CONFIGURATION_ERROR
        )

def validate_config(config: ConfigDict, required_keys: list[str]) -> None:
    """Validate configuration dictionary.

    Args:
        config: Configuration dictionary to validate
        required_keys: List of required configuration keys

    Raises:
        DeepCrawlError: If configuration is invalid
    """
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise DeepCrawlError(
            f"Missing required configuration keys: {', '.join(missing_keys)}",
            ErrorCode.VALIDATION_ERROR,
            {"missing_keys": missing_keys}
        )