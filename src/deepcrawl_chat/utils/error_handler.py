# custom error handler
import sys

class DocumentLoadingError(Exception):
    """Error raised when document loading fails."""
    def __init__(self, message="Error loading document", *args: object) -> None:
        super().__init__(message, *args)
        self.message = message
        self.args = args

