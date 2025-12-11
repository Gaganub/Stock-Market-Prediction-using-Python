"""Error handling and custom exceptions for FinalyticsBot.

This module defines custom exception classes and error handling utilities
for the stock market prediction application.
"""
from typing import Optional, Type
import logging
import traceback


logger = logging.getLogger(__name__)


class FinalyticsBotException(Exception):
    """Base exception for all FinalyticsBot errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None):
        """Initialize FinalyticsBotException.
        
        Args:
            message: Error message.
            error_code: Optional error code for tracking.
        """
        self.message = message
        self.error_code = error_code or "UNKNOWN"
        super().__init__(self.message)
    
    def __str__(self) -> str:
        """Return string representation of exception."""
        return f"[{self.error_code}] {self.message}"


class ConfigurationError(FinalyticsBotException):
    """Raised when configuration is invalid."""
    def __init__(self, message: str):
        super().__init__(message, "CONFIG_ERROR")


class DataProcessingError(FinalyticsBotException):
    """Raised when data processing fails."""
    def __init__(self, message: str):
        super().__init__(message, "DATA_PROC_ERROR")


class APIError(FinalyticsBotException):
    """Raised when API call fails."""
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message, "API_ERROR")
        self.status_code = status_code


class ValidationError(FinalyticsBotException):
    """Raised when validation fails."""
    def __init__(self, message: str):
        super().__init__(message, "VALIDATION_ERROR")


class PredictionError(FinalyticsBotException):
    """Raised when prediction generation fails."""
    def __init__(self, message: str):
        super().__init__(message, "PREDICTION_ERROR")


class DatabaseError(FinalyticsBotException):
    """Raised when database operations fail."""
    def __init__(self, message: str):
        super().__init__(message, "DATABASE_ERROR")


class CacheError(FinalyticsBotException):
    """Raised when cache operations fail."""
    def __init__(self, message: str):
        super().__init__(message, "CACHE_ERROR")


def handle_exception(exception: Exception, context: Optional[str] = None) -> None:
    """Handle exceptions with logging and tracking.
    
    Args:
        exception: Exception to handle.
        context: Additional context about where exception occurred.
    """
    error_message = f"Exception occurred: {str(exception)}"
    if context:
        error_message += f" | Context: {context}"
    
    logger.error(error_message)
    logger.debug(traceback.format_exc())


def safe_execute(func, *args, default_return=None, error_context: Optional[str] = None, **kwargs):
    """Safely execute a function with error handling.
    
    Args:
        func: Function to execute.
        *args: Positional arguments for function.
        default_return: Value to return if exception occurs.
        error_context: Context for error logging.
        **kwargs: Keyword arguments for function.
        
    Returns:
        Function return value or default_return if error occurs.
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        handle_exception(e, error_context)
        return default_return
