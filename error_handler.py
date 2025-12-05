"""
Error handling and custom exceptions for FinalyticsBot
"""

from typing import Optional, Type
from utility_logger import logger

class FinalyticsBotException(Exception):
    """Base exception for FinalyticsBot"""
    pass

class ConfigurationError(FinalyticsBotException):
    """Raised when configuration is invalid"""
    pass

class DataProcessingError(FinalyticsBotException):
    """Raised when data processing fails"""
    pass

class APIError(FinalyticsBotException):
    """Raised when API call fails"""
    pass

def handle_exception(exc: Exception, context: str = "") -> None:
    """Centralized exception handling with logging"""
    logger.error(f"Error in {context}: {type(exc).__name__}: {str(exc)}")
    if isinstance(exc, FinalyticsBotException):
        raise exc
    else:
        raise FinalyticsBotException(f"Unexpected error: {str(exc)}")
