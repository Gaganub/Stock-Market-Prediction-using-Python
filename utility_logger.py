"""Structured logging utilities for FinalyticsBot.

Centralized logging configuration with support for file and console output.
"""
import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional
from constants import LOG_LEVEL, LOG_FORMAT, LOG_FILE, LOG_MAX_BYTES, LOG_BACKUP_COUNT


def setup_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """Configure logger with file and console handlers.
    
    Args:
        name: Logger name.
        log_file: Optional log file path.
        
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    
    formatter = logging.Formatter(LOG_FORMAT)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_file or LOG_FILE:
        file_path = log_file or LOG_FILE
        os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
        
        file_handler = RotatingFileHandler(
            file_path,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class StructuredLogger:
    """Wrapper for structured logging with context."""
    
    def __init__(self, name: str):
        """Initialize structured logger.
        
        Args:
            name: Logger name.
        """
        self.logger = setup_logger(name)
        self.context = {}
    
    def set_context(self, **kwargs) -> None:
        """Set logging context.
        
        Args:
            **kwargs: Context variables.
        """
        self.context.update(kwargs)
    
    def _format_message(self, msg: str) -> str:
        """Format message with context.
        
        Args:
            msg: Log message.
            
        Returns:
            Formatted message.
refactor: Implement structured logging with context support        if self.context:
            context_str = ' | '.join(f"{k}={v}" for k, v in self.context.items())
            return f"{msg} | {context_str}"
        return msg
    
    def debug(self, msg: str) -> None:
        """Log debug message."""
        self.logger.debug(self._format_message(msg))
    
    def info(self, msg: str) -> None:
        """Log info message."""
        self.logger.info(self._format_message(msg))
    
    def warning(self, msg: str) -> None:
        """Log warning message."""
        self.logger.warning(self._format_message(msg))
    
    def error(self, msg: str) -> None:
        """Log error message."""
        self.logger.error(self._format_message(msg))
    
    def critical(self, msg: str) -> None:
        """Log critical message."""
        self.logger.critical(self._format_message(msg))
