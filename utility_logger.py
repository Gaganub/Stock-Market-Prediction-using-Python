"""
Logging utilities module for FinalyticsBot
Provides centralized logging configuration and utilities
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from config import LOG_LEVEL, LOG_FORMAT, LOGS_DIR
from typing import Optional

def setup_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """Setup and configure logger with file and console handlers"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL))
    
    formatter = logging.Formatter(LOG_FORMAT)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        os.makedirs(LOGS_DIR, exist_ok=True)
        file_path = os.path.join(LOGS_DIR, log_file)
        file_handler = RotatingFileHandler(file_path, maxBytes=10485760, backupCount=5)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Module logger
logger = setup_logger(__name__, 'bot.log')
