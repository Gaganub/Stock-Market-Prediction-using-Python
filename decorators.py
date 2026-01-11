"""Function decorators for common operations.

Module: decorators.py v2.0
Author: Stock Market Prediction Team
Updated: January 2026

Provides reusable decorators for retry logic, logging, and performance monitoring.
"""
This module provides reusable decorators for retry logic, logging,
performance monitoring, and other cross-cutting concerns.
"""
import functools
import logging
import time
from typing import Callable, Any, Optional

logger = logging.getLogger(__name__)


def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator to retry function execution on failure.
    
    Args:
        max_attempts: Maximum number of retry attempts.
        delay: Initial delay between retries in seconds.
        backoff: Multiplier for delay after each retry.
        
    Returns:
        Decorated function that retries on exception.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts:
                        logger.warning(
                            f"Attempt {attempt}/{max_attempts} failed for {func.__name__}: {str(e)}. "
                            f"Retrying in {current_delay}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}"
                        )
            
            raise last_exception
        
        return wrapper
    return decorator


def log_execution(log_level: int = logging.INFO):
    """Decorator to log function execution.
    
    Args:
        log_level: Logging level to use.
        
    Returns:
        Decorated function that logs execution details.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger.log(log_level, f"Executing {func.__name__} with args={args}, kwargs={kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.log(log_level, f"{func.__name__} completed successfully")
                return result
            except Exception as e:
                logger.error(f"{func.__name__} raised {type(e).__name__}: {str(e)}")
                raise
        
        return wrapper
    return decorator


def timing(func: Callable) -> Callable:
    """Decorator to measure function execution time.
    
    Args:
        func: Function to measure.
        
    Returns:
        Decorated function that logs execution time.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
        finally:
            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"{func.__name__} took {duration:.4f} seconds")
        return result
    
    return wrapper


def validate_types(**type_checks):
    """Decorator to validate argument types.
    
    Args:
        **type_checks: Mapping of argument names to expected types.
        
    Returns:
        Decorated function that validates argument types.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Check keyword arguments
            for arg_name, expected_type in type_checks.items():
                if arg_name in kwargs:
                    if not isinstance(kwargs[arg_name], expected_type):
                        raise TypeError(
                            f"Argument '{arg_name}' must be {expected_type.__name__}, "
                            f"got {type(kwargs[arg_name]).__name__}"
                        )
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator
