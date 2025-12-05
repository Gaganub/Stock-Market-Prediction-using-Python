"""Function decorators for common operations"""
import functools
from typing import Callable, Any
from utility_logger import logger

def retry(max_attempts: int = 3):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    logger.warning(f"Retry {attempt + 1}/{max_attempts} for {func.__name__}")
            return None
        return wrapper
    return decorator

def log_execution(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        logger.info(f"Executing {func.__name__}")
        return func(*args, **kwargs)
    return wrapper
