"""In-memory caching utilities for performance optimization.

Module Version: 2.0
Author: Stock Market Prediction Team
Date: January 2026
Description: This module provides caching mechanisms to reduce redundant computations
and improve application performance.
"""
This module provides caching mechanisms to reduce redundant computations
and improve application performance.
"""
import time
from typing import Dict, Any, Optional, Callable, TypeVar
from functools import wraps
import threading

T = TypeVar('T')


class SimpleCache:
    """Simple in-memory cache with TTL support.
    
    Attributes:
        ttl: Time-to-live in seconds for cached items.
        cache: Dictionary storing cached key-value pairs.
    """
    
    def __init__(self, ttl: int = 3600):
        """Initialize cache with time-to-live setting.
        
        Args:
            ttl: Time-to-live for cached items in seconds.
        """
        self.cache: Dict[str, tuple] = {}
        self.ttl = ttl
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if exists and not expired.
        
        Args:
            key: Cache key.
            
        Returns:
            Cached value or None if not found or expired.
        """
        with self._lock:
            if key in self.cache:
                value, timestamp = self.cache[key]
                if time.time() - timestamp < self.ttl:
                    return value
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Store value in cache.
        
        Args:
            key: Cache key.
            value: Value to cache.
        """
        with self._lock:
            self.cache[key] = (value, time.time())
    
    def delete(self, key: str) -> None:
        """Delete item from cache.
        
        Args:
            key: Cache key to delete.
        """
        with self._lock:
            if key in self.cache:
                del self.cache[key]
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self.cache.clear()
    
    def get_size(self) -> int:
        """Get number of items currently cached.
        
        Returns:
            Number of cached items.
        """
        with self._lock:
            # Remove expired items before returning size
            current_time = time.time()
            expired_keys = [
                k for k, (_, ts) in self.cache.items()
                if current_time - ts >= self.ttl
            ]
            for k in expired_keys:
                del self.cache[k]
            return len(self.cache)


def cache_result(ttl: int = 3600):
    """Decorator to cache function results.
    
    Args:
        ttl: Time-to-live for cached results in seconds.
        
    Returns:
        Decorated function with caching.
    """
    cache = SimpleCache(ttl=ttl)
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # Try to get from cache
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                return cached_value
            
            # Compute and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result)
            return result
        
        return wrapper
    return decorator



# Memoization decorator with TTL and statistics
def memoize(ttl: int = 3600):
    """Decorator for caching function results with TTL support."""
    def decorator(func):
        cache = {}
        timestamps = {}
        stats = {'hits': 0, 'misses': 0}
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = str((args, sorted(kwargs.items())))
            now = time.time()
            
            if key in cache and (now - timestamps[key]) < ttl:
                stats['hits'] += 1
                return cache[key]
            
            stats['misses'] += 1
            result = func(*args, **kwargs)
            cache[key] = result
            timestamps[key] = now
            return result
        
        wrapper.cache_stats = lambda: stats
        wrapper.clear_cache = lambda: (cache.clear(), timestamps.clear())
        return wrapper
    return decorator
