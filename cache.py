"""In-memory caching utilities"""
import time
from typing import Dict, Any, Optional

class SimpleCache:
    def __init__(self, ttl: int = 3600):
        self.cache: Dict[str, tuple] = {}
        self.ttl = ttl
    
    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            del self.cache[key]
        return None
    
    def set(self, key: str, value: Any) -> None:
        self.cache[key] = (value, time.time())
