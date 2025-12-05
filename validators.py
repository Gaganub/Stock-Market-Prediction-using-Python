"""Input validators and verification functions"""
from typing import Any, Callable
import re

def validate_email(email: str) -> bool:
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_stock_symbol(symbol: str) -> bool:
    return bool(re.match(r'^[A-Z0-9&-]{1,10}$', symbol.strip()))

def validate_number_range(value: float, min_val: float, max_val: float) -> bool:
    return min_val <= value <= max_val
