"""Input validators and verification functions.

This module provides validation functions for various input types used throughout
the stock market prediction application, with comprehensive error handling.
"""
from typing import Any, Callable, Optional
import re


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


def validate_email(email: str) -> bool:
    """Validate email format.
    
    Args:
        email: Email address to validate.
        
    Returns:
        bool: True if valid, False otherwise.
        
    Raises:
        ValidationError: If email is not a string.
    """
    if not isinstance(email, str):
        raise ValidationError(f"Email must be a string, got {type(email).__name__}")
    pattern = r'^[a-zA-Z0-9._%-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def validate_stock_symbol(symbol: str) -> bool:
    """Validate stock ticker symbol format.
    
    Args:
        symbol: Stock symbol to validate.
        
    Returns:
        bool: True if valid, False otherwise.
        
    Raises:
        ValidationError: If symbol is not a string or exceeds length.
    """
    if not isinstance(symbol, str):
        raise ValidationError(f"Symbol must be a string, got {type(symbol).__name__}")
    symbol = symbol.strip().upper()
    return bool(re.match(r'^[A-Z0-9]{1,10}$', symbol))


def validate_number_range(value: float, min_val: float = None, max_val: float = None) -> bool:
    """Validate that a number falls within a specified range.
    
    Args:
        value: Number to validate.
        min_val: Minimum allowed value (inclusive).
        max_val: Maximum allowed value (inclusive).
        
    Returns:
        bool: True if within range, False otherwise.
        
    Raises:
        ValidationError: If value is not numeric or range is invalid.
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(f"Value must be numeric, got {type(value).__name__}")
    
    if min_val is not None and max_val is not None:
        if min_val > max_val:
            raise ValidationError(f"Invalid range: min ({min_val}) > max ({max_val})")
        return min_val <= value <= max_val
    elif min_val is not None:
        return value >= min_val
    elif max_val is not None:
        return value <= max_val
    return True


def validate_list_not_empty(items: list) -> bool:
    """Validate that a list is not empty.
    
    Args:
        items: List to validate.
        
    Returns:
        bool: True if list is not empty, False otherwise.
        
    Raises:
        ValidationError: If items is not a list.
    """
    if not isinstance(items, list):
        raise ValidationError(f"Expected list, got {type(items).__name__}")
    return len(items) > 0


def validate_dict_keys(data: dict, required_keys: list) -> bool:
    """Validate that a dictionary contains all required keys.
    
    Args:
        data: Dictionary to validate.
        required_keys: List of required keys.
        
    Returns:
        bool: True if all keys present, False otherwise.
        
    Raises:
        ValidationError: If data is not a dict or keys missing.
    """
    if not isinstance(data, dict):
        raise ValidationError(f"Expected dict, got {type(data).__name__}")
    
    missing_keys = set(required_keys) - set(data.keys())
    if missing_keys:
        raise ValidationError(f"Missing required keys: {missing_keys}")
    return True
