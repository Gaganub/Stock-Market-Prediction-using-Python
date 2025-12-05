"""
Data utilities module
Provides functions for CSV processing and data management
"""

import pandas as pd
import csv
from typing import Dict, List, Any
import os
from utility_logger import logger

def read_csv_safe(file_path: str) -> Dict[str, Any]:
    """Safely read CSV file and return as dictionary"""
    try:
        if os.path.exists(file_path):
            data = pd.read_csv(file_path)
            return dict(data)
        else:
            logger.warning(f"File not found: {file_path}")
            return {}
    except Exception as e:
        logger.error(f"Error reading CSV {file_path}: {str(e)}")
        return {}

def write_csv(data: Dict[str, Any], file_path: str) -> bool:
    """Write data to CSV file"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)
        logger.info(f"Data written to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error writing CSV {file_path}: {str(e)}")
        return False

def validate_data_integrity(data: Dict[str, List]) -> bool:
    """Validate that all columns have same length"""
    if not data:
        return False
    lengths = [len(v) for v in data.values()]
    return len(set(lengths)) == 1
