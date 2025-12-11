"""Data utilities for CSV processing and data management.

Provides functions for loading, validating, and processing stock market data.
"""
import pandas as pd
import csv
from typing import Dict, List, Any, Optional, Tuple
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def read_csv_safe(file_path: str) -> Optional[Dict[str, Any]]:
    """Safely read CSV file and return as dictionary.
    
    Args:
        file_path: Path to CSV file.
        
    Returns:
        Dictionary with CSV data or None if error.
    """
    try:
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return None
        
        df = pd.read_csv(file_path)
        return df.to_dict(orient='records')
    except Exception as e:
        logger.error(f"Error reading CSV: {str(e)}")
        return None


def load_stock_data(file_path: str) -> Optional[pd.DataFrame]:
    """Load stock market data from CSV.
    
    Args:
        file_path: Path to stock data CSV.
        
    Returns:
        DataFrame with stock data or None.
    """
    try:
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values('Date', inplace=True)
        return df
    except Exception as e:
        logger.error(f"Error loading stock data: {str(e)}")
        return None


def validate_data_quality(df: pd.DataFrame, min_rows: int = 10) -> Tuple[bool, List[str]]:
    """Validate data quality and completeness.
    
    Args:
        df: DataFrame to validate.
        min_rows: Minimum required rows.
        
    Returns:
        Tuple of (is_valid, error_messages).
    """
    errors = []
    
    if df is None or df.empty:
        errors.append("DataFrame is empty")
        return False, errors
    
    if len(df) < min_rows:
        errors.append(f"Insufficient rows: {len(df)} < {min_rows}")
    
    # Check for required columns
    required_cols = ['Date', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")
    
    # Check for null values
    null_counts = df[required_cols].isnull().sum()
    if null_counts.any():
        for col, count in null_counts[null_counts > 0].items():
            errors.append(f"Null values in {col}: {count}")
    
    return len(errors) == 0, errors


def resample_data(df: pd.DataFrame, frequency: str = 'D') -> pd.DataFrame:
    """Resample time series data to different frequency.
    
    Args:
        df: DataFrame with time series data.
        frequency: Resampling frequency (D=daily, W=weekly, M=monthly).
        
    Returns:
        Resampled DataFrame.
    """
    df_resampled = df.resample(frequency).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })
    return df_resampled.dropna()


def calculate_returns(df: pd.DataFrame, column: str = 'Close') -> pd.DataFrame:
    """Calculate daily returns from price data.
    
    Args:
        df: DataFrame with price data.
        column: Column to calculate returns for.
        
    Returns:
        DataFrame with returns column added.
    """
    df_copy = df.copy()
    df_copy['Returns'] = df_copy[column].pct_change()
    return df_copy


def normalize_data(df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
    """Normalize data using min-max scaling.
    
    Args:
        df: DataFrame to normalize.
        columns: Columns to normalize (None = all numeric).
        
    Returns:
        Normalized DataFrame.
    """
    df_norm = df.copy()
    cols_to_norm = columns or df.select_dtypes(include=['number']).columns
    
    for col in cols_to_norm:
        if col in df_norm.columns:
            min_val = df_norm[col].min()
            max_val = df_norm[col].max()
            if max_val != min_val:
                df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
    
    return df_norm
