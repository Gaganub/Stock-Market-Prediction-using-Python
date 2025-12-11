"""Data models for type safety and data validation.

This module contains dataclass definitions for various entities used throughout
the stock market prediction application, providing type safety and validation.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class RiskLevel(Enum):
    """Enumeration for risk levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class RiskProfile:
    """Data model for user risk profile assessment.
    
    Attributes:
        financial_risk: Financial risk tolerance level (low/medium/high).
        psychological_risk: Psychological risk tolerance level.
        score: Overall risk score (0-100).
    """
    financial_risk: str
    psychological_risk: str
    score: int
    
    def __post_init__(self):
        """Validate risk profile data."""
        if not 0 <= self.score <= 100:
            raise ValueError(f"Risk score must be between 0 and 100, got {self.score}")


@dataclass
class StockInfo:
    """Data model for stock information.
    
    Attributes:
        symbol: Stock ticker symbol (e.g., 'AAPL').
        name: Company name.
        industry: Industry classification.
        market_cap: Market capitalization string.
        risk_level: Risk level classification.
        opinion: Investment opinion.
    """
    symbol: str
    name: str
    industry: str
    market_cap: str
    risk_level: str
    opinion: str


@dataclass
class PredictionResult:
    """Data model for stock price prediction results.
    
    Attributes:
        symbol: Stock ticker symbol.
        predicted_value: Predicted stock price value.
        confidence: Confidence score (0-1).
        timestamp: Prediction timestamp.
    """
    symbol: str
    predicted_value: float
    confidence: float
    timestamp: str
    prediction_date: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate prediction result data."""
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")
        if self.predicted_value < 0:
            raise ValueError(f"Predicted value cannot be negative, got {self.predicted_value}")


@dataclass
class MarketData:
    """Data model for market data snapshots.
    
    Attributes:
        symbol: Stock ticker symbol.
        open_price: Opening price.
        close_price: Closing price.
        high_price: Highest price.
        low_price: Lowest price.
        volume: Trading volume.
        date: Date of market data.
    """
    symbol: str
    open_price: float
    close_price: float
    high_price: float
    low_price: float
    volume: int
    date: str


@dataclass
class AnalysisMetrics:
    """Data model for analysis metrics and indicators.
    
    Attributes:
        symbol: Stock ticker symbol.
        pe_ratio: Price-to-earnings ratio.
        dividend_yield: Dividend yield percentage.
        debt_to_equity: Debt-to-equity ratio.
        roe: Return on equity percentage.
        metrics: Additional custom metrics.
    """
    symbol: str
    pe_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    debt_to_equity: Optional[float] = None
    roe: Optional[float] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
