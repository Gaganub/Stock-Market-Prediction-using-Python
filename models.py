"""Data models for type safety and data validation.

This module contains dataclass definitions for various entities used throughout
the stock market prediction application, providing type safety and validation.
Uses Protocol for flexible interface definitions and enhanced type checking.
"""
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Protocol, runtime_checkable
from datetime import datetime
from enum import Enum
import json


@runtime_checkable
class Validatable(Protocol):
    """Protocol for validatable data objects."""
    def validate(self) -> bool:
        """Validate the data object."""
        ...


class RiskLevel(Enum):
    """Enumeration for risk levels with numeric values."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"
    
    @property
    def numeric_value(self) -> int:
        """Return numeric representation of risk level."""
        risk_map = {"low": 1, "medium": 2, "high": 3, "very_high": 4}
        return risk_map.get(self.value, 0)


@dataclass
class RiskProfile:
    """Data model for user risk profile assessment.
    
    Attributes:
        financial_risk: Financial risk tolerance level (low/medium/high).
        psychological_risk: Psychological risk tolerance level.
        score: Overall risk score (0-100).
        created_at: Timestamp of profile creation.
        metadata: Additional profile metadata.
    """
    financial_risk: str
    psychological_risk: str
    score: int
    created_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate risk profile data."""
        if not 0 <= self.score <= 100:
            raise ValueError(f"Risk score must be between 0 and 100, got {self.score}")
        if self.created_at is None:
            self.created_at = datetime.now()
        
        valid_risks = {"low", "medium", "high", "very_high"}
        if self.financial_risk not in valid_risks:
            raise ValueError(f"Invalid financial_risk: {self.financial_risk}")
        if self.psychological_risk not in valid_risks:
            raise ValueError(f"Invalid psychological_risk: {self.psychological_risk}")
    
    def __eq__(self, other: object) -> bool:
        """Compare two RiskProfile instances."""
        if not isinstance(other, RiskProfile):
            return NotImplemented
        return (self.financial_risk == other.financial_risk and
                self.psychological_risk == other.psychological_risk and
                self.score == other.score)
    
    def __hash__(self) -> int:
        """Make RiskProfile hashable."""
        return hash((self.financial_risk, self.psychological_risk, self.score))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert RiskProfile to dictionary."""
        return {
            'financial_risk': self.financial_risk,
            'psychological_risk': self.psychological_risk,
            'score': self.score,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RiskProfile':
        """Create RiskProfile from dictionary."""
        data_copy = data.copy()
        if 'created_at' in data_copy and isinstance(data_copy['created_at'], str):
            data_copy['created_at'] = datetime.fromisoformat(data_copy['created_at'])
        return cls(**data_copy)


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
        last_updated: Last update timestamp.
    """
    symbol: str
    name: str
    industry: str
    market_cap: str
    risk_level: str
    opinion: str
    last_updated: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate stock info on initialization."""
        if not self.symbol or len(self.symbol.strip()) == 0:
            raise ValueError("Stock symbol cannot be empty")
        if self.last_updated is None:
            self.last_updated = datetime.now()
    
    def __eq__(self, other: object) -> bool:
        """Compare two StockInfo instances."""
        if not isinstance(other, StockInfo):
            return NotImplemented
        return self.symbol == other.symbol
    
    def __hash__(self) -> int:
        """Make StockInfo hashable."""
        return hash(self.symbol)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert StockInfo to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StockInfo':
        """Create StockInfo from dictionary."""
        return cls(**data)


@dataclass
class PredictionResult:
    """Data model for stock price prediction results.
    
    Attributes:
        symbol: Stock ticker symbol.
        predicted_value: Predicted stock price value.
        confidence: Confidence score (0-1).
        timestamp: Prediction timestamp.
        prediction_date: Optional datetime of prediction.
        model_version: Version of the prediction model used.
    """
    symbol: str
    predicted_value: float
    confidence: float
    timestamp: str
    prediction_date: Optional[datetime] = None
    model_version: str = "1.0"
    
    def __post_init__(self):
        """Validate prediction result data."""
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")
        if self.predicted_value < 0:
            raise ValueError(f"Predicted value cannot be negative, got {self.predicted_value}")
        if self.prediction_date is None:
            self.prediction_date = datetime.now()
    
    def __eq__(self, other: object) -> bool:
        """Compare two PredictionResult instances."""
        if not isinstance(other, PredictionResult):
            return NotImplemented
        return (self.symbol == other.symbol and
                abs(self.predicted_value - other.predicted_value) < 1e-6 and
                abs(self.confidence - other.confidence) < 1e-6)
    
    def __hash__(self) -> int:
        """Make PredictionResult hashable."""
        return hash((self.symbol, round(self.predicted_value, 2)))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert PredictionResult to dictionary."""
        return {
            'symbol': self.symbol,
            'predicted_value': self.predicted_value,
            'confidence': self.confidence,
            'timestamp': self.timestamp,
            'prediction_date': self.prediction_date.isoformat() if self.prediction_date else None,
            'model_version': self.model_version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PredictionResult':
        """Create PredictionResult from dictionary."""
        data_copy = data.copy()
        if 'prediction_date' in data_copy and isinstance(data_copy['prediction_date'], str):
            data_copy['prediction_date'] = datetime.fromisoformat(data_copy['prediction_date'])
        return cls(**data_copy)


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
    
    def __post_init__(self):
        """Validate market data."""
        if self.high_price < self.low_price:
            raise ValueError(f"High price {self.high_price} cannot be less than low price {self.low_price}")
        if self.volume < 0:
            raise ValueError(f"Volume cannot be negative, got {self.volume}")
        if any(price < 0 for price in [self.open_price, self.close_price, self.high_price, self.low_price]):
            raise ValueError("Prices cannot be negative")
    
    def get_daily_change(self) -> float:
        """Calculate daily price change."""
        return self.close_price - self.open_price
    
    def get_daily_change_percent(self) -> float:
        """Calculate daily percentage change."""
        if self.open_price == 0:
            return 0.0
        return (self.get_daily_change() / self.open_price) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert MarketData to dictionary."""
        return asdict(self)


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
        analysis_date: Date of analysis.
    """
    symbol: str
    pe_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    debt_to_equity: Optional[float] = None
    roe: Optional[float] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    analysis_date: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate analysis metrics."""
        if self.analysis_date is None:
            self.analysis_date = datetime.now()
        
        # Validate ranges for optional metrics
        if self.pe_ratio is not None and self.pe_ratio < 0:
            raise ValueError(f"PE ratio cannot be negative, got {self.pe_ratio}")
        if self.dividend_yield is not None and self.dividend_yield < 0:
            raise ValueError(f"Dividend yield cannot be negative, got {self.dividend_yield}")
        if self.roe is not None and self.roe < -100:
            raise ValueError(f"ROE should typically be > -100%, got {self.roe}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert AnalysisMetrics to dictionary."""
        return {
            'symbol': self.symbol,
            'pe_ratio': self.pe_ratio,
            'dividend_yield': self.dividend_yield,
            'debt_to_equity': self.debt_to_equity,
            'roe': self.roe,
            'metrics': self.metrics,
            'analysis_date': self.analysis_date.isoformat() if self.analysis_date else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisMetrics':
        """Create AnalysisMetrics from dictionary."""
        data_copy = data.copy()
        if 'analysis_date' in data_copy and isinstance(data_copy['analysis_date'], str):
            data_copy['analysis_date'] = datetime.fromisoformat(data_copy['analysis_date'])
        return cls(**data_copy)
