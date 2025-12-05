"""Data models for type safety"""
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class RiskProfile:
    financial_risk: str
    psychological_risk: str
    score: int

@dataclass
class StockInfo:
    symbol: str
    name: str
    industry: str
    market_cap: str
    risk_level: str
    opinion: str

@dataclass
class PredictionResult:
    symbol: str
    predicted_value: float
    confidence: float
    timestamp: str
