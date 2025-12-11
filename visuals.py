"""Visualization utilities for stock market data.

Provides functions for plotting and visualizing stock predictions and analysis.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def plot_price_prediction(historical: pd.DataFrame, predicted: pd.DataFrame,
                         title: str = "Stock Price Prediction",
                         figsize: Tuple[int, int] = (12, 6)) -> None:
    """Plot historical and predicted prices.
    
    Args:
        historical: DataFrame with historical prices.
        predicted: DataFrame with predicted prices.
        title: Plot title.
        figsize: Figure size as (width, height).
    """
    plt.figure(figsize=figsize)
    plt.plot(historical['Date'], historical['Close'], label='Historical', marker='o')
    plt.plot(predicted['Date'], predicted['Predicted'], label='Predicted', marker='s', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_confidence_interval(dates, predictions, upper_bound, lower_bound,
                           title: str = "Prediction with Confidence Interval",
                           figsize: Tuple[int, int] = (12, 6)) -> None:
    """Plot predictions with confidence intervals.
    
    Args:
        dates: Array of dates.
        predictions: Array of predicted values.
        upper_bound: Upper confidence bound.
        lower_bound: Lower confidence bound.
        title: Plot title.
        figsize: Figure size.
    """
    plt.figure(figsize=figsize)
    plt.plot(dates, predictions, 'b-', label='Prediction')
    plt.fill_between(dates, lower_bound, upper_bound, alpha=0.3, label='95% Confidence')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_performance_metrics(metrics: dict, figsize: Tuple[int, int] = (10, 6)) -> None:
    """Plot model performance metrics.
    
    Args:
        metrics: Dictionary of metric names and values.
        figsize: Figure size.
    """
    fig, ax = plt.subplots(figsize=figsize)
    names = list(metrics.keys())
    values = list(metrics.values())
    
    bars = ax.bar(names, values)
    ax.set_ylabel('Value')
    ax.set_title('Model Performance Metrics')
    ax.set_ylim(0, 1)
    
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()


def plot_volatility(df: pd.DataFrame, window: int = 20,
                   figsize: Tuple[int, int] = (12, 6)) -> None:
    """Plot rolling volatility.
    
    Args:
        df: DataFrame with price data.
        window: Rolling window size.
        figsize: Figure size.
    """
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(window=window).std()
    
    plt.figure(figsize=figsize)
    plt.subplot(2, 1, 1)
    plt.plot(df['Date'], df['Close'])
    plt.title('Stock Price')
    plt.ylabel('Price')
    
    plt.subplot(2, 1, 2)
    plt.plot(df['Date'], df['Volatility'])
    plt.title('Rolling Volatility')
    plt.ylabel('Volatility')
    plt.xlabel('Date')
    
    plt.tight_layout()
