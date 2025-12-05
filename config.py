"""
Configuration module for FinalyticsBot
Handles all configuration settings and environment variables
"""

import os
from typing import Dict, Optional

# API Configuration
TELEGRAM_API_TOKEN: str = os.getenv('TELEGRAM_API_TOKEN', '1704757799:AAGJRzgiQP-m4YINSAfWrRsYbcikFtTJryo')

# Database Configuration
DATABASE_HOST: str = os.getenv('DB_HOST', 'localhost')
DATABASE_PORT: int = int(os.getenv('DB_PORT', 5432))
DATABASE_NAME: str = os.getenv('DB_NAME', 'finalyticsbot')

# File paths
DATA_DIR: str = os.path.join(os.path.dirname(__file__), 'data')
LOGS_DIR: str = os.path.join(os.path.dirname(__file__), 'logs')
CSV_SUBSCRIBER_PATH: str = os.path.join(DATA_DIR, 'subscriber_ids.csv')
CSV_ADMIN_PATH: str = os.path.join(DATA_DIR, 'admin_ids.csv')
CSV_DATASET_PATH: str = os.path.join(DATA_DIR, 'stockDataset_v1.csv')

# Logging Configuration
LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Bot Configuration
BOT_TIMEOUT: int = int(os.getenv('BOT_TIMEOUT', 30))
REQUEST_TIMEOUT: int = int(os.getenv('REQUEST_TIMEOUT', 10))

# Risk Profile Configuration
FINANCIAL_RISK_RANGES: Dict[str, tuple] = {
    'Very Low': (0, 8),
    'Low': (8, 12),
    'Moderate': (12, 15),
    'High': (15, 17),
    'Very High': (17, 20)
}

PSYCHOLOGICAL_RISK_RANGES: Dict[str, tuple] = {
    'Very Low': (0, 6),
    'Low': (6, 9),
    'Moderate': (9, 12),
    'High': (12, 14),
    'Very High': (14, 15)
}

# Stock Market Categories
MARKET_CAP_CATEGORIES: list = ['Largecap', 'Midcap', 'Smallcap']
RISK_LEVEL_CATEGORIES: list = ['Low Risk', 'Moderate Risk', 'High Risk']

# Feature flags
ENABLE_CACHING: bool = os.getenv('ENABLE_CACHING', 'True').lower() == 'true'
ENABLE_NOTIFICATIONS: bool = os.getenv('ENABLE_NOTIFICATIONS', 'True').lower() == 'true'

def get_config() -> Dict[str, any]:
    """Get all configuration as a dictionary"""
    return {
        'api_token': TELEGRAM_API_TOKEN,
        'db_host': DATABASE_HOST,
        'db_port': DATABASE_PORT,
        'db_name': DATABASE_NAME,
        'data_dir': DATA_DIR,
        'logs_dir': LOGS_DIR,
        'log_level': LOG_LEVEL,
        'bot_timeout': BOT_TIMEOUT,
    }
