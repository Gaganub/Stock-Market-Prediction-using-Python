"""Application constants and configuration values.

Centralized location for all constant values used throughout the application.
"""

# Application Information
APP_NAME = "FinalyticsBot"
APP_VERSION = "2.0.0"
APP_AUTHOR = "Gaganub"

# API Configuration
API_BASE_URL = "https://api.example.com"
API_TIMEOUT = 30
API_MAX_RETRIES = 3
API_RETRY_DELAY = 1.0
API_RETRY_BACKOFF = 2.0

# Database Configuration
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "finalyticsbot"
DB_USER = "postgres"
DB_POOL_SIZE = 10
DB_MAX_OVERFLOW = 20

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = None
LOG_MAX_BYTES = 10485760  # 10MB
LOG_BACKUP_COUNT = 5

# Cache Configuration
CACHE_TTL = 3600  # 1 hour
CACHE_MAX_SIZE = 1000

# Prediction Configuration
PREDICTION_CONFIDENCE_THRESHOLD = 0.7
PREDICTION_LOOKBACK_DAYS = 90
PREDICTION_FORECAST_DAYS = 30

# Risk Assessment Configuration
RISK_LOW_THRESHOLD = 33
RISK_MEDIUM_THRESHOLD = 66
RISK_HIGH_THRESHOLD = 100

# Validation Configuration
MAX_EMAIL_LENGTH = 254
MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 128
STOCK_SYMBOL_PATTERN = r'^[A-Z0-9]{1,10}$'

# Request Configuration
REQUEST_TIMEOUT = 30
REQUEST_MAX_RETRIES = 3
REQUEST_BATCH_SIZE = 100

# Performance Configuration
EXECUTION_TIMEOUT = 300  # 5 minutes
WARNING_EXECUTION_TIME = 60  # 1 minute

# Status Codes
STATUS_SUCCESS = "success"
STATUS_ERROR = "error"
STATUS_PENDING = "pending"
STATUS_TIMEOUT = "timeout"

# Supported Stock Markets
SUPPORTED_MARKETS = {
    'NSE': 'National Stock Exchange (India)',
    'BSE': 'Bombay Stock Exchange (India)',
    'NYSE': 'New York Stock Exchange',
    'NASDAQ': 'NASDAQ',
}

# Technical Indicators
SMA_PERIODS = [20, 50, 200]
RSI_PERIOD = 14
MACD_FAST_PERIOD = 12
MACD_SLOW_PERIOD = 26
MACD_SIGNAL_PERIOD = 9

# Data Validation Rules
MIN_DATA_POINTS = 10
MAX_MISSING_DATA_PERCENT = 5
