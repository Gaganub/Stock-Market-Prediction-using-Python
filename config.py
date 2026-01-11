"""Configuration management for FinalyticsBot.

Module: config.py
Version: 2.1
Last Updated: January 2026
Author: Stock Market Prediction Team

Provides centralized configuration management for handling application settings,
environment variables, database connections, and runtime parameters.
"""
Centralized configuration module handling application settings,
environment variables, and runtime parameters.
"""
import os
from typing import Dict, Optional, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DatabaseConfig:
    """Database configuration."""
    host: str = "localhost"
    port: int = 5432
    name: str = "finalyticsbot"
    user: str = "postgres"
    password: str = ""
    
    @classmethod
    def from_env(cls) -> 'DatabaseConfig':
        """Create config from environment variables."""
        return cls(
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', '5432')),
            name=os.getenv('DB_NAME', 'finalyticsbot'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', '')
        )


@dataclass
class APIConfig:
    """API configuration."""
    api_key: str = ""
    api_timeout: int = 30
    max_retries: int = 3
    base_url: str = "https://api.example.com"
    
    @classmethod
    def from_env(cls) -> 'APIConfig':
        """Create config from environment variables."""
        return cls(
            api_key=os.getenv('API_KEY', ''),
            api_timeout=int(os.getenv('API_TIMEOUT', '30')),
            max_retries=int(os.getenv('MAX_RETRIES', '3')),
            base_url=os.getenv('BASE_URL', 'https://api.example.com')
        )


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format_str: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> 'LoggingConfig':
        """Create config from environment variables."""
        return cls(
            level=os.getenv('LOG_LEVEL', 'INFO'),
            format_str=os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            log_file=os.getenv('LOG_FILE')
        )


class Config:
    """Main configuration manager."""
    
    def __init__(self):
        """Initialize configuration."""
        self.env = os.getenv('APP_ENV', 'development')
        self.debug = os.getenv('DEBUG', 'False').lower() == 'true'
        self.database = DatabaseConfig.from_env()
        self.api = APIConfig.from_env()
        self.logging = LoggingConfig.from_env()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary.
        
        Returns:
            Dictionary representation of configuration.
        """
        return {
            'env': self.env,
            'debug': self.debug,
            'database': self.database.__dict__,
            'api': self.api.__dict__,
            'logging': self.logging.__dict__
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.
        
        Args:
            key: Configuration key (dot-separated for nested access).
            default: Default value if key not found.
            
        Returns:
            Configuration value or default.
        """
        parts = key.split('.')
        value = self.to_dict()
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        
        return value


# Global config instance
config = Config()


# Environment-specific configuration
class EnvironmentConfig:
    """Configuration handler for different environments."""
    
    @staticmethod
    def get_config(env: str = 'development'):
        """Load configuration for specified environment."""
        envs = {
            'development': {
                'debug': True,
                'db_pool_size': 5,
                'cache_ttl': 300,
                'log_level': 'DEBUG'
            },
            'production': {
                'debug': False,
                'db_pool_size': 20,
                'cache_ttl': 3600,
                'log_level': 'INFO'
            },
            'testing': {
                'debug': True,
                'db_pool_size': 1,
                'cache_ttl': 60,
                'log_level': 'DEBUG'
            }
        }
        return envs.get(env, envs['development'])
    
    @staticmethod
    def validate_config(config: dict) -> bool:
        """Validate configuration settings."""
        required_keys = ['debug', 'db_pool_size', 'cache_ttl', 'log_level']
        return all(key in config for key in required_keys)
