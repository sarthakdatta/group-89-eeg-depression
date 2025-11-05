"""
Application Configuration
"""

from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """Application settings."""
    
    # API Settings
    app_name: str = "EEG Depression Classification API"
    app_version: str = "1.0.0"
    api_prefix: str = "/api/v1"
    
    # Model Settings
    model_path: str = "eeg_depression_model.pkl"
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
