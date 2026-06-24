# ============================================================================
# AI Stock Analysis Platform - Configuration
# ============================================================================
from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # --- App ---
    APP_NAME: str = "AIStockAnalysis"
    APP_VERSION: str = "1.4.0"
    APP_ENV: str = "development"
    APP_DEBUG: bool = True

    # --- Server ---
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4

    # --- Database ---
    DATABASE_URL: str = "postgresql+asyncpg://stock_user:stock_pass@localhost:5432/stock_analysis"
    DATABASE_URL_SYNC: str = "postgresql://stock_user:stock_pass@localhost:5432/stock_analysis"

    # --- Redis ---
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_CACHE_TTL: int = 60
    REDIS_CONFIG_TTL: int = 300

    # --- Celery ---
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"

    # --- JWT ---
    JWT_SECRET_KEY: str = "change-me-to-a-random-secret-key"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 1440

    # --- Encryption ---
    ENCRYPTION_KEY: str = "change-me-32-byte-key-for-aes256!"

    # --- AI Model Defaults ---
    DEFAULT_LLM_PROVIDER: str = "opencodezen"
    DEFAULT_LLM_MODEL: str = "opencode/gpt-5.5"
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    DEEPSEEK_API_KEY: Optional[str] = None
    DEEPSEEK_BASE_URL: Optional[str] = None
    OPENROUTER_API_KEY: Optional[str] = None
    OPENROUTER_BASE_URL: Optional[str] = None
    OPENCODEZEN_API_KEY: Optional[str] = None
    OPENCODEZEN_BASE_URL: str = "https://opencode.ai/zen/v1"

    # --- Tushare ---
    TUSHARE_TOKEN: Optional[str] = None

    # --- ChromaDB ---
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"

    # --- Data Retention ---
    NEWS_RETENTION_HOURS: int = 72
    KOL_OPINION_RETENTION_DAYS: int = 30
    KLINE_CACHE_DAYS: int = 90

    # --- Scheduling ---
    HOT_TOPIC_UPDATE_INTERVAL_MINUTES: int = 15
    KOL_CRAWL_INTERVAL_MINUTES: int = 30
    REPORT_UPDATE_INTERVAL_HOURS: int = 24

    # --- Monitoring ---
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090

    # --- Logging ---
    LOG_LEVEL: str = "INFO"           # DEBUG / INFO / WARNING / ERROR
    LOG_FILE: str = "./logs/app.log"  # File path, set to "" to disable file logging
    LOG_FORMAT: str = "%(asctime)s [%(levelname)-5s] %(name)s | %(message)s"
    LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()
