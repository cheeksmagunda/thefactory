"""App configuration. Loaded from environment variables."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Redis
    REDIS_URL: str = Field(default="redis://localhost:6379/0")

    # BallparkPal premium
    BALLPARKPAL_EMAIL: str | None = None
    BALLPARKPAL_PASSWORD: str | None = None
    BALLPARKPAL_DATA_DIR: str = "data/ballparkpal"

    # Odds (future)
    FANDUEL_API_KEY: str | None = None
    DRAFTKINGS_API_KEY: str | None = None

    # App behavior
    LOG_LEVEL: str = "INFO"
    TIMEZONE: str = "America/Chicago"

    # Scheduler
    SCHEDULE_HOUR_UTC: int = 14  # 14:00 UTC = 9:00 AM CT
    SCHEDULE_MINUTE_UTC: int = 0
    DISABLE_SCHEDULER: bool = False

    # Local cache paths for weekly pulls
    CACHE_DIR: str = "data/cache"
    BAT_TRACKING_TTL_DAYS: int = 7
    PITCHER_PROFILE_TTL_DAYS: int = 7

    @property
    def cache_path(self) -> Path:
        p = Path(self.CACHE_DIR)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def bpp_data_path(self) -> Path:
        p = Path(self.BALLPARKPAL_DATA_DIR)
        p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
