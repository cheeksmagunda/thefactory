"""Redis connection + pick caching + pipeline status + daily lock."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from functools import lru_cache

import redis

from src.config import settings

logger = logging.getLogger(__name__)

PICKS_KEY = "picks:{date}"
STATUS_KEY = "status:{date}"
LOCK_KEY = "lock:{date}"
TTL_SECONDS = 86_400  # 24h


@lru_cache
def get_redis() -> redis.Redis:
    return redis.from_url(settings.REDIS_URL, decode_responses=True)


def cache_picks(date_str: str, picks: list[dict]) -> None:
    r = get_redis()
    payload = json.dumps(picks, default=str)
    r.setex(PICKS_KEY.format(date=date_str), TTL_SECONDS, payload)


def picks_exist_for_today(date_str: str) -> dict | None:
    r = get_redis()
    data = r.get(PICKS_KEY.format(date=date_str))
    if not data:
        return None
    return {
        "date": date_str,
        "status": "cached",
        "picks": json.loads(data),
    }


def set_pipeline_status(date_str: str, status: str, error: str | None = None) -> None:
    r = get_redis()
    payload = {
        "date": date_str,
        "status": status,
        "error": error,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    r.setex(STATUS_KEY.format(date=date_str), TTL_SECONDS, json.dumps(payload))


def get_pipeline_status(date_str: str) -> dict:
    r = get_redis()
    raw = r.get(STATUS_KEY.format(date=date_str))
    if not raw:
        return {"date": date_str, "status": "not_run", "error": None}
    return json.loads(raw)


def acquire_daily_lock(date_str: str) -> bool:
    """
    Acquire a 24h lock so the pipeline only runs once per day.
    Returns True if we got the lock, False if someone else already has it.
    """
    r = get_redis()
    return bool(r.set(LOCK_KEY.format(date=date_str), "locked", nx=True, ex=TTL_SECONDS))


def release_daily_lock(date_str: str) -> None:
    r = get_redis()
    r.delete(LOCK_KEY.format(date=date_str))


def ping() -> bool:
    try:
        return bool(get_redis().ping())
    except Exception as e:  # noqa: BLE001
        logger.warning("Redis ping failed: %s", e)
        return False
