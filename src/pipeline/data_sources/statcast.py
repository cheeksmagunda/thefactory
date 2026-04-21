"""Statcast pitcher profiles via pybaseball.

Weekly refresh: pulls season-level pitcher metrics that drive vulnerability
scoring — fastball velocity, barrel% allowed, hard-hit% allowed, HR/9, FB%.
Pitch entropy is computed locally from the pitch type distribution.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import date, datetime, timedelta
from typing import Any

from src.config import settings
from src.pipeline import DataSourceError

logger = logging.getLogger(__name__)

CACHE_FILE = "statcast_pitchers.json"


def fetch_pitcher_profiles(force_refresh: bool = False) -> dict[str, dict[str, Any]]:
    """
    Return pitcher profiles keyed by normalized pitcher name.

    Shape per pitcher:
        {
          "fb_velo": 93.4,
          "barrel_rate_allowed": 0.065,
          "hard_hit_rate_allowed": 0.38,
          "hr_per_9": 1.25,
          "fb_pct_allowed": 0.37,
          "breaking_zone_rate": 0.45,
          "pitch_entropy": 1.62,
        }

    Raises:
        DataSourceError: on pybaseball failure with no fresh cache.
    """
    cached = _read_cache() if not force_refresh else None
    if cached:
        return cached

    try:
        from pybaseball import pitching_stats
    except ImportError as e:  # pragma: no cover
        raise DataSourceError(f"pybaseball not installed: {e}") from e

    try:
        season = date.today().year
        df = pitching_stats(season, qual=20)
    except Exception as e:  # noqa: BLE001
        raise DataSourceError(f"Statcast/FanGraphs pitching pull failed: {e}") from e

    if df is None or len(df) == 0:
        raise DataSourceError("pitching_stats returned empty.")

    out: dict[str, dict[str, Any]] = {}
    for row in df.to_dict(orient="records"):
        name = str(row.get("Name") or "").strip()
        if not name:
            continue
        out[name] = {
            "fb_velo": _num(row.get("FBv") or row.get("vFA (pi)")),
            "barrel_rate_allowed": _pct(row.get("Barrel%")),
            "hard_hit_rate_allowed": _pct(row.get("HardHit%") or row.get("Hard%")),
            "hr_per_9": _num(row.get("HR/9")),
            "fb_pct_allowed": _pct(row.get("FB%")),
            "breaking_zone_rate": _pct(row.get("Zone%")),
            # Entropy derived from usage if present; otherwise neutral default.
            "pitch_entropy": _entropy_from_usage(row),
        }

    if not out:
        raise DataSourceError("pitching_stats yielded no usable rows.")

    _write_cache(out)
    return out


def _entropy_from_usage(row: dict[str, Any]) -> float:
    """
    Approximate Shannon entropy over primary pitch-type usage. Uses FanGraphs
    pitch% columns when available; falls back to a neutral mid-value (1.5).
    """
    keys = ["FA%", "FT%", "FC%", "SI%", "SL%", "CU%", "CH%", "KN%", "XX%"]
    probs: list[float] = []
    for k in keys:
        v = row.get(k)
        if v is None:
            continue
        try:
            p = float(v)
        except (TypeError, ValueError):
            continue
        if p > 1.0:
            p = p / 100.0
        if p > 0:
            probs.append(p)

    if not probs:
        return 1.5

    total = sum(probs)
    if total <= 0:
        return 1.5
    probs = [p / total for p in probs]
    return -sum(p * math.log(p) for p in probs if p > 0)


def _cache_file() -> Any:
    return settings.cache_path / CACHE_FILE


def _read_cache() -> dict[str, dict[str, Any]] | None:
    fpath = _cache_file()
    if not fpath.exists():
        return None
    age = datetime.now() - datetime.fromtimestamp(fpath.stat().st_mtime)
    if age > timedelta(days=settings.PITCHER_PROFILE_TTL_DAYS):
        return None
    try:
        with fpath.open() as f:
            return json.load(f)
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to read pitcher-profile cache: %s", e)
        return None


def _write_cache(data: dict[str, dict[str, Any]]) -> None:
    fpath = _cache_file()
    try:
        with fpath.open("w") as f:
            json.dump(data, f)
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to write pitcher-profile cache: %s", e)


def _num(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _pct(v: Any, default: float = 0.0) -> float:
    """FanGraphs percentage columns arrive as 0.065 or 6.5 depending on source.
    Normalize to fractional form (0.065)."""
    try:
        if v is None:
            return default
        f = float(v)
        return f / 100.0 if f > 1.0 else f
    except (TypeError, ValueError):
        return default
