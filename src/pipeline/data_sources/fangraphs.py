"""FanGraphs bat tracking data via pybaseball.

Weekly refresh: pulls season-level batting stats including Pull%, Bat Speed,
Blast Contact%, and exit velocity. These are process-based metrics that signal
HR ability before outcomes catch up — the core edge of the Nuke E model.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from typing import Any

from src.config import settings
from src.pipeline import DataSourceError

logger = logging.getLogger(__name__)

COLUMN_MAP = {
    "Name": "name",
    "Team": "team",
    "Pull%": "pull_pct",
    "Pull pct": "pull_pct",
    "Bat Speed (MPH)": "bat_speed",
    "BatSpeed": "bat_speed",
    "Blast%": "blast_contact_pct",
    "Blast Contact %": "blast_contact_pct",
    "Blast Contact%": "blast_contact_pct",
    "EV": "avg_ev",
    "Exit Velocity": "avg_ev",
}

CACHE_FILE = "fangraphs_bat_tracking.json"


def fetch_bat_tracking(force_refresh: bool = False) -> dict[str, dict[str, Any]]:
    """
    Return bat tracking data keyed by normalized batter name|team.

    Uses a local cache that refreshes every BAT_TRACKING_TTL_DAYS days.

    Raises:
        DataSourceError: if pybaseball / FanGraphs is unreachable and no
            fresh cache is available.
    """
    cached = _read_cache() if not force_refresh else None
    if cached:
        return cached

    try:
        from pybaseball import batting_stats
    except ImportError as e:  # pragma: no cover - enforced by requirements
        raise DataSourceError(f"pybaseball not installed: {e}") from e

    try:
        season = date.today().year
        df = batting_stats(season, qual=50)
    except Exception as e:  # noqa: BLE001
        raise DataSourceError(f"FanGraphs pull via pybaseball failed: {e}") from e

    if df is None or len(df) == 0:
        raise DataSourceError("FanGraphs batting_stats returned empty.")

    # Rename known columns to canonical names; drop rows missing essentials.
    normalized: dict[str, dict[str, Any]] = {}
    rename: dict[str, str] = {src: dst for src, dst in COLUMN_MAP.items() if src in df.columns}
    df = df.rename(columns=rename)

    for row in df.to_dict(orient="records"):
        name = str(row.get("name") or row.get("Name") or "").strip()
        team = str(row.get("team") or row.get("Team") or "").strip()
        if not name or not team:
            continue
        key = f"{name}|{team}"
        normalized[key] = {
            "name": name,
            "team": team,
            "pull_pct": _num(row.get("pull_pct")),
            "bat_speed": _num(row.get("bat_speed")),
            "blast_contact_pct": _num(row.get("blast_contact_pct")),
            "avg_ev": _num(row.get("avg_ev")),
        }

    if not normalized:
        raise DataSourceError("FanGraphs result contained no usable rows.")

    _write_cache(normalized)
    return normalized


def _cache_file() -> Any:
    return settings.cache_path / CACHE_FILE


def _read_cache() -> dict[str, dict[str, Any]] | None:
    fpath = _cache_file()
    if not fpath.exists():
        return None
    age = datetime.now() - datetime.fromtimestamp(fpath.stat().st_mtime)
    if age > timedelta(days=settings.BAT_TRACKING_TTL_DAYS):
        return None
    try:
        with fpath.open() as f:
            return json.load(f)
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to read bat-tracking cache: %s", e)
        return None


def _write_cache(data: dict[str, dict[str, Any]]) -> None:
    fpath = _cache_file()
    try:
        with fpath.open("w") as f:
            json.dump(data, f)
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to write bat-tracking cache: %s", e)


def _num(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        f = float(v)
        # pybaseball sometimes reports Pull% etc as 0.38 instead of 38.
        return f
    except (TypeError, ValueError):
        return default
