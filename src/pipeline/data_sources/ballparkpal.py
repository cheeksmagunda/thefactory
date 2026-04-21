"""BallparkPal matchup ingestion.

Primary path: scrape/download the premium daily XLSX export.
Manual fallback: read from ``settings.BALLPARKPAL_DATA_DIR/Matchups_YYYY-MM-DD.xlsx``.

The 38-column XLSX export contains, per batter, the full matchup simulation:
HR Prob, HR Prob (no park), HR Boost, vs Grade, K Prob, RC, XB Prob,
PA/HR/AVG history, and more.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any

from src.config import settings
from src.pipeline import DataSourceError

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    "Batter",
    "Team",
    "Pitcher",
    "Throws",
    "Starter",
    "HR Prob",
    "HR Prob (no park)",
    "HR Boost",
    "vs Grade",
    "K Prob",
    "RC",
}


def fetch_matchups(target_date: date | None = None) -> dict[str, Any]:
    """
    Return today's BallparkPal matchup data keyed by batter name.

    Shape:
        {
          "Aaron Judge|NYY": {
            "batter": "Aaron Judge", "team": "NYY", "pitcher": "...",
            "throws": "R", "is_starter": True,
            "hr_prob": 4.1, "hr_prob_no_park": 3.8, "hr_boost": 8.0,
            "vs_grade": 6, "k_prob": 24.5, "rc": 0.18, "xb_prob": ...,
            "pa_history": ..., "hr_history": ..., "avg_history": ...,
          },
          ...
        }

    Resolution order:
        1. Look for a manual XLSX upload at BALLPARKPAL_DATA_DIR/Matchups_YYYY-MM-DD.xlsx.
        2. (Future) Authenticated scrape using BALLPARKPAL_EMAIL / BALLPARKPAL_PASSWORD.

    Raises:
        DataSourceError: on missing file, empty export, or missing required columns.
    """
    target = target_date or date.today()
    fname = f"Matchups_{target.isoformat()}.xlsx"
    fpath = settings.bpp_data_path / fname

    if not fpath.exists():
        raise DataSourceError(
            "BallparkPal data missing for "
            f"{target.isoformat()}. Drop the daily export at {fpath} "
            "or implement the automated scraper."
        )

    try:
        return _parse_xlsx(fpath)
    except DataSourceError:
        raise
    except Exception as e:  # noqa: BLE001
        raise DataSourceError(f"Failed to parse {fpath}: {e}") from e


def _parse_xlsx(fpath: Path) -> dict[str, Any]:
    """Parse the BPP XLSX export into a batter-keyed dict."""
    import pandas as pd

    df = pd.read_excel(fpath)
    if df.empty:
        raise DataSourceError(f"BallparkPal export at {fpath} is empty.")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise DataSourceError(
            f"BallparkPal export at {fpath} missing required columns: "
            f"{sorted(missing)}"
        )

    out: dict[str, Any] = {}
    for row in df.to_dict(orient="records"):
        batter = str(row.get("Batter", "")).strip()
        team = str(row.get("Team", "")).strip()
        if not batter or not team:
            continue
        key = f"{batter}|{team}"
        out[key] = {
            "batter": batter,
            "team": team,
            "pitcher": row.get("Pitcher"),
            "throws": row.get("Throws"),
            "is_starter": bool(row.get("Starter", 0)),
            "hr_prob": _num(row.get("HR Prob")),
            "hr_prob_no_park": _num(row.get("HR Prob (no park)")),
            "hr_boost": _num(row.get("HR Boost")),
            "vs_grade": _num(row.get("vs Grade")),
            "k_prob": _num(row.get("K Prob"), default=25.0),
            "rc": _num(row.get("RC")),
            "xb_prob": _num(row.get("XB Prob")),
            "pa_history": _num(row.get("PA")),
            "hr_history": _num(row.get("HR")),
            "avg_history": _num(row.get("AVG")),
        }
    if not out:
        raise DataSourceError(f"BallparkPal export at {fpath} yielded zero rows.")
    return out


def _num(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except (TypeError, ValueError):
        return default
