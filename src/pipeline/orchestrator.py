"""Daily pipeline orchestrator.

Runs all fetch → score → select → cache steps in order. Either the entire
run completes successfully (picks cached, endpoint serves them) or it fails
hard with a clear DataSourceError. No partial results. No fallbacks.
"""

from __future__ import annotations

import logging
from datetime import date

from src.cache.redis_client import (
    acquire_daily_lock,
    cache_picks,
    picks_exist_for_today,
    release_daily_lock,
    set_pipeline_status,
)
from src.pipeline import DataSourceError, PipelineError
from src.pipeline.data_sources import ballparkpal, fangraphs, rotowire, statcast
from src.pipeline.scoring.nuke_e import compute_nuke_e_scores
from src.pipeline.selection.picker import select_top_3

logger = logging.getLogger(__name__)


def run_daily_pipeline() -> dict:
    """
    Run the full pipeline for today.

    Returns:
        dict with keys: date, picks (list[Pick]), status.

    Raises:
        PipelineError: if any data source fails or selection can't produce 3 picks.
    """
    today = date.today().isoformat()

    existing = picks_exist_for_today(today)
    if existing:
        logger.info("Picks already cached for %s; returning cached.", today)
        return existing

    if not acquire_daily_lock(today):
        logger.info("Daily lock held for %s; skipping.", today)
        cached = picks_exist_for_today(today)
        if cached:
            return cached
        raise PipelineError(
            f"Daily lock is held for {today} but no picks are cached. "
            "A prior run failed — check pipeline status."
        )

    try:
        set_pipeline_status(today, "running")

        try:
            lineups = rotowire.fetch_lineups()
        except DataSourceError:
            raise
        except Exception as e:  # noqa: BLE001
            raise DataSourceError(f"RotoWire fetch failed: {e}") from e

        try:
            bpp_data = ballparkpal.fetch_matchups()
        except DataSourceError:
            raise
        except Exception as e:  # noqa: BLE001
            raise DataSourceError(f"BallparkPal fetch failed: {e}") from e

        try:
            bat_tracking = fangraphs.fetch_bat_tracking()
        except DataSourceError:
            raise
        except Exception as e:  # noqa: BLE001
            raise DataSourceError(f"FanGraphs bat tracking fetch failed: {e}") from e

        try:
            pitcher_profiles = statcast.fetch_pitcher_profiles()
        except DataSourceError:
            raise
        except Exception as e:  # noqa: BLE001
            raise DataSourceError(f"Statcast pitcher fetch failed: {e}") from e

        scores = compute_nuke_e_scores(
            lineups=lineups,
            bpp_data=bpp_data,
            bat_tracking=bat_tracking,
            pitcher_profiles=pitcher_profiles,
        )
        picks = select_top_3(scores)

        serialized = [p.model_dump() for p in picks]
        cache_picks(today, serialized)
        set_pipeline_status(today, "success")

        logger.info(
            "Pipeline complete for %s: %s",
            today,
            [p.batter_name for p in picks],
        )
        return {"date": today, "picks": serialized, "status": "fresh"}

    except PipelineError as e:
        set_pipeline_status(today, "failed", error=str(e))
        # Release lock on failure so a manual re-run is possible.
        release_daily_lock(today)
        logger.error("Pipeline FAILED for %s: %s", today, e)
        raise
    except Exception as e:  # noqa: BLE001
        set_pipeline_status(today, "failed", error=str(e))
        release_daily_lock(today)
        logger.exception("Pipeline crashed for %s", today)
        raise PipelineError(f"Unhandled pipeline error: {e}") from e
