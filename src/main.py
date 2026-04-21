"""FastAPI entrypoint — single /picks endpoint + daily scheduler."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import date

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from src.cache.redis_client import (
    get_pipeline_status,
    picks_exist_for_today,
    ping as redis_ping,
    set_pipeline_status,
)
from src.config import settings
from src.pipeline import PipelineError
from src.pipeline.orchestrator import run_daily_pipeline

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_scheduler: BackgroundScheduler | None = None


def _scheduled_run() -> None:
    today = date.today().isoformat()
    try:
        run_daily_pipeline()
    except PipelineError as e:
        set_pipeline_status(today, "failed", error=str(e))
        logger.error("Scheduled pipeline failed: %s", e)
    except Exception as e:  # noqa: BLE001
        set_pipeline_status(today, "failed", error=str(e))
        logger.exception("Scheduled pipeline crashed")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _scheduler
    if settings.DISABLE_SCHEDULER:
        logger.info("Scheduler disabled via DISABLE_SCHEDULER.")
    else:
        _scheduler = BackgroundScheduler(timezone="UTC")
        _scheduler.add_job(
            _scheduled_run,
            "cron",
            hour=settings.SCHEDULE_HOUR_UTC,
            minute=settings.SCHEDULE_MINUTE_UTC,
            id="daily_pipeline",
            replace_existing=True,
        )
        _scheduler.start()
        logger.info(
            "Scheduler started: daily pipeline at %02d:%02d UTC",
            settings.SCHEDULE_HOUR_UTC,
            settings.SCHEDULE_MINUTE_UTC,
        )
    try:
        yield
    finally:
        if _scheduler is not None:
            _scheduler.shutdown(wait=False)


app = FastAPI(title="The Factory", version="0.1.0", lifespan=lifespan)


@app.get("/")
def root() -> dict:
    return {"service": "the-factory", "version": "0.1.0"}


@app.get("/health")
def health() -> dict:
    return {"ok": True, "redis": redis_ping()}


@app.get("/picks")
def get_picks() -> JSONResponse:
    """
    Return today's 3 HR picks.

    200: picks cached → { date, status: "cached", picks: [...] }
    200: pipeline failed → { date, status: "failed", error, picks: [] }
    200: pipeline not run yet → { date, status: "pending", picks: [] }
    """
    today = date.today().isoformat()

    cached = picks_exist_for_today(today)
    if cached:
        return JSONResponse(status_code=200, content=cached)

    status = get_pipeline_status(today)
    if status.get("status") == "failed":
        return JSONResponse(
            status_code=200,
            content={
                "date": today,
                "status": "failed",
                "error": status.get("error"),
                "picks": [],
            },
        )

    return JSONResponse(
        status_code=200,
        content={
            "date": today,
            "status": "pending",
            "picks": [],
            "message": "Pipeline has not run yet today.",
        },
    )


@app.post("/picks/run")
def trigger_run() -> JSONResponse:
    """Manually trigger the pipeline. Useful for ops / retries."""
    today = date.today().isoformat()
    try:
        result = run_daily_pipeline()
        return JSONResponse(status_code=200, content=result)
    except PipelineError as e:
        return JSONResponse(
            status_code=503,
            content={"date": today, "status": "failed", "error": str(e), "picks": []},
        )
