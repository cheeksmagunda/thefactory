"""
MLB HR Model — FastAPI Backend
================================

Single backend API that runs the entire HR prediction pipeline.

Endpoints:
    GET  /api/health          — Pipeline status
    POST /api/pipeline/seed   — Full pipeline: pull Statcast data + train model + seed DB
    POST /api/picks           — Analyze a slate of matchups, return top 3 picks

Startup behavior:
    On deploy, the app loads cached data + trained model from disk (seconds).
    If no cached data exists, it starts unseeded — call POST /api/pipeline/seed
    to pull fresh Statcast data and train the model.

Usage:
    cd mlb_hr_model
    uvicorn api:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import logging
from contextlib import asynccontextmanager
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Ensure mlb_hr_model package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from schemas import (
    SlateRequest,
    PickResult, PairingResult, PredictionResponse,
    PipelineStatus, SeedResponse,
)
from daily_picks import (
    load_player_database,
    analyze_slate,
    generate_pairings,
)
from phase1_data_pipeline import run_phase1_pipeline, CONFIG
from phase2_model_training import (
    run_phase2_pipeline,
    load_model,
)

logger = logging.getLogger("mlb_hr_api")
logging.basicConfig(level=logging.INFO)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")


# ============================================================================
# PIPELINE STATE
# ============================================================================

class PipelineState:
    """Holds the seeded pipeline data for serving requests."""

    def __init__(self):
        self.seeded: bool = False
        self.batter_stats: dict = {}
        self.pitcher_stats: dict = {}
        self.name_map: dict = {}
        self.model = None
        self.calibrator = None
        self.model_config: dict = {}
        self.data_date_range: str = ""

    def reset(self):
        self.__init__()


pipeline = PipelineState()


# ============================================================================
# STARTUP: Fast load from cached files (seconds, not minutes)
# ============================================================================

def load_from_cache():
    """
    Load pre-existing cached data and trained model from disk.
    This is the FAST path — runs on every deploy/restart.
    If no cached data exists, the app starts unseeded.
    """
    logger.info("=" * 60)
    logger.info("STARTUP — Loading from cache")
    logger.info("=" * 60)

    pipeline.reset()
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Find cached Statcast parquet
    # ------------------------------------------------------------------
    parquet_files = sorted(
        [f for f in os.listdir(DATA_DIR) if f.endswith(".parquet")]
    ) if os.path.exists(DATA_DIR) else []

    if not parquet_files:
        logger.warning(
            "No cached Statcast data found in data/. "
            "Call POST /api/pipeline/seed to pull data and train model."
        )
        return

    parquet_path = os.path.join(DATA_DIR, parquet_files[-1])
    logger.info("[1/3] Loading player database from %s", parquet_path)

    result = load_player_database(parquet_path)
    if result is None or (isinstance(result, tuple) and result[0] is None):
        logger.error("load_player_database failed. Cached parquet may be corrupt.")
        return

    batter_stats, pitcher_stats, name_map = result
    pipeline.batter_stats = batter_stats
    pipeline.pitcher_stats = pitcher_stats
    pipeline.name_map = name_map

    logger.info(
        "  Player DB: %d batters, %d pitchers",
        len(batter_stats), len(pitcher_stats),
    )

    # ------------------------------------------------------------------
    # Step 2: Load trained model (if exists)
    # ------------------------------------------------------------------
    model_file = os.path.join(MODEL_DIR, "lgb_hr_model.txt")
    calibrator_file = os.path.join(MODEL_DIR, "calibrator.pkl")
    config_file = os.path.join(MODEL_DIR, "config.json")

    if all(os.path.exists(f) for f in [model_file, calibrator_file, config_file]):
        logger.info("[2/3] Loading trained model from models/")
        model, calibrator, model_config = load_model(MODEL_DIR)
        pipeline.model = model
        pipeline.calibrator = calibrator
        pipeline.model_config = model_config
        logger.info(
            "  Model loaded. Features: %d",
            len(model_config.get("feature_names", [])),
        )
    else:
        logger.warning(
            "No trained model found in models/. "
            "Call POST /api/pipeline/seed to train."
        )

    # ------------------------------------------------------------------
    # Step 3: Extract date range
    # ------------------------------------------------------------------
    logger.info("[3/3] Reading data date range...")
    try:
        df = pd.read_parquet(parquet_path, columns=["game_date"])
        df["game_date"] = pd.to_datetime(df["game_date"])
        date_min = df["game_date"].min().strftime("%Y-%m-%d")
        date_max = df["game_date"].max().strftime("%Y-%m-%d")
        pipeline.data_date_range = f"{date_min} to {date_max}"
        logger.info("  Data range: %s", pipeline.data_date_range)
    except Exception as e:
        logger.warning("  Could not extract date range: %s", e)

    pipeline.seeded = True
    logger.info("=" * 60)
    logger.info("STARTUP COMPLETE — Pipeline seeded from cache")
    logger.info(
        "  %d batters | %d pitchers | model: %s",
        len(pipeline.batter_stats),
        len(pipeline.pitcher_stats),
        "loaded" if pipeline.model is not None else "not trained yet",
    )
    logger.info("=" * 60)


# ============================================================================
# FULL SEED: Pull fresh data + train model (minutes)
# ============================================================================

def run_full_seed():
    """
    Full pipeline: pull Statcast data, run feature engineering,
    train LightGBM model, build player database.

    Called by POST /api/pipeline/seed.
    """
    logger.info("=" * 60)
    logger.info("FULL SEED — Pulling data + training model")
    logger.info("=" * 60)

    pipeline.reset()
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    start = CONFIG["train_start"]
    end = CONFIG["train_end"]
    cache_path = os.path.join(DATA_DIR, f"statcast_{start}_{end}.parquet")

    # ------------------------------------------------------------------
    # Phase 1: Data pipeline + feature engineering
    # ------------------------------------------------------------------
    logger.info("[1/3] Phase 1: Data Pipeline & Feature Engineering")
    logger.info("  Date range: %s to %s", start, end)
    phase1_result = run_phase1_pipeline(start, end, cache_path)

    # ------------------------------------------------------------------
    # Phase 2: Model training + calibration
    # ------------------------------------------------------------------
    logger.info("[2/3] Phase 2: Model Training & Calibration")
    phase2_result = run_phase2_pipeline(phase1_result)

    pipeline.model = phase2_result["model"]
    pipeline.calibrator = phase2_result["calibrator"]
    pipeline.model_config = {"feature_names": phase2_result["feature_names"]}

    # ------------------------------------------------------------------
    # Load player database from the freshly-cached parquet
    # ------------------------------------------------------------------
    logger.info("[3/3] Building player database from fresh data")

    parquet_files = sorted(
        [f for f in os.listdir(DATA_DIR) if f.endswith(".parquet")]
    )
    if not parquet_files:
        raise RuntimeError("No parquet file found after Phase 1. Data pull failed.")

    parquet_path = os.path.join(DATA_DIR, parquet_files[-1])
    result = load_player_database(parquet_path)
    if result is None or (isinstance(result, tuple) and result[0] is None):
        raise RuntimeError("load_player_database failed on fresh data.")

    batter_stats, pitcher_stats, name_map = result
    pipeline.batter_stats = batter_stats
    pipeline.pitcher_stats = pitcher_stats
    pipeline.name_map = name_map

    # Date range
    try:
        df = pd.read_parquet(parquet_path, columns=["game_date"])
        df["game_date"] = pd.to_datetime(df["game_date"])
        pipeline.data_date_range = (
            f"{df['game_date'].min().strftime('%Y-%m-%d')} to "
            f"{df['game_date'].max().strftime('%Y-%m-%d')}"
        )
    except Exception:
        pipeline.data_date_range = f"{start} to {end}"

    pipeline.seeded = True
    logger.info("=" * 60)
    logger.info("FULL SEED COMPLETE")
    logger.info(
        "  %d batters | %d pitchers | model trained",
        len(pipeline.batter_stats),
        len(pipeline.pitcher_stats),
    )
    logger.info("=" * 60)


# ============================================================================
# FASTAPI APP
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load from cache on startup (fast). Full seed is triggered via API."""
    try:
        load_from_cache()
    except Exception as e:
        logger.error("Startup cache load failed: %s", e, exc_info=True)
    yield


app = FastAPI(
    title="MLB HR Prediction API",
    description=(
        "MLB home run prediction pipeline. "
        "Submit a slate of batter/pitcher matchups and get the top 3 HR picks."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/api/health", response_model=PipelineStatus)
def health():
    """Pipeline status and health check."""
    return PipelineStatus(
        status="seeded" if pipeline.seeded else "unseeded",
        player_db_loaded=len(pipeline.batter_stats) > 0,
        model_loaded=pipeline.model is not None,
        batter_count=len(pipeline.batter_stats),
        pitcher_count=len(pipeline.pitcher_stats),
        data_date_range=pipeline.data_date_range or None,
    )


@app.post("/api/pipeline/seed", response_model=SeedResponse)
def seed():
    """
    Full pipeline re-run: pull Statcast data, train model, seed player DB.

    This clears all cached state and starts fresh. Takes several minutes
    for full data pull + training.
    """
    try:
        run_full_seed()
        return SeedResponse(
            status="success",
            message="Pipeline seeded successfully",
            batter_count=len(pipeline.batter_stats),
            pitcher_count=len(pipeline.pitcher_stats),
        )
    except Exception as e:
        logger.error("Pipeline seed failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline seed failed: {e}")


@app.post("/api/picks", response_model=PredictionResponse)
def get_picks(request: SlateRequest):
    """
    Analyze a slate of matchups and return the top 3 HR picks.

    Requires the pipeline to be seeded — either from cached data on startup,
    or via POST /api/pipeline/seed.
    """
    if not pipeline.seeded:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not seeded. Call POST /api/pipeline/seed first.",
        )

    # Convert Pydantic models to DataFrame
    slate_records = [m.model_dump() for m in request.slate]
    slate_df = pd.DataFrame(slate_records)

    # Run the analysis pipeline
    results_df = analyze_slate(
        slate_df,
        pipeline.batter_stats,
        pipeline.pitcher_stats,
        pipeline.name_map,
    )

    # Filter out matchups where provided batter/pitcher IDs weren't found in DB
    found_mask = pd.Series(True, index=results_df.index)
    for idx, row in results_df.iterrows():
        batter_id = row.get("batter_id")
        pitcher_id = row.get("pitcher_id")
        if batter_id is not None and batter_id not in pipeline.batter_stats:
            found_mask[idx] = False
        if pitcher_id is not None and pitcher_id not in pipeline.pitcher_stats:
            found_mask[idx] = False

    scored_df = results_df[found_mask].copy()

    if scored_df.empty:
        raise HTTPException(
            status_code=422,
            detail=(
                "No matchups could be scored. "
                "Batter/pitcher IDs not found in player database."
            ),
        )

    # Top 3 picks by edge
    top_picks = scored_df.head(3)

    # Generate pairings from all positive-edge picks
    pairings_df = generate_pairings(scored_df, top_n=5)

    # Build response
    picks_out = _build_picks_response(top_picks)
    pairings_out = _build_pairings_response(pairings_df)
    tier_counts = scored_df["tier"].value_counts().to_dict()

    return PredictionResponse(
        date=datetime.now().strftime("%Y-%m-%d"),
        total_matchups_analyzed=len(scored_df),
        picks=picks_out,
        pairings=pairings_out,
        slate_summary=tier_counts,
    )


# ============================================================================
# RESPONSE BUILDERS
# ============================================================================

def _build_picks_response(top_picks: pd.DataFrame) -> list[PickResult]:
    picks = []
    for rank, (_, row) in enumerate(top_picks.iterrows(), start=1):
        picks.append(
            PickResult(
                rank=rank,
                batter_name=row["batter_name"],
                team=row["team"],
                pitcher_name=row["pitcher_name"],
                opp_team=row["opp_team"],
                venue=str(row.get("venue", "")),
                odds=int(row["odds"]),
                power_score=round(float(row["power_score"]), 1),
                pitcher_vuln=round(float(row["pitcher_vuln"]), 1),
                env_score=round(float(row["env_score"]), 1),
                model_prob=round(float(row["model_prob"]), 4),
                model_prob_pct=f"{row['model_prob'] * 100:.1f}%",
                market_implied=round(float(row["market_implied"]), 4),
                edge_pct=round(float(row["edge_pct"]), 2),
                tier=row["tier"],
            )
        )
    return picks


def _build_pairings_response(pairings_df: pd.DataFrame) -> list[PairingResult]:
    if pairings_df.empty:
        return []
    pairings = []
    for _, prow in pairings_df.iterrows():
        pairings.append(
            PairingResult(
                player_a=str(prow["player_a"]),
                player_b=str(prow["player_b"]),
                odds_a=int(prow["odds_a"]),
                odds_b=int(prow["odds_b"]),
                parlay_odds=int(prow["parlay_odds"]),
                model_combined_prob_pct=round(float(prow["model_combined_prob"]), 2),
                book_combined_prob_pct=round(float(prow["book_combined_prob"]), 2),
                combined_edge_pct=round(float(prow["combined_edge_pct"]), 2),
                pairing_type=str(prow["type"]),
            )
        )
    return pairings
