"""
MLB HR Model — API Schemas
============================

Pydantic models for request/response validation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import date


# ============================================================================
# REQUEST MODELS
# ============================================================================

class MatchupInput(BaseModel):
    """A single batter vs pitcher matchup for scoring."""
    batter_name: str = Field(..., description="Batter full name")
    batter_id: Optional[int] = Field(None, description="Statcast player ID for batter")
    team: str = Field(..., description="Batter's team abbreviation (e.g. PHI)")
    pitcher_name: str = Field(..., description="Pitcher full name")
    pitcher_id: Optional[int] = Field(None, description="Statcast player ID for pitcher")
    opp_team: str = Field(..., description="Opposing team abbreviation")
    odds: int = Field(..., description="American odds for HR prop (e.g. +400 = 400)")
    temperature: Optional[float] = Field(None, description="Game-day temperature (F)")
    wind_speed: Optional[float] = Field(None, description="Wind speed (mph)")
    wind_dir: Optional[str] = Field(None, description="Wind direction: 'out', 'in', 'L-R', 'R-L', 'dome'")
    stand: Optional[str] = Field(None, description="Batter handedness: 'L' or 'R'")
    p_throws: Optional[str] = Field(None, description="Pitcher handedness: 'L' or 'R'")
    venue: Optional[str] = Field(None, description="Ballpark team code (e.g. COL, NYY)")


class SlateRequest(BaseModel):
    """Request body for the picks endpoint."""
    slate: List[MatchupInput] = Field(..., min_length=1, description="List of matchups to analyze")


# ============================================================================
# RESPONSE MODELS
# ============================================================================

class PickResult(BaseModel):
    """A single HR pick recommendation."""
    rank: int
    batter_name: str
    team: str
    pitcher_name: str
    opp_team: str
    venue: str
    odds: int
    power_score: float
    pitcher_vuln: float
    env_score: float
    model_prob: float = Field(..., description="Game-level HR probability (decimal)")
    model_prob_pct: str = Field(..., description="Formatted probability (e.g. '14.2%')")
    market_implied: float = Field(..., description="Market implied probability (vig-removed)")
    edge_pct: float = Field(..., description="Edge vs market (model - market, in %)")
    tier: str = Field(..., description="Classification: TRIPLE CONVERGENCE, VALUE + POWER, POWER PLAY, MONITOR, SKIP")


class PairingResult(BaseModel):
    """A recommended 2-leg HR parlay pairing."""
    player_a: str
    player_b: str
    odds_a: int
    odds_b: int
    parlay_odds: int
    model_combined_prob_pct: float
    book_combined_prob_pct: float
    combined_edge_pct: float
    pairing_type: str = Field(..., description="Same-team stack, Same-game, or Cross-game")


class PredictionResponse(BaseModel):
    """Response from the picks endpoint."""
    date: str
    total_matchups_analyzed: int
    picks: List[PickResult]
    pairings: List[PairingResult]
    slate_summary: Dict[str, int] = Field(..., description="Count of picks per tier")


class PipelineStatus(BaseModel):
    """Response from the health/status endpoint."""
    status: str = Field(..., description="'seeded' or 'unseeded'")
    player_db_loaded: bool
    model_loaded: bool
    batter_count: int
    pitcher_count: int
    data_date_range: Optional[str] = None


class SeedResponse(BaseModel):
    """Response from the pipeline seed endpoint."""
    status: str
    message: str
    batter_count: int
    pitcher_count: int
