"""Pydantic schemas for picks, pipeline status, and batter scoring."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class Weather(BaseModel):
    temperature_f: float | None = None
    wind_speed_mph: float | None = None
    wind_direction: str | None = None
    precipitation_pct: float | None = None
    is_dome: bool = False


class GameContext(BaseModel):
    game_id: str
    home_team: str
    away_team: str
    starting_pitcher_home: str | None = None
    starting_pitcher_away: str | None = None
    sp_home_hand: Literal["L", "R"] | None = None
    sp_away_hand: Literal["L", "R"] | None = None
    over_under: float | None = None
    weather: Weather | None = None


class Batter(BaseModel):
    name: str
    team: str
    batting_order: int | None = None
    position: str | None = None
    hand: Literal["L", "R", "S"] | None = None
    is_starter: bool = True
    opponent_pitcher: str | None = None


class BatterScore(BaseModel):
    """A scored batter with full component breakdown."""

    batter_name: str
    team: str
    opponent_team: str
    opponent_pitcher: str | None = None
    batting_order: int | None = None
    is_starter: bool = True

    # Composite
    nuke_e_score: float

    # Components (raw values fed into scoring)
    bpp_hr_prob: float = 0.0
    bpp_hr_prob_no_park: float = 0.0
    bpp_hr_boost: float = 0.0
    bpp_vs_grade: float = 0.0
    bpp_k_prob: float = 25.0
    bpp_rc: float = 0.0

    bat_speed: float | None = None
    blast_contact_pct: float | None = None
    pull_pct: float | None = None
    avg_ev: float | None = None

    pitcher_barrel_rate_allowed: float | None = None
    pitcher_fb_velo: float | None = None
    pitcher_hr_per_9: float | None = None
    pitcher_pitch_entropy: float | None = None

    # Scoring sub-components
    bpp_score: float = 0.0
    matchup_score: float = 0.0
    env_score: float = 0.0
    grade_score: float = 0.0
    ceiling_score: float = 0.0
    contact_score: float = 0.0
    pitcher_vuln_score: float = 0.0
    validation_bonus: float = 0.0


class Pick(BaseModel):
    """A single daily HR pick served by the API."""

    batter_name: str
    team: str
    opponent_team: str
    opponent_pitcher: str | None = None
    nuke_e_score: float
    bpp_hr_prob: float
    validation: int = Field(
        default=0,
        description="Number of independent signals that agree (0-5).",
    )
    rationale: str | None = None


PipelineStatusState = Literal["not_run", "running", "success", "failed"]


class PipelineStatus(BaseModel):
    date: str
    status: PipelineStatusState
    error: str | None = None
    updated_at: datetime | None = None


class PicksResponse(BaseModel):
    date: str
    status: Literal["cached", "fresh", "pending", "failed"]
    picks: list[Pick] = Field(default_factory=list)
    error: str | None = None
    message: str | None = None
