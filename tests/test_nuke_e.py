"""Unit tests for Nuke E Score components and composite."""

import pytest

from src.pipeline.scoring.bat_tracking import hr_physical_ceiling
from src.pipeline.scoring.bpp_features import (
    bpp_env_score,
    bpp_grade_score,
    bpp_matchup_score,
    bpp_simulation_score,
    contact_quality_score,
    tier_from_hr_prob,
)
from src.pipeline.scoring.nuke_e import compute_nuke_e_score
from src.pipeline.scoring.pitcher_vuln import pitcher_vulnerability_score


def test_bpp_simulation_score_scales():
    assert bpp_simulation_score(4.0) == 40.0
    assert bpp_simulation_score(0.0) == 0.0


def test_env_score_caps_negative():
    # Huge negative park factor (e.g. -100%) should cap at -20.
    assert bpp_env_score(-100.0) == -20.0
    assert bpp_env_score(20.0) == 10.0


def test_grade_score_maps():
    assert bpp_grade_score(10) == 30
    assert bpp_grade_score(-5) == -15


def test_contact_quality_score_zero_at_high_k():
    assert contact_quality_score(30.0) == 0.0
    assert contact_quality_score(15.0) == 7.5


def test_matchup_score_scales():
    assert bpp_matchup_score(3.5) == 35.0


def test_tier_from_hr_prob():
    assert tier_from_hr_prob(4.5) == "A"
    assert tier_from_hr_prob(3.1) == "B"
    assert tier_from_hr_prob(2.4) == "C"
    assert tier_from_hr_prob(1.0) == "D"


def test_hr_physical_ceiling_average_near_one():
    """An average batter (at baselines) scores ~1.0."""
    value = hr_physical_ceiling(bat_speed=75, blast_contact_pct=20, pull_pct=40, avg_ev=92)
    assert value == pytest.approx(1.0, rel=0.01)


def test_hr_physical_ceiling_handles_fractional_pct():
    # Pull% / Blast% may arrive as 0.40 / 0.20 from pybaseball.
    value = hr_physical_ceiling(
        bat_speed=75, blast_contact_pct=0.20, pull_pct=0.40, avg_ev=92
    )
    assert value == pytest.approx(1.0, rel=0.01)


def test_pitcher_vulnerability_nonzero_for_soft_tosser():
    score = pitcher_vulnerability_score(
        fb_velo=90.0, barrel_rate_allowed=0.10, pitch_entropy=1.2
    )
    # velo_penalty=10, barrel_signal=10, entropy_signal=8 -> 28
    assert score == pytest.approx(28.0)


def test_compute_nuke_e_score_returns_all_components():
    batter = {
        "batter": "Test Slugger",
        "team": "NYY",
        "pitcher": "Some Guy",
        "is_starter": True,
        "hr_prob": 4.1,
        "hr_prob_no_park": 3.8,
        "hr_boost": 8.0,
        "vs_grade": 6,
        "k_prob": 22,
        "rc": 0.15,
        "bat_speed": 76,
        "blast_contact_pct": 22,
        "pull_pct": 45,
        "avg_ev": 93,
        "pitcher_barrel_rate_allowed": 0.09,
        "pitcher_fb_velo": 91,
        "pitcher_hr_per_9": 1.4,
        "pitcher_pitch_entropy": 1.3,
    }
    result = compute_nuke_e_score(batter)
    assert result.batter_name == "Test Slugger"
    assert result.team == "NYY"
    assert result.nuke_e_score > 0
    assert result.bpp_score == pytest.approx(41.0)
    assert result.validation_bonus > 0  # several signals align
