"""Nuke E Score — the composite ranking metric.

Combines BPP simulation, matchup quality, environment, vs Grade,
bat-tracking physical ceiling, contact quality, pitcher vulnerability,
and an independent-signal convergence bonus. These weights are v1
estimates and will be tuned against graded picks.
"""

from __future__ import annotations

import logging
from typing import Any

from src.models.schemas import BatterScore
from src.pipeline.scoring.bat_tracking import hr_physical_ceiling
from src.pipeline.scoring.bpp_features import (
    bpp_env_score,
    bpp_grade_score,
    bpp_matchup_score,
    bpp_simulation_score,
    contact_quality_score,
)
from src.pipeline.scoring.pitcher_vuln import pitcher_vulnerability_score

logger = logging.getLogger(__name__)

# v1 weights — tune as graded outcomes accumulate.
WEIGHTS = {
    "bpp": 0.25,
    "matchup": 0.15,
    "env": 0.10,
    "grade": 0.10,
    "ceiling": 0.15,
    "contact": 0.05,
    "pitcher_vuln": 0.10,
    "validation": 0.10,
}


def compute_nuke_e_score(batter: dict[str, Any]) -> BatterScore:
    """Compute the Nuke E composite score for a single batter."""

    bpp_hr_prob = _num(batter.get("hr_prob"))
    bpp_hr_raw = _num(batter.get("hr_prob_no_park"))
    bpp_boost = _num(batter.get("hr_boost"))
    bpp_vs_grade = _num(batter.get("vs_grade"))
    bpp_k_prob = _num(batter.get("k_prob"), default=25.0)
    bpp_rc = _num(batter.get("rc"))

    bat_speed = batter.get("bat_speed")
    blast_pct = batter.get("blast_contact_pct")
    pull_pct = batter.get("pull_pct")
    avg_ev = batter.get("avg_ev")

    p_barrel = batter.get("pitcher_barrel_rate_allowed")
    p_fb_velo = batter.get("pitcher_fb_velo")
    p_hr_per_9 = batter.get("pitcher_hr_per_9")
    p_entropy = batter.get("pitcher_pitch_entropy")

    bpp_score = bpp_simulation_score(bpp_hr_prob)
    matchup_score = bpp_matchup_score(bpp_hr_raw)
    env_score = bpp_env_score(bpp_boost)
    grade_score = bpp_grade_score(bpp_vs_grade)

    ceiling = hr_physical_ceiling(bat_speed, blast_pct, pull_pct, avg_ev)
    ceiling_score = ceiling * 15.0

    contact_score = contact_quality_score(bpp_k_prob)

    pitcher_vuln_score = pitcher_vulnerability_score(
        fb_velo=p_fb_velo,
        barrel_rate_allowed=p_barrel,
        pitch_entropy=p_entropy,
    )

    validation = 0
    if bpp_hr_prob >= 3.5:
        validation += 1
    if bpp_hr_raw >= 3.5:
        validation += 1
    if bpp_vs_grade >= 5:
        validation += 1
    if bpp_boost >= 0:
        validation += 1
    if ceiling >= 1.2:
        validation += 1
    validation_bonus = validation * 5.0

    nuke_e = (
        bpp_score * WEIGHTS["bpp"]
        + matchup_score * WEIGHTS["matchup"]
        + env_score * WEIGHTS["env"]
        + grade_score * WEIGHTS["grade"]
        + ceiling_score * WEIGHTS["ceiling"]
        + contact_score * WEIGHTS["contact"]
        + pitcher_vuln_score * WEIGHTS["pitcher_vuln"]
        + validation_bonus * WEIGHTS["validation"]
    )

    return BatterScore(
        batter_name=batter.get("batter") or batter.get("name") or "",
        team=batter.get("team", ""),
        opponent_team=batter.get("opponent_team", ""),
        opponent_pitcher=batter.get("pitcher") or batter.get("opponent_pitcher"),
        batting_order=batter.get("batting_order"),
        is_starter=bool(batter.get("is_starter", True)),
        nuke_e_score=round(nuke_e, 2),
        bpp_hr_prob=bpp_hr_prob,
        bpp_hr_prob_no_park=bpp_hr_raw,
        bpp_hr_boost=bpp_boost,
        bpp_vs_grade=bpp_vs_grade,
        bpp_k_prob=bpp_k_prob,
        bpp_rc=bpp_rc,
        bat_speed=bat_speed,
        blast_contact_pct=blast_pct,
        pull_pct=pull_pct,
        avg_ev=avg_ev,
        pitcher_barrel_rate_allowed=p_barrel,
        pitcher_fb_velo=p_fb_velo,
        pitcher_hr_per_9=p_hr_per_9,
        pitcher_pitch_entropy=p_entropy,
        bpp_score=round(bpp_score, 2),
        matchup_score=round(matchup_score, 2),
        env_score=round(env_score, 2),
        grade_score=round(grade_score, 2),
        ceiling_score=round(ceiling_score, 2),
        contact_score=round(contact_score, 2),
        pitcher_vuln_score=round(pitcher_vuln_score, 2),
        validation_bonus=round(validation_bonus, 2),
    )


def compute_nuke_e_scores(
    lineups: dict[str, Any],
    bpp_data: dict[str, dict[str, Any]],
    bat_tracking: dict[str, dict[str, Any]],
    pitcher_profiles: dict[str, dict[str, Any]],
) -> list[BatterScore]:
    """
    Score every eligible batter on today's slate.

    Joins BPP rows (the spine) with lineup metadata, bat tracking, and
    pitcher profiles. Returns a list of BatterScore objects, one per row
    in the BPP export.
    """
    lineup_index = _index_lineups(lineups)
    scores: list[BatterScore] = []

    for _key, bpp_row in bpp_data.items():
        batter_name = bpp_row.get("batter", "")
        team = bpp_row.get("team", "")
        lineup_entry = lineup_index.get(f"{batter_name}|{team}", {})

        bat_key = f"{batter_name}|{team}"
        bt = bat_tracking.get(bat_key, {})

        opp_pitcher = bpp_row.get("pitcher") or lineup_entry.get("opponent_pitcher")
        pprof = pitcher_profiles.get(opp_pitcher, {}) if opp_pitcher else {}

        enriched = {
            **bpp_row,
            "opponent_team": lineup_entry.get("opponent_team", ""),
            "batting_order": lineup_entry.get("batting_order"),
            "bat_speed": bt.get("bat_speed"),
            "blast_contact_pct": bt.get("blast_contact_pct"),
            "pull_pct": bt.get("pull_pct"),
            "avg_ev": bt.get("avg_ev"),
            "pitcher_barrel_rate_allowed": pprof.get("barrel_rate_allowed"),
            "pitcher_fb_velo": pprof.get("fb_velo"),
            "pitcher_hr_per_9": pprof.get("hr_per_9"),
            "pitcher_pitch_entropy": pprof.get("pitch_entropy"),
        }
        scores.append(compute_nuke_e_score(enriched))

    return scores


def _index_lineups(lineups: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Flatten RotoWire games into a batter-keyed lookup."""
    index: dict[str, dict[str, Any]] = {}
    for game in lineups.get("games", []) or []:
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        sp_home = (game.get("starting_pitcher_home") or {}).get("name")
        sp_away = (game.get("starting_pitcher_away") or {}).get("name")
        for batter in game.get("lineup_home", []) or []:
            index[f"{batter.get('name')}|{home}"] = {
                "opponent_team": away,
                "opponent_pitcher": sp_away,
                "batting_order": batter.get("order"),
            }
        for batter in game.get("lineup_away", []) or []:
            index[f"{batter.get('name')}|{away}"] = {
                "opponent_team": home,
                "opponent_pitcher": sp_home,
                "batting_order": batter.get("order"),
            }
    return index


def _num(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except (TypeError, ValueError):
        return default
