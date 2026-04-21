"""Top-3 picker with diversification.

Rules:
  * Exactly 3 picks.
  * Max 1 batter per team.
  * Starters only.
  * Sorted by Nuke E score, descending.
"""

from __future__ import annotations

from src.models.schemas import BatterScore, Pick
from src.pipeline import PipelineError

MAX_PICKS = 3


def select_top_3(scored_batters: list[BatterScore]) -> list[Pick]:
    eligible = [b for b in scored_batters if b.is_starter]
    eligible.sort(key=lambda b: b.nuke_e_score, reverse=True)

    picks: list[Pick] = []
    teams_used: set[str] = set()

    for batter in eligible:
        if len(picks) >= MAX_PICKS:
            break
        if batter.team in teams_used:
            continue
        picks.append(
            Pick(
                batter_name=batter.batter_name,
                team=batter.team,
                opponent_team=batter.opponent_team,
                opponent_pitcher=batter.opponent_pitcher,
                nuke_e_score=batter.nuke_e_score,
                bpp_hr_prob=batter.bpp_hr_prob,
                validation=_validation_count(batter),
                rationale=_rationale(batter),
            )
        )
        teams_used.add(batter.team)

    if len(picks) < MAX_PICKS:
        raise PipelineError(
            f"Could not find {MAX_PICKS} eligible picks. Only found {len(picks)}."
        )
    return picks


def _validation_count(b: BatterScore) -> int:
    count = 0
    if b.bpp_hr_prob >= 3.5:
        count += 1
    if b.bpp_hr_prob_no_park >= 3.5:
        count += 1
    if b.bpp_vs_grade >= 5:
        count += 1
    if b.bpp_hr_boost >= 0:
        count += 1
    if b.ceiling_score >= 18:
        count += 1
    return count


def _rationale(b: BatterScore) -> str:
    parts = []
    if b.bpp_hr_prob >= 4.0:
        parts.append(f"BPP HR Prob {b.bpp_hr_prob:.1f}%")
    if b.bpp_vs_grade >= 5:
        parts.append(f"vs Grade +{b.bpp_vs_grade:.0f}")
    if b.ceiling_score >= 20:
        parts.append(f"elite bat tracking ({b.ceiling_score:.0f})")
    if b.pitcher_vuln_score >= 15:
        parts.append(f"vulnerable SP ({b.pitcher_vuln_score:.0f})")
    if b.bpp_hr_boost >= 10:
        parts.append(f"park/weather boost +{b.bpp_hr_boost:.0f}%")
    return "; ".join(parts) if parts else "composite edge"
