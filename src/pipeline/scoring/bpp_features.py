"""BallparkPal feature extraction and tier classification.

Turns the raw BPP XLSX row into the per-component scores consumed by the
Nuke E composite, and classifies batters into tiers for downstream
rationale / reporting.
"""

from __future__ import annotations


def bpp_simulation_score(hr_prob: float) -> float:
    """BPP HR Prob (0-10%) scaled to 0-100ish."""
    return hr_prob * 10.0


def bpp_matchup_score(hr_prob_no_park: float) -> float:
    """Pure matchup quality, independent of park/weather."""
    return hr_prob_no_park * 10.0


def bpp_env_score(hr_boost: float) -> float:
    """Environment delta (% change). Cap negative impact at -20."""
    return max(hr_boost / 2.0, -20.0)


def bpp_grade_score(vs_grade: float) -> float:
    """BPP vs Grade (-10..+10) scaled to ~ -30..+30."""
    return vs_grade * 3.0


def contact_quality_score(k_prob: float) -> float:
    """Lower K% = more balls in play. 15% K => 7.5, 30% => 0."""
    return max(0.0, (30.0 - k_prob)) * 0.5


def tier_from_hr_prob(hr_prob: float) -> str:
    if hr_prob >= 4.0:
        return "A"
    if hr_prob >= 3.0:
        return "B"
    if hr_prob >= 2.0:
        return "C"
    return "D"
