"""Pitcher vulnerability from Statcast profile.

Combines fastball velocity penalty, barrel rate allowed, and pitch-type
entropy into a single score. Higher = more exploitable.
"""

from __future__ import annotations


def pitcher_vulnerability_score(
    fb_velo: float | None,
    barrel_rate_allowed: float | None,
    pitch_entropy: float | None,
) -> float:
    velo = fb_velo if fb_velo else 93.0
    barrel = barrel_rate_allowed if barrel_rate_allowed is not None else 0.06
    entropy = pitch_entropy if pitch_entropy is not None else 1.5

    velo_penalty = max(0.0, (95.0 - velo)) * 2.0  # slower velo = more vulnerable
    barrel_signal = barrel * 100.0                 # fractional -> 0..100
    entropy_signal = max(0.0, (2.0 - entropy)) * 10.0  # predictable = exploitable
    return velo_penalty + barrel_signal + entropy_signal
