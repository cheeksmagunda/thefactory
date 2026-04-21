"""HR Physical Ceiling from FanGraphs bat tracking.

Multiplicative process score that captures a hitter's raw HR tooling:
bat speed, blast contact rate, pull rate, and exit velocity. Scaled so
an average major leaguer lands near 1.0 and elite hitters land near 1.5+.
"""

from __future__ import annotations


BASE_BAT_SPEED = 75.0
BASE_BLAST_PCT = 20.0
BASE_PULL_PCT = 40.0
BASE_AVG_EV = 92.0


def hr_physical_ceiling(
    bat_speed: float | None,
    blast_contact_pct: float | None,
    pull_pct: float | None,
    avg_ev: float | None,
) -> float:
    bs = bat_speed if bat_speed else 72.0
    blast = blast_contact_pct if blast_contact_pct else 18.0
    pull = pull_pct if pull_pct else 38.0
    ev = avg_ev if avg_ev else 88.0

    # FanGraphs may report Pull% / Blast% as 0.38 or 38 — normalize to pct.
    if blast <= 1.0:
        blast = blast * 100
    if pull <= 1.0:
        pull = pull * 100

    return (
        (bs / BASE_BAT_SPEED)
        * (blast / BASE_BLAST_PCT)
        * (pull / BASE_PULL_PCT)
        * (ev / BASE_AVG_EV)
    )
