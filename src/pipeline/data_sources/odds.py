"""HR prop odds fetcher — STUB.

Planned sources: FanDuel and DraftKings HR prop lines. Edge calculation
(model prob vs implied market prob) will layer on top of Nuke E once this
is wired in.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def fetch_hr_odds() -> dict[str, Any]:
    """
    Return HR prop odds keyed by batter.

    For now this returns an empty dict so the pipeline can compute Nuke E
    without odds. Replace with real API calls when available.
    """
    logger.debug("Odds integration not implemented — returning empty dict.")
    return {}
