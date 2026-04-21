"""Unit tests for the top-3 diversification picker."""

import pytest

from src.models.schemas import BatterScore
from src.pipeline import PipelineError
from src.pipeline.selection.picker import select_top_3


def _score(name: str, team: str, value: float, starter: bool = True) -> BatterScore:
    return BatterScore(
        batter_name=name,
        team=team,
        opponent_team="OPP",
        nuke_e_score=value,
        is_starter=starter,
    )


def test_picks_top_three_across_distinct_teams():
    scores = [
        _score("A", "NYY", 90),
        _score("B", "NYY", 85),  # second Yankee — should be skipped
        _score("C", "BOS", 80),
        _score("D", "LAD", 75),
        _score("E", "SFG", 50),
    ]
    picks = select_top_3(scores)
    names = [p.batter_name for p in picks]
    teams = [p.team for p in picks]
    assert names == ["A", "C", "D"]
    assert len(set(teams)) == 3


def test_excludes_non_starters():
    scores = [
        _score("Bench Masher", "NYY", 100, starter=False),
        _score("Regular A", "BOS", 60),
        _score("Regular B", "LAD", 55),
        _score("Regular C", "SFG", 50),
    ]
    picks = select_top_3(scores)
    assert "Bench Masher" not in [p.batter_name for p in picks]


def test_raises_when_fewer_than_three_distinct_teams():
    scores = [
        _score("A", "NYY", 80),
        _score("B", "NYY", 70),
        _score("C", "BOS", 60),
    ]
    with pytest.raises(PipelineError):
        select_top_3(scores)


def test_raises_on_empty_slate():
    with pytest.raises(PipelineError):
        select_top_3([])
