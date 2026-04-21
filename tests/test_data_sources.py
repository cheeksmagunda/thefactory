"""Basic tests for data source stubs and BPP parsing."""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.pipeline import DataSourceError
from src.pipeline.data_sources import ballparkpal, odds, rotowire


def test_rotowire_stub_raises_data_source_error():
    with pytest.raises(DataSourceError):
        rotowire.fetch_lineups()


def test_odds_returns_empty_dict():
    assert odds.fetch_hr_odds() == {}


def test_ballparkpal_missing_file_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.pipeline.data_sources.ballparkpal.settings",
        type("S", (), {"bpp_data_path": tmp_path})(),
    )
    with pytest.raises(DataSourceError):
        ballparkpal.fetch_matchups(target_date=date(2026, 4, 21))


def test_ballparkpal_parses_valid_xlsx(tmp_path, monkeypatch):
    df = pd.DataFrame(
        [
            {
                "Batter": "Aaron Judge",
                "Team": "NYY",
                "Pitcher": "Chris Sale",
                "Throws": "L",
                "Starter": 1,
                "HR Prob": 4.1,
                "HR Prob (no park)": 3.8,
                "HR Boost": 8.0,
                "vs Grade": 6,
                "K Prob": 22.0,
                "RC": 0.15,
                "XB Prob": 1.1,
                "PA": 12,
                "HR": 3,
                "AVG": 0.333,
            }
        ]
    )
    target = date(2026, 4, 21)
    fpath = tmp_path / f"Matchups_{target.isoformat()}.xlsx"
    df.to_excel(fpath, index=False)

    monkeypatch.setattr(
        "src.pipeline.data_sources.ballparkpal.settings",
        type("S", (), {"bpp_data_path": tmp_path})(),
    )
    result = ballparkpal.fetch_matchups(target_date=target)
    key = "Aaron Judge|NYY"
    assert key in result
    row = result[key]
    assert row["hr_prob"] == 4.1
    assert row["is_starter"] is True


def test_ballparkpal_rejects_missing_columns(tmp_path, monkeypatch):
    df = pd.DataFrame([{"Batter": "X", "Team": "Y"}])
    target = date(2026, 4, 21)
    fpath = tmp_path / f"Matchups_{target.isoformat()}.xlsx"
    df.to_excel(fpath, index=False)

    monkeypatch.setattr(
        "src.pipeline.data_sources.ballparkpal.settings",
        type("S", (), {"bpp_data_path": tmp_path})(),
    )
    with pytest.raises(DataSourceError):
        ballparkpal.fetch_matchups(target_date=target)
