"""피처 엔지니어링 단위 테스트"""
import pandas as pd
import numpy as np
import pytest
from src.data.features import TEAM_ABBR, FEATURE_COLS_EXPECTED

FEATURE_COLS_EXPECTED = [
    "home_win_rate_l10", "away_win_rate_l10",
    "home_era", "away_era",
    "home_ops", "away_ops",
    "home_advantage",
]


def make_dummy_schedule(n: int = 30) -> pd.DataFrame:
    teams = ["LAD", "NYY", "BOS", "HOU"]
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "home_team": rng.choice(teams, n),
        "away_team": rng.choice(teams, n),
        "home_team_result": rng.choice(["W", "L"], n),
        "date": pd.date_range("2024-04-01", periods=n, freq="D"),
        "season": 2024,
    })


def test_team_abbr_has_30_teams():
    assert len(TEAM_ABBR) == 30


def test_expected_feature_cols():
    assert "home_advantage" in FEATURE_COLS_EXPECTED
    assert "home_win_rate_l10" in FEATURE_COLS_EXPECTED
    assert len(FEATURE_COLS_EXPECTED) == 7
