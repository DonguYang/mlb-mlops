"""피처 엔지니어링 단위 테스트"""
import numpy as np
import pytest
from src.data.features import FEATURE_COLS

EXPECTED_FEATURE_COLS = [
    "home_win_rate_l10", "away_win_rate_l10",
    "home_era", "away_era",
    "home_ops", "away_ops",
    "home_advantage",
]


def test_feature_cols_complete():
    assert set(FEATURE_COLS) == set(EXPECTED_FEATURE_COLS)


def test_feature_cols_count():
    assert len(FEATURE_COLS) == 7


def test_home_advantage_in_features():
    assert "home_advantage" in FEATURE_COLS
