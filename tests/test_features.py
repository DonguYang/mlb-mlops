"""피처 엔지니어링 단위 테스트"""
import numpy as np
import pandas as pd
import pytest

from src.data.features import FEATURE_COLS, _add_rolling_win_rate


def test_feature_cols_complete():
    expected = {
        "home_win_rate_l10", "away_win_rate_l10",
        "home_era", "away_era",
        "home_ops", "away_ops",
    }
    assert set(FEATURE_COLS) == expected


def test_feature_cols_count():
    assert len(FEATURE_COLS) == 6


def _make_games(results: list) -> pd.DataFrame:
    """(home_win, home_team, away_team) 리스트로 샘플 DataFrame 생성"""
    rows = []
    for i, (home_win, home, away) in enumerate(results):
        rows.append({
            "season": 2023,
            "date": pd.Timestamp("2023-04-01") + pd.Timedelta(days=i),
            "home_team": home,
            "away_team": away,
            "home_win": home_win,
        })
    return pd.DataFrame(rows)


def test_rolling_win_rate_no_leakage():
    """경기 결과가 해당 경기 승률에 반영되지 않아야 함 (리키지 방지)"""
    games = _make_games([
        (1, "LAD", "NYY"),
        (1, "LAD", "NYY"),
        (1, "LAD", "NYY"),
        (1, "LAD", "NYY"),  # LAD 4연승 후
        (0, "LAD", "NYY"),  # 이 경기 시점 LAD 승률은 1.0이어야 함
    ])
    result = _add_rolling_win_rate(games, window=10)

    # 5번째 경기(인덱스 4) 시점에서 LAD는 이전 4경기 모두 승 → 1.0
    fifth_game = result.iloc[4]
    assert fifth_game["home_win_rate_l10"] == 1.0

    # 5번째 경기 결과(패)가 해당 행 승률에 반영되면 안 됨
    assert fifth_game["home_win_rate_l10"] != pytest.approx(0.8, abs=0.01)


def test_rolling_win_rate_nan_before_min_games():
    """최소 3경기 미만이면 NaN을 반환해야 함"""
    games = _make_games([
        (1, "LAD", "NYY"),
        (1, "LAD", "NYY"),
        (1, "LAD", "NYY"),
    ])
    result = _add_rolling_win_rate(games, window=10)

    # 1~2번째 경기는 히스토리 부족으로 NaN
    assert np.isnan(result.iloc[0]["home_win_rate_l10"])
    assert np.isnan(result.iloc[1]["home_win_rate_l10"])
    # 3번째 경기는 이전 2경기 승 → NaN (3경기 미만)
    assert np.isnan(result.iloc[2]["home_win_rate_l10"])


def test_rolling_win_rate_window_limit():
    """window=3일 때 최근 3경기만 반영되어야 함"""
    # LAD: 패패패승승승승 → 마지막 3경기 기준 승률 1.0
    games = _make_games([
        (0, "LAD", "NYY"),
        (0, "LAD", "NYY"),
        (0, "LAD", "NYY"),
        (1, "LAD", "NYY"),
        (1, "LAD", "NYY"),
        (1, "LAD", "NYY"),
        (1, "LAD", "NYY"),  # 이 시점 window=3: 직전 3경기 모두 승 → 1.0
    ])
    result = _add_rolling_win_rate(games, window=3)
    assert result.iloc[6]["home_win_rate_l10"] == pytest.approx(1.0)
