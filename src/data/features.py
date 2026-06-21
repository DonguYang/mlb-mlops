"""경기 결과 + 팀 통계로 ML 피처 생성"""
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"

FEATURE_COLS = [
    "home_win_rate_l10", "away_win_rate_l10",
    "home_era", "away_era",
    "home_ops", "away_ops",
]


def _add_rolling_win_rate(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """팀별 직전 window경기 승률 (리키지 방지: 경기 결과 반영 전 승률 기록)"""
    df = df.sort_values(["season", "date"]).copy()

    home_hist: dict = {}
    h_rates, a_rates = [], []

    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]

        hh = home_hist.setdefault(h, deque(maxlen=window))
        ah = home_hist.setdefault(a, deque(maxlen=window))

        h_rates.append(np.mean(hh) if len(hh) >= 3 else np.nan)
        a_rates.append(np.mean(ah) if len(ah) >= 3 else np.nan)

        hh.append(1 if row["home_win"] == 1 else 0)
        ah.append(0 if row["home_win"] == 1 else 1)

    df["home_win_rate_l10"] = h_rates
    df["away_win_rate_l10"] = a_rates
    return df


def build_features() -> pd.DataFrame:
    games = pd.read_parquet(RAW_DIR / "schedules.parquet")
    stats = pd.read_parquet(RAW_DIR / "team_stats.parquet")

    games["date"] = pd.to_datetime(games["date"])
    games = games.dropna(subset=["home_team", "away_team"])

    stats_idx = stats.set_index(["season", "team"])
    for side in ("home", "away"):
        games = games.join(
            stats_idx[["ops", "era"]].rename(columns={"ops": f"{side}_ops", "era": f"{side}_era"}),
            on=["season", f"{side}_team"],
        )

    games = _add_rolling_win_rate(games)

    feature_cols = ["season", "date", "home_team", "away_team"] + FEATURE_COLS + ["home_win"]
    result = games[feature_cols].dropna()
    result = result.rename(columns={"home_win": "label"})

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    result.to_parquet(PROCESSED_DIR / "features.parquet", index=False)
    print(f"피처 생성 완료: {len(result)}행 → {PROCESSED_DIR / 'features.parquet'}")
    print(f"홈팀 승률: {result['label'].mean():.3f}")
    return result


if __name__ == "__main__":
    build_features()
