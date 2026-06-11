"""경기 결과 + 팀 통계로 ML 피처 생성"""
import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path(__file__).parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"

FEATURE_COLS = [
    "home_win_rate_l10", "away_win_rate_l10",
    "home_era", "away_era",
    "home_ops", "away_ops",
    "home_advantage",
]


def _add_rolling_win_rate(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """팀별 직전 window경기 승률 (리키지 방지: shift(1))"""
    df = df.sort_values(["season", "date"]).copy()

    home_rates, away_rates = {}, {}
    result_home = []
    result_away = []

    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        result_home.append(home_rates.get(h, np.nan))
        result_away.append(away_rates.get(a, np.nan))

        # 경기 후 업데이트
        for team, won in [(h, row["home_win"] == 1), (a, row["home_win"] == 0)]:
            key = team
            if key not in home_rates:
                home_rates[key] = []
            if key not in away_rates:
                away_rates[key] = []
            # 공용 히스토리

    # 벡터화 방식으로 재계산
    home_hist = {}  # team -> deque of results
    from collections import deque

    h_rates, a_rates = [], []
    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]

        # 현재 기록 이전 승률
        hh = home_hist.setdefault(h, deque(maxlen=window))
        ah = home_hist.setdefault(a, deque(maxlen=window))

        h_rates.append(np.mean(hh) if len(hh) >= 3 else np.nan)
        a_rates.append(np.mean(ah) if len(ah) >= 3 else np.nan)

        # 경기 결과 추가
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

    # 시즌 통계 조인
    stats_idx = stats.set_index(["season", "team"])
    for side in ("home", "away"):
        games = games.join(
            stats_idx[["ops", "era"]].rename(columns={"ops": f"{side}_ops", "era": f"{side}_era"}),
            on=["season", f"{side}_team"],
        )

    games["home_advantage"] = 1.0
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
