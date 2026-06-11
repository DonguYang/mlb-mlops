"""경기 결과 + 팀 통계로 ML 피처 생성"""
import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path(__file__).parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"

# pybaseball schedule_and_record 팀명 약어 → 표준화
TEAM_ABBR = {
    "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS", "Chicago Cubs": "CHC", "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE", "Cleveland Indians": "CLE",
    "Colorado Rockies": "COL", "Detroit Tigers": "DET", "Houston Astros": "HOU",
    "Kansas City Royals": "KCR", "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN",
    "New York Mets": "NYM", "New York Yankees": "NYY", "Oakland Athletics": "OAK",
    "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT", "San Diego Padres": "SDP",
    "San Francisco Giants": "SFG", "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TBR", "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSN",
}


def _rolling_win_rate(schedule: pd.DataFrame, team_col: str, window: int = 10) -> pd.Series:
    """팀별 직전 window 경기 승률 계산 (리키지 방지: shift(1) 사용)"""
    results = []
    for _, grp in schedule.groupby(team_col):
        won = (grp["home_team_result"] == "W").astype(float)
        rolling = won.shift(1).rolling(window, min_periods=3).mean()
        results.append(rolling)
    return pd.concat(results).reindex(schedule.index)


def build_features() -> pd.DataFrame:
    """raw 데이터를 읽어 ML용 피처 테이블 생성"""
    schedules = pd.read_parquet(RAW_DIR / "schedules.parquet")
    batting = pd.read_parquet(RAW_DIR / "team_batting.parquet")
    pitching = pd.read_parquet(RAW_DIR / "team_pitching.parquet")

    # 완료된 경기만, 결측 제거
    df = schedules[schedules["home_team_result"].isin(["W", "L"])].copy()
    df = df.dropna(subset=["home_team", "away_team", "date"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["season", "date"]).reset_index(drop=True)

    # 홈팀 승리 레이블
    df["label"] = (df["home_team_result"] == "W").astype(int)

    # 시즌별 팀 OPS / ERA 조인
    batting_map = batting.set_index(["season", "Team"])[["OPS"]]
    pitching_map = pitching.set_index(["season", "Team"])[["ERA"]]

    df = df.join(batting_map.rename(columns={"OPS": "home_ops"}), on=["season", "home_team"])
    df = df.join(batting_map.rename(columns={"OPS": "away_ops"}), on=["season", "away_team"])
    df = df.join(pitching_map.rename(columns={"ERA": "home_era"}), on=["season", "home_team"])
    df = df.join(pitching_map.rename(columns={"ERA": "away_era"}), on=["season", "away_team"])

    # 최근 10경기 승률 (홈팀 기준)
    df["home_win_rate_l10"] = _rolling_win_rate(df, "home_team", 10)
    df["away_win_rate_l10"] = _rolling_win_rate(df, "away_team", 10)

    # 홈 어드밴티지 상수
    df["home_advantage"] = 1.0

    feature_cols = [
        "season", "date", "home_team", "away_team",
        "home_win_rate_l10", "away_win_rate_l10",
        "home_era", "away_era",
        "home_ops", "away_ops",
        "home_advantage",
        "label",
    ]

    result = df[feature_cols].dropna()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    result.to_parquet(PROCESSED_DIR / "features.parquet", index=False)
    print(f"피처 생성 완료: {len(result)}행 → {PROCESSED_DIR / 'features.parquet'}")
    return result


if __name__ == "__main__":
    build_features()
