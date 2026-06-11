"""MLB 경기 결과 및 팀 통계 수집 (pybaseball)"""
import pandas as pd
from pathlib import Path
import pybaseball

RAW_DIR = Path(__file__).parents[2] / "data" / "raw"


def fetch_schedule_and_record(season: int) -> pd.DataFrame:
    """시즌 전체 경기 결과 수집. 홈팀 기준 승패 포함."""
    pybaseball.cache.enable()
    df = pybaseball.schedule_and_record(season, "ALL")
    df["season"] = season
    return df


def fetch_team_batting(season: int) -> pd.DataFrame:
    return pybaseball.team_batting(season)


def fetch_team_pitching(season: int) -> pd.DataFrame:
    return pybaseball.team_pitching(season)


def collect_seasons(start: int = 2020, end: int = 2024) -> None:
    """연도 범위의 경기/타격/투구 데이터를 raw/ 에 저장"""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    schedules = []
    batting_frames = []
    pitching_frames = []

    for season in range(start, end + 1):
        print(f"  수집 중: {season}시즌")
        schedules.append(fetch_schedule_and_record(season))
        batting_frames.append(fetch_team_batting(season).assign(season=season))
        pitching_frames.append(fetch_team_pitching(season).assign(season=season))

    pd.concat(schedules).to_parquet(RAW_DIR / "schedules.parquet", index=False)
    pd.concat(batting_frames).to_parquet(RAW_DIR / "team_batting.parquet", index=False)
    pd.concat(pitching_frames).to_parquet(RAW_DIR / "team_pitching.parquet", index=False)
    print(f"저장 완료: {RAW_DIR}")


if __name__ == "__main__":
    collect_seasons()
