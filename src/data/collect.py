"""MLB 경기 결과 및 팀 통계 수집 (공식 MLB Stats API)"""
import pandas as pd
import requests
from pathlib import Path
import statsapi

RAW_DIR = Path(__file__).parents[2] / "data" / "raw"
MLB_API = "https://statsapi.mlb.com/api/v1"

TEAM_IDS = {
    108: "LAA", 109: "ARI", 110: "BAL", 111: "BOS", 112: "CHC",
    113: "CIN", 114: "CLE", 115: "COL", 116: "DET", 117: "HOU",
    118: "KCR", 119: "LAD", 120: "WSN", 121: "NYM", 133: "OAK",
    134: "PIT", 135: "SDP", 136: "SEA", 137: "SFG", 138: "STL",
    139: "TBR", 140: "TEX", 141: "TOR", 142: "MIN", 143: "PHI",
    144: "ATL", 145: "CWS", 146: "MIA", 147: "NYY", 158: "MIL",
}


def fetch_season_games(season: int) -> pd.DataFrame:
    print(f"  {season}시즌 경기 수집 중...")
    schedule = statsapi.schedule(
        start_date=f"{season}-03-01",
        end_date=f"{season}-10-31",
        sportId=1,
    )
    rows = []
    for g in schedule:
        if g.get("status") != "Final":
            continue
        rows.append({
            "game_id": g["game_id"],
            "date": g["game_date"],
            "season": season,
            "home_team": TEAM_IDS.get(g["home_id"], str(g["home_id"])),
            "away_team": TEAM_IDS.get(g["away_id"], str(g["away_id"])),
            "home_score": g["home_score"],
            "away_score": g["away_score"],
            "home_win": int(g["home_score"] > g["away_score"]),
        })
    df = pd.DataFrame(rows)
    print(f"    → {len(df)}경기")
    return df


def _get_team_stat(team_id: int, group: str, season: int) -> dict:
    url = f"{MLB_API}/teams/{team_id}/stats"
    params = {"stats": "season", "group": group, "season": season}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    splits = r.json().get("stats", [{}])[0].get("splits", [{}])
    return splits[0].get("stat", {}) if splits else {}


def fetch_team_season_stats(season: int) -> pd.DataFrame:
    print(f"  {season}시즌 팀 스탯 수집 중...")
    rows = []
    for team_id, abbr in TEAM_IDS.items():
        try:
            hit = _get_team_stat(team_id, "hitting", season)
            pit = _get_team_stat(team_id, "pitching", season)
            rows.append({
                "season": season,
                "team": abbr,
                "ops": float(hit.get("ops", 0) or 0),
                "avg": float(hit.get("avg", 0) or 0),
                "era": float(pit.get("era", 0) or 0),
                "whip": float(pit.get("whip", 0) or 0),
            })
        except Exception as e:
            print(f"    스킵 {abbr}: {e}")
    df = pd.DataFrame(rows)
    print(f"    → {len(df)}팀")
    return df


def collect_seasons(start: int = 2020, end: int = 2024) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    all_games, all_stats = [], []

    for season in range(start, end + 1):
        print(f"\n[{season}시즌]")
        all_games.append(fetch_season_games(season))
        all_stats.append(fetch_team_season_stats(season))

    games_df = pd.concat(all_games, ignore_index=True)
    stats_df = pd.concat(all_stats, ignore_index=True)

    games_df.to_parquet(RAW_DIR / "schedules.parquet", index=False)
    stats_df.to_parquet(RAW_DIR / "team_stats.parquet", index=False)
    print(f"\n저장 완료: {len(games_df)}경기, {len(stats_df)}팀×시즌 → {RAW_DIR}")


if __name__ == "__main__":
    collect_seasons()
