"""MLB 경기 예측 대시보드"""
from collections import deque
from datetime import date, timedelta
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import requests
import streamlit as st

from src.data.collect import TEAM_IDS, MLB_API
from src.data.features import FEATURE_COLS

RAW_DIR = Path("data/raw")
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

# ── 팀 이름 전체명 매핑 ───────────────────────────────────────────
TEAM_NAMES = {
    "LAA": "LA Angels", "ARI": "Arizona", "BAL": "Baltimore", "BOS": "Boston",
    "CHC": "Chicago Cubs", "CIN": "Cincinnati", "CLE": "Cleveland", "COL": "Colorado",
    "DET": "Detroit", "HOU": "Houston", "KCR": "Kansas City", "LAD": "LA Dodgers",
    "WSN": "Washington", "NYM": "NY Mets", "OAK": "Oakland", "PIT": "Pittsburgh",
    "SDP": "San Diego", "SEA": "Seattle", "SFG": "San Francisco", "STL": "St. Louis",
    "TBR": "Tampa Bay", "TEX": "Texas", "TOR": "Toronto", "MIN": "Minnesota",
    "PHI": "Philadelphia", "ATL": "Atlanta", "CWS": "Chicago Sox", "MIA": "Miami",
    "NYY": "NY Yankees", "MIL": "Milwaukee",
}

# ── 데이터 로드 함수 ──────────────────────────────────────────────

@st.cache_resource
def load_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions("mlb_win_predictor", stages=["Production"])
    if not versions:
        return None, None
    v = versions[0]
    model = mlflow.sklearn.load_model(f"models:/mlb_win_predictor/Production")
    return model, v


@st.cache_data(ttl=3600)
def load_team_stats(season: int) -> pd.DataFrame:
    path = RAW_DIR / "team_stats.parquet"
    if not path.exists():
        return pd.DataFrame()
    stats = pd.read_parquet(path)
    return stats[stats["season"] == season].set_index("team")


@st.cache_data(ttl=3600)
def get_team_win_rates(window: int = 10) -> dict:
    """각 팀의 최근 N경기 승률 계산"""
    path = RAW_DIR / "schedules.parquet"
    if not path.exists():
        return {}
    games = pd.read_parquet(path)
    games["date"] = pd.to_datetime(games["date"])
    games = games[games["date"] < pd.Timestamp.today().normalize()]
    games = games.sort_values("date")

    team_hist: dict = {}
    for _, row in games.iterrows():
        h, a = row["home_team"], row["away_team"]
        team_hist.setdefault(h, deque(maxlen=window))
        team_hist.setdefault(a, deque(maxlen=window))
        team_hist[h].append(1 if row["home_win"] == 1 else 0)
        team_hist[a].append(0 if row["home_win"] == 1 else 1)

    return {
        team: round(float(np.mean(list(hist))), 3) if len(hist) >= 3 else None
        for team, hist in team_hist.items()
    }


@st.cache_data(ttl=600)
def get_games(target_date: date) -> list:
    """특정 날짜의 MLB 경기 목록 조회"""
    date_str = target_date.strftime("%Y-%m-%d")
    r = requests.get(
        f"{MLB_API}/schedule",
        params={
            "startDate": date_str,
            "endDate": date_str,
            "sportId": 1,
            "gameType": "R",
            "fields": "dates,date,games,gamePk,status,detailedState,teams,home,away,team,id,name,gameDate",
        },
        timeout=15,
    )
    r.raise_for_status()

    games = []
    for date_entry in r.json().get("dates", []):
        for g in date_entry.get("games", []):
            home_id = g["teams"]["home"]["team"]["id"]
            away_id = g["teams"]["away"]["team"]["id"]
            games.append({
                "game_id": g["gamePk"],
                "home_id": home_id,
                "away_id": away_id,
                "home_team": TEAM_IDS.get(home_id, str(home_id)),
                "away_team": TEAM_IDS.get(away_id, str(away_id)),
                "status": g["status"]["detailedState"],
            })
    return games


def predict_game(model, home_team: str, away_team: str,
                 stats: pd.DataFrame, win_rates: dict):
    """단일 경기 예측"""
    if home_team not in stats.index or away_team not in stats.index:
        return None

    home_stats = stats.loc[home_team]
    away_stats = stats.loc[away_team]

    features = pd.DataFrame([{
        "home_win_rate_l10": win_rates.get(home_team, 0.5),
        "away_win_rate_l10": win_rates.get(away_team, 0.5),
        "home_era": home_stats.get("era", 4.0),
        "away_era": away_stats.get("era", 4.0),
        "home_ops": home_stats.get("ops", 0.720),
        "away_ops": away_stats.get("ops", 0.720),
    }])[FEATURE_COLS]

    prob = float(model.predict_proba(features)[0][1])
    return {
        "home_prob": round(prob, 4),
        "away_prob": round(1 - prob, 4),
        "winner": home_team if prob >= 0.5 else away_team,
        "home_era": round(float(home_stats.get("era", 0)), 2),
        "away_era": round(float(away_stats.get("era", 0)), 2),
        "home_ops": round(float(home_stats.get("ops", 0)), 3),
        "away_ops": round(float(away_stats.get("ops", 0)), 3),
        "home_wr": win_rates.get(home_team),
        "away_wr": win_rates.get(away_team),
    }


# ── UI ───────────────────────────────────────────────────────────

st.set_page_config(page_title="MLB 경기 예측", page_icon="⚾", layout="wide")

# 사이드바
with st.sidebar:
    st.title("⚾ MLB Predictor")
    st.divider()

    target_date = st.date_input(
        "날짜 선택",
        value=date.today(),
        min_value=date(2021, 1, 1),
        max_value=date.today() + timedelta(days=1),
    )
    st.divider()

    # 모델 정보
    st.subheader("모델 정보")
    model, model_version = load_model()
    if model_version:
        st.metric("버전", f"v{model_version.version}")
        st.caption(f"Run ID: {model_version.run_id[:8]}...")

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        run = mlflow.get_run(model_version.run_id)
        acc = run.data.metrics.get("accuracy")
        auc = run.data.metrics.get("roc_auc")
        model_type = run.data.params.get("model_type", "-")
        if acc:
            st.metric("Accuracy", f"{acc:.1%}")
        if auc:
            st.metric("AUC", f"{auc:.4f}")
        st.caption(f"모델: {model_type}")
    else:
        st.warning("Production 모델 없음")
    st.divider()

    # 데이터 현황
    st.subheader("데이터 현황")
    sched_path = RAW_DIR / "schedules.parquet"
    if sched_path.exists():
        df_all = pd.read_parquet(sched_path)
        seasons = sorted(df_all["season"].unique())
        st.metric("수집 시즌", f"{seasons[0]} ~ {seasons[-1]}")
        st.metric("총 경기 수", f"{len(df_all):,}경기")
    else:
        st.warning("데이터 없음")

# 메인 영역
st.title(f"⚾ MLB 경기 예측 — {target_date.strftime('%Y년 %m월 %d일')}")
st.divider()

if model is None:
    st.error("Production 모델이 없습니다. 학습을 먼저 실행해주세요.")
    st.stop()

# 데이터 로드
current_season = target_date.year
team_stats = load_team_stats(current_season)
win_rates = get_team_win_rates()

if team_stats.empty:
    st.error("팀 통계 데이터가 없습니다. 데이터 수집을 먼저 실행해주세요.")
    st.stop()

# 경기 조회
with st.spinner("경기 일정 불러오는 중..."):
    try:
        games = get_games(target_date)
    except Exception as e:
        st.error(f"경기 조회 실패: {e}")
        st.stop()

if not games:
    st.info(f"{target_date} 에 예정된 정규시즌 경기가 없습니다.")
    st.stop()

# 예측 실행
predictions = []
for g in games:
    pred = predict_game(model, g["home_team"], g["away_team"], team_stats, win_rates)
    if pred:
        predictions.append({**g, **pred})

st.subheader(f"오늘의 경기 ({len(games)}경기, 예측 가능: {len(predictions)}경기)")

# 경기 카드 (2열)
cols = st.columns(2)
for i, p in enumerate(predictions):
    home = p["home_team"]
    away = p["away_team"]
    home_name = TEAM_NAMES.get(home, home)
    away_name = TEAM_NAMES.get(away, away)

    with cols[i % 2]:
        with st.container(border=True):
            # 경기 헤더
            h_col, vs_col, a_col = st.columns([4, 1, 4])
            with h_col:
                is_home_winner = p["winner"] == home
                st.markdown(f"### {'🏆 ' if is_home_winner else ''}{home}")
                st.caption(f"{home_name} (홈)")
            with vs_col:
                st.markdown("<br><br>**VS**", unsafe_allow_html=True)
            with a_col:
                st.markdown(f"### {'🏆 ' if not is_home_winner else ''}{away}")
                st.caption(f"{away_name} (원정)")

            st.divider()

            # 승리 확률 바
            home_pct = int(p["home_prob"] * 100)
            away_pct = int(p["away_prob"] * 100)

            prob_cols = st.columns([1, 8, 1])
            with prob_cols[0]:
                st.markdown(f"**{home_pct}%**")
            with prob_cols[1]:
                st.progress(p["home_prob"])
            with prob_cols[2]:
                st.markdown(f"**{away_pct}%**")

            st.divider()

            # 팀 스탯 비교
            stat_cols = st.columns(3)
            with stat_cols[0]:
                st.metric("ERA", p["home_era"],
                          delta=round(p["away_era"] - p["home_era"], 2),
                          delta_color="inverse",
                          help="낮을수록 좋음 (홈팀 기준)")
            with stat_cols[1]:
                st.metric("OPS", p["home_ops"],
                          delta=round(p["home_ops"] - p["away_ops"], 3),
                          help="높을수록 좋음 (홈팀 기준)")
            with stat_cols[2]:
                home_wr = f"{p['home_wr']:.1%}" if p["home_wr"] is not None else "N/A"
                away_wr = f"{p['away_wr']:.1%}" if p["away_wr"] is not None else "N/A"
                st.metric("최근 10경기 승률",
                          f"{home} {home_wr}",
                          delta=f"{away} {away_wr}",
                          delta_color="off",
                          help="홈팀 최근 10경기 승률")

            # 경기 상태
            status = p.get("status", "")
            if status and status != "Preview":
                st.caption(f"상태: {status}")

st.divider()

# 하단 요약 테이블
st.subheader("전체 예측 요약")
if predictions:
    summary_df = pd.DataFrame([{
        "홈팀": f"{p['home_team']} ({TEAM_NAMES.get(p['home_team'], '')})",
        "원정팀": f"{p['away_team']} ({TEAM_NAMES.get(p['away_team'], '')})",
        "홈팀 승리확률": f"{p['home_prob']:.1%}",
        "예측 승팀": p["winner"],
        "홈 ERA": p["home_era"],
        "원정 ERA": p["away_era"],
        "홈 OPS": p["home_ops"],
        "원정 OPS": p["away_ops"],
        "홈 최근승률": f"{p['home_wr']:.1%}" if p["home_wr"] else "N/A",
        "원정 최근승률": f"{p['away_wr']:.1%}" if p["away_wr"] else "N/A",
    } for p in predictions])
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
