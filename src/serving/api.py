"""FastAPI 예측 서버"""
import json
import pickle
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.data.features import FEATURE_COLS

app = FastAPI(title="MLB Win Predictor", version="1.0")

MODELS_DIR = Path(__file__).parents[2] / "models"
MODEL_PKL = MODELS_DIR / "production_model.pkl"
MODEL_META = MODELS_DIR / "production_model_meta.json"

_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PKL.exists():
            raise FileNotFoundError(f"모델 파일 없음: {MODEL_PKL}")
        with open(MODEL_PKL, "rb") as f:
            _model = pickle.load(f)
    return _model


class PredictRequest(BaseModel):
    home_team: str
    away_team: str
    home_win_rate_l10: float
    away_win_rate_l10: float
    home_era: float
    away_era: float
    home_ops: float
    away_ops: float


class PredictResponse(BaseModel):
    home_win_prob: float
    away_win_prob: float
    prediction: str
    model_name: str
    model_stage: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    try:
        if not MODEL_META.exists():
            return {"model": "mlb_win_predictor", "version": "none"}
        with open(MODEL_META) as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        model = get_model()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"모델 로드 실패: {e}")

    features = pd.DataFrame([{col: getattr(req, col) for col in FEATURE_COLS}])

    proba = model.predict(features)
    if hasattr(model._model_impl, "predict_proba"):
        prob = model._model_impl.predict_proba(features)[0][1]
    else:
        prob = float(proba[0])

    meta = json.loads(MODEL_META.read_text()) if MODEL_META.exists() else {}
    return PredictResponse(
        home_win_prob=round(prob, 4),
        away_win_prob=round(1 - prob, 4),
        prediction=req.home_team if prob >= 0.5 else req.away_team,
        model_name=meta.get("model_name", "mlb_win_predictor"),
        model_stage=meta.get("version", "unknown"),
    )
