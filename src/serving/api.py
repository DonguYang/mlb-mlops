"""FastAPI 예측 서버"""
import os

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.data.features import FEATURE_COLS

app = FastAPI(title="MLB Win Predictor", version="1.0")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_NAME = "mlb_win_predictor"
MODEL_STAGE = "Production"

_model = None


def get_model():
    global _model
    if _model is None:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        _model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")
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
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
        if not versions:
            return {"model": MODEL_NAME, "stage": MODEL_STAGE, "version": "none"}
        v = versions[0]
        return {"model": MODEL_NAME, "stage": MODEL_STAGE,
                "version": v.version, "run_id": v.run_id}
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

    return PredictResponse(
        home_win_prob=round(prob, 4),
        away_win_prob=round(1 - prob, 4),
        prediction=req.home_team if prob >= 0.5 else req.away_team,
        model_name=MODEL_NAME,
        model_stage=MODEL_STAGE,
    )
