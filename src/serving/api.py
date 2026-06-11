"""FastAPI 예측 서버"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
import os

app = FastAPI(title="MLB Win Predictor", version="1.0")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
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

    features = pd.DataFrame([{
        "home_win_rate_l10": req.home_win_rate_l10,
        "away_win_rate_l10": req.away_win_rate_l10,
        "home_era": req.home_era,
        "away_era": req.away_era,
        "home_ops": req.home_ops,
        "away_ops": req.away_ops,
        "home_advantage": 1.0,
    }])

    proba = model.predict(features)
    # pyfunc은 numpy array 반환, sklearn pipeline은 predict_proba 필요 시 unwrap
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
