"""Production 모델을 models/ 디렉토리에 파일로 export"""
import json
import pickle
from datetime import date
from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.tracking import MlflowClient

MODELS_DIR = Path(__file__).parents[2] / "models"
MODEL_NAME = "mlb_win_predictor"


def export_production_model(tracking_uri: str = "sqlite:///mlflow.db") -> Path:
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    if not versions:
        raise RuntimeError("Production 스테이지 모델이 없습니다.")

    v = versions[0]
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/Production")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "production_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    meta = {
        "model_name": MODEL_NAME,
        "version": v.version,
        "run_id": v.run_id,
        "exported_at": date.today().isoformat(),
    }
    meta_path = MODELS_DIR / "production_model_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"export 완료: {model_path}  (version={v.version}, run={v.run_id[:8]})")
    return model_path


if __name__ == "__main__":
    export_production_model()
