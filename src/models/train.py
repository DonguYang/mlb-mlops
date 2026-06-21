"""MLflow 연동 모델 학습 (Logistic Regression → XGBoost)"""
import argparse
from datetime import date
from pathlib import Path

import mlflow
import mlflow.sklearn
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.features import FEATURE_COLS

PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"


def load_data(holdout_season: int = None):
    if holdout_season is None:
        holdout_season = date.today().year - 1

    import pandas as pd
    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    train = df[df["season"] < holdout_season]
    test = df[df["season"] == holdout_season]
    X_train = train[FEATURE_COLS]
    y_train = train["label"]
    X_test = test[FEATURE_COLS]
    y_test = test["label"]
    return X_train, y_train, X_test, y_test


def train_logistic(X_train, y_train, X_test, y_test, C: float = 1.0):
    with mlflow.start_run(run_name="logistic_regression"):
        mlflow.log_param("model_type", "logistic_regression")
        mlflow.log_param("C", C)

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=C, max_iter=1000, random_state=42)),
        ])
        pipe.fit(X_train, y_train)

        acc = accuracy_score(y_test, pipe.predict(X_test))
        auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", auc)

        mlflow.sklearn.log_model(pipe, "model", registered_model_name="mlb_win_predictor")
        print(f"[LR] accuracy={acc:.4f}  roc_auc={auc:.4f}")
        return acc


def train_xgboost(X_train, y_train, X_test, y_test,
                  n_estimators: int = 200, max_depth: int = 4, lr: float = 0.05):
    with mlflow.start_run(run_name="xgboost"):
        mlflow.log_params({"model_type": "xgboost", "n_estimators": n_estimators,
                           "max_depth": max_depth, "learning_rate": lr})

        model = xgb.XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=lr, use_label_encoder=False,
            eval_metric="logloss", random_state=42,
        )
        model.fit(X_train, y_train)

        acc = accuracy_score(y_test, model.predict(X_test))
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", auc)

        mlflow.sklearn.log_model(model, "model", registered_model_name="mlb_win_predictor")
        print(f"[XGB] accuracy={acc:.4f}  roc_auc={auc:.4f}")
        return acc


def promote_best_model():
    """가장 높은 accuracy 실험을 Production으로 승격"""
    client = mlflow.tracking.MlflowClient()
    runs = mlflow.search_runs(order_by=["metrics.accuracy DESC"], max_results=1)
    if runs.empty:
        return
    best_run_id = runs.iloc[0]["run_id"]
    versions = client.search_model_versions(f"name='mlb_win_predictor'")
    for v in versions:
        if v.run_id == best_run_id:
            client.transition_model_version_stage(
                name="mlb_win_predictor", version=v.version, stage="Production"
            )
            print(f"Production 승격: version={v.version}  run={best_run_id[:8]}")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["logistic", "xgboost", "all"], default="all")
    parser.add_argument("--holdout-season", type=int, default=None)
    args = parser.parse_args()

    mlflow.set_experiment("mlb_win_prediction")
    X_train, y_train, X_test, y_test = load_data(args.holdout_season)
    print(f"학습: {len(X_train)}행  테스트: {len(X_test)}행")

    if args.model in ("logistic", "all"):
        train_logistic(X_train, y_train, X_test, y_test)
    if args.model in ("xgboost", "all"):
        train_xgboost(X_train, y_train, X_test, y_test)

    promote_best_model()
