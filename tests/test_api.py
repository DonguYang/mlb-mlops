"""FastAPI 엔드포인트 테스트 (모델 없이 /health, /model-info 확인)"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


def test_health():
    from src.serving.api import app
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_returns_prob(monkeypatch):
    from src.serving.api import app
    import src.serving.api as api_module

    mock_model = MagicMock()
    mock_model.predict.return_value = [0]
    mock_model._model_impl.predict_proba.return_value = [[0.42, 0.58]]

    monkeypatch.setattr(api_module, "_model", mock_model)

    client = TestClient(app)
    payload = {
        "home_team": "LAD", "away_team": "NYY",
        "home_win_rate_l10": 0.6, "away_win_rate_l10": 0.5,
        "home_era": 3.5, "away_era": 4.0,
        "home_ops": 0.750, "away_ops": 0.720,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert 0 <= data["home_win_prob"] <= 1
    assert data["prediction"] in ("LAD", "NYY")
