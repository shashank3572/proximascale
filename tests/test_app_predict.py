"""POST /predict must call model.predict.predict_load and return its tuple as JSON."""
import sys
import types
from unittest.mock import MagicMock

import pytest

pytest.importorskip("flask")

from app.app import app  # noqa: E402


@pytest.fixture
def client(monkeypatch):
    # Don't touch the SQLite request counter from tests.
    monkeypatch.setattr("app.app.increment_request_count", lambda: None)
    return app.test_client()


def _fake_predict(monkeypatch, result=(61.5, 70.25, False)):
    mod = types.ModuleType("model.predict")
    mod.predict_load = MagicMock(return_value=result)
    monkeypatch.setitem(sys.modules, "model.predict", mod)
    return mod.predict_load


def test_predict_returns_json(client, monkeypatch):
    fn = _fake_predict(monkeypatch)
    records = [{"timestamp": "2026-09-20T10:00:00", "cpu_percent": 50.0,
                "memory_percent": 40.0, "request_rate": 100}] * 10
    resp = client.post("/predict", json={"records": records})
    assert resp.status_code == 200
    assert resp.get_json() == {"predicted_cpu": 61.5, "upper_bound": 70.25, "anomaly": False}
    fn.assert_called_once_with(records)


def test_predict_requires_ten_records(client, monkeypatch):
    _fake_predict(monkeypatch)
    assert client.post("/predict", json={"records": [{}] * 3}).status_code == 400
    assert client.post("/predict", json={}).status_code == 400


def test_predict_503_when_model_unavailable(client, monkeypatch):
    monkeypatch.setitem(sys.modules, "model.predict", None)  # forces ImportError
    assert client.post("/predict", json={"records": [{}] * 10}).status_code == 503
