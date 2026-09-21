"""Regression tests for the 2026-09-21 live-demo fixes.

Covers:
  1. Engine: upper_bound_risk must respect cooldown (the scale-up storm fix)
  2. predict_load: clamps forecasts to the physical 0-100 CPU% range
  3. Collector: two-sample CPU math + HTTP-first request reset channel
  4. App: POST /request-rate/reset contract (and that it never self-counts)
  5. main._compute_shap: wiring contract (never raises, reuses scaled window)
"""
import time
from unittest.mock import MagicMock, patch

import pytest

import os


# ── 1. Engine: cooldown gates upper_bound_risk ───────────────────────────────
@pytest.fixture
def engine():
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "config.yaml"
    )
    with patch("docker.from_env") as mock_docker:
        mock_client = MagicMock()
        mock_docker.return_value = mock_client
        mock_client.containers.list.return_value = []
        mock_client.containers.run.return_value = MagicMock()

        from decision.engine import DecisionEngine
        eng = DecisionEngine(config_path=config_path)
        eng.hysteresis.last_action_time = 0
        yield eng


def test_upper_bound_risk_respects_cooldown(engine):
    """Regression: during spike decay the upper bound stays elevated for many
    polls; the old ordering re-fired scale_up on every poll (the storm)."""
    r1 = engine.evaluate(predicted_cpu=90.0, upper_bound=95.0, anomaly_flag=False)
    assert r1 == "scale_up"
    assert engine.last_reason == "upper_bound_risk"

    # Next poll, still cooling down, bound still risky → must HOLD now
    r2 = engine.evaluate(predicted_cpu=70.0, upper_bound=90.0, anomaly_flag=False)
    assert r2 == "hold"
    assert engine.last_reason == "cooldown"


def test_upper_bound_risk_fires_again_after_cooldown_expires(engine):
    engine.evaluate(predicted_cpu=90.0, upper_bound=95.0, anomaly_flag=False)
    # Simulate cooldown elapsed
    engine.hysteresis.last_action_time = time.time() - 999

    r = engine.evaluate(predicted_cpu=70.0, upper_bound=90.0, anomaly_flag=False)
    assert r == "scale_up"
    assert engine.last_reason == "upper_bound_risk"


def test_anomaly_still_bypasses_cooldown(engine):
    engine.evaluate(predicted_cpu=90.0, upper_bound=95.0, anomaly_flag=False)
    r = engine.evaluate(predicted_cpu=10.0, upper_bound=None, anomaly_flag=True)
    assert r == "scale_up"
    assert engine.last_reason == "anomaly"


def test_cpu_high_still_gated_by_cooldown(engine):
    engine.evaluate(predicted_cpu=90.0, upper_bound=None, anomaly_flag=False)
    r = engine.evaluate(predicted_cpu=90.0, upper_bound=None, anomaly_flag=False)
    assert r == "hold"
    assert engine.last_reason == "cooldown"


# ── 2. predict_load clamps to 0-100 ──────────────────────────────────────────
pytest.importorskip("tensorflow", reason="TF stack not installed")
pytest.importorskip("prophet", reason="Prophet not installed")

import numpy as np  # noqa: E402

import model.predict as P  # noqa: E402


def _window(cpu=50.0, n=10):
    return [{"timestamp": f"2026-09-21T10:{i // 2:02d}:{(i % 2) * 30:02d}",
             "cpu_percent": cpu, "memory_percent": 40.0, "request_rate": 100,
             "post_scaling": False} for i in range(n)]


def _stub_artifacts(monkeypatch, prophet, uncertainty):
    monkeypatch.setattr(P, "_load_artifacts", lambda: None)
    monkeypatch.setattr(P, "_scaler", object(), raising=False)
    monkeypatch.setattr(P, "_model", object(), raising=False)
    monkeypatch.setattr(P, "_residual_scaler", object(), raising=False)
    monkeypatch.setattr(P, "_prophet_model", object(), raising=False)
    monkeypatch.setattr(P, "scale_features", lambda df, s: np.zeros((P.WINDOW_SIZE, 3)))
    monkeypatch.setattr(P, "get_prophet_fitted", lambda m, ts: prophet)
    monkeypatch.setattr(P, "predict_with_uncertainty", lambda *a, **k: uncertainty)


def test_forecast_above_100_is_clamped(monkeypatch):
    """Regression: live run produced upper bounds of 102-156 with the
    container pegged at 100 -- physically meaningless."""
    _stub_artifacts(monkeypatch,
                    prophet=[90.0, 100.0, 110.0],
                    uncertainty={"mean": [5.0, 5.0, 5.0], "upper_bound": [8.0, 8.0, 8.0]})
    predicted, upper, _ = P.predict_load(_window())
    assert predicted == 100.0   # raw 115.0
    assert upper == 100.0       # raw 118.0


def test_clamp_preserves_mean_below_bound(monkeypatch):
    _stub_artifacts(monkeypatch,
                    prophet=[60.0, 60.0, 60.0],
                    uncertainty={"mean": [30.0, 30.0, 30.0], "upper_bound": [40.0, 40.0, 40.0]})
    predicted, upper, _ = P.predict_load(_window())
    assert predicted == 90.0
    assert upper == 100.0
    assert predicted < upper


def test_fallback_is_clamped_too(monkeypatch):
    monkeypatch.setattr(P, "_load_artifacts",
                        lambda: (_ for _ in ()).throw(RuntimeError("no artifacts")))
    predicted, upper, anomaly = P.predict_load(_window(cpu=95.0))
    assert predicted == 100.0   # 95 * 1.15 = 109.25 raw
    assert upper == 100.0       # 95 * 1.3  = 123.5  raw
    assert anomaly is True


# ── 3. Collector: two-sample CPU + request reset channel ─────────────────────
import monitoring.collector as collector  # noqa: E402


def _stats(cpu_total, sys_total, online_cpus=2):
    return {
        "cpu_stats": {
            "cpu_usage": {"total_usage": cpu_total},
            "system_cpu_usage": sys_total,
            "online_cpus": online_cpus,
        },
        "memory_stats": {"usage": 100, "limit": 1000},
    }


def test_cpu_percent_between_two_snapshots():
    # 500k / 10M of one CPU-equivalent = 5% x 2 CPUs = 10%
    a = _stats(1_000_000, 100_000_000_000)
    b = _stats(1_500_000, 100_010_000_000)
    assert collector._cpu_percent_from_stats(a, b) == 10.0


def test_cpu_percent_clamped_at_100():
    a = _stats(0, 100_000_000_000)
    b = _stats(9_000_000, 100_010_000_000, online_cpus=12)
    assert collector._cpu_percent_from_stats(a, b) == 100.0


def test_cpu_percent_zero_when_system_delta_is_zero():
    a = _stats(1_000_000, 100_000_000_000)
    b = _stats(2_000_000, 100_000_000_000)  # system counter did not move
    assert collector._cpu_percent_from_stats(a, b) == 0.0


def test_collect_request_count_uses_http_reset(monkeypatch):
    class FakeResp:
        ok = True
        def json(self):
            return {"request_count": 7}

    seen = {}
    def fake_post(url, timeout=None):
        seen["url"] = url
        return FakeResp()

    monkeypatch.setattr("requests.post", fake_post)
    assert collector.collect_request_count() == 7
    assert seen["url"].endswith("/request-rate/reset")


def test_collect_request_count_falls_back_to_sqlite(monkeypatch):
    def unreachable(url, timeout=None):
        raise ConnectionError("app not reachable")
    monkeypatch.setattr("requests.post", unreachable)
    monkeypatch.setattr(collector, "reset_request_count", lambda: 3)
    assert collector.collect_request_count() == 3


def test_collect_request_count_returns_minus_one_when_both_fail(monkeypatch):
    def unreachable(url, timeout=None):
        raise ConnectionError("app not reachable")
    monkeypatch.setattr("requests.post", unreachable)
    def no_db():
        raise RuntimeError("no sqlite")
    monkeypatch.setattr(collector, "reset_request_count", no_db)
    assert collector.collect_request_count() == -1


# ── 4. App: /request-rate/reset contract ─────────────────────────────────────
pytest.importorskip("flask")

from app.app import app  # noqa: E402


@pytest.fixture
def flask_client(monkeypatch):
    monkeypatch.setattr("app.app.increment_request_count", lambda: None)
    return app.test_client()


def test_reset_endpoint_returns_window_count(flask_client, monkeypatch):
    monkeypatch.setattr("app.app.reset_request_count", lambda: 5)
    resp = flask_client.post("/request-rate/reset")
    assert resp.status_code == 200
    assert resp.get_json() == {"request_count": 5}


def test_reset_endpoint_is_excluded_from_self_counting(monkeypatch):
    inc = MagicMock()
    monkeypatch.setattr("app.app.increment_request_count", inc)
    monkeypatch.setattr("app.app.reset_request_count", lambda: 0)
    app.test_client().post("/request-rate/reset")
    inc.assert_not_called()


def test_readonly_request_rate_endpoint_still_get_only(flask_client):
    assert flask_client.post("/request-rate").status_code == 405
    assert flask_client.get("/request-rate").status_code == 200


# ── 5. main._compute_shap wiring ─────────────────────────────────────────────
from monitoring.schema import MetricRecord  # noqa: E402


def _records(n=10):
    return [MetricRecord(
        timestamp=f"2026-09-21T10:{i // 2:02d}:{(i % 2) * 30:02d}",
        cpu_percent=50.0, memory_percent=40.0, request_rate=100,
        post_scaling=False,
    ) for i in range(n)]


def test_compute_shap_uses_scaled_window_and_returns_attribution(monkeypatch):
    pytest.importorskip("tensorflow")
    import main

    captured = {}

    def fake_prepare(window):
        captured["window_len"] = len(window)
        return np.zeros((10, 3), dtype="float32"), np.zeros(10)

    import model.predict as P
    monkeypatch.setattr(P, "prepare_scaled_window", fake_prepare)
    monkeypatch.setattr(P, "get_shap_artifacts", lambda: ("fake-model", None))

    class FakeExplainer:
        def __init__(self, model, background, nsamples=256):
            captured["background_shape"] = np.asarray(background).shape
            captured["nsamples"] = nsamples
        def explain(self, x):
            captured["input_shape"] = np.asarray(x).shape
            return {"CPU%": 70.0, "Memory%": 20.0, "Request Rate": 10.0}

    import dashboard.shap_explain as se
    monkeypatch.setattr(se, "ShapExplainer", FakeExplainer)

    out = main._compute_shap(_records())
    assert out == {"CPU%": 70.0, "Memory%": 20.0, "Request Rate": 10.0}
    assert captured["window_len"] == 10
    assert captured["background_shape"] == (2, 10, 3)   # window + neutral baseline
    assert captured["input_shape"] == (1, 10, 3)
    assert captured["nsamples"] == 256


def test_compute_shap_never_raises(monkeypatch):
    pytest.importorskip("tensorflow")
    import main
    import model.predict as P

    def boom(window):
        raise RuntimeError("scaler exploded")
    monkeypatch.setattr(P, "prepare_scaled_window", boom)

    assert main._compute_shap(_records()) is None


# ── 6. events.csv migration (old 10-col logs gain 'reason' safely) ───────────
import csv  # noqa: E402


def test_log_event_migrates_old_header(tmp_path, monkeypatch):
    import main

    p = tmp_path / "events.csv"
    old_cols = main.EVENT_COLUMNS[:-1]          # header without 'reason'
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=old_cols)
        w.writeheader()
        w.writerow({c: 1.0 for c in old_cols})

    monkeypatch.setattr(main, "EVENTS_PATH", str(p))
    main.log_event({"signal": "scale_up", "reason": "cpu_high"})

    rows = list(csv.DictReader(open(p)))
    assert len(rows) == 2
    assert list(rows[0].keys()) == main.EVENT_COLUMNS   # uniform header now
    assert rows[0]["reason"] == "pre-reason log"
    assert rows[1]["reason"] == "cpu_high"
    assert rows[1]["signal"] == "scale_up"


def test_log_event_does_not_rewrite_current_format(tmp_path, monkeypatch):
    import main

    p = tmp_path / "events.csv"
    monkeypatch.setattr(main, "EVENTS_PATH", str(p))
    main.log_event({"signal": "hold"})
    first_size = p.stat().st_size
    main.log_event({"signal": "hold"})

    rows = list(csv.DictReader(open(p)))
    assert len(rows) == 2
    assert list(rows[0].keys()) == main.EVENT_COLUMNS
    # two appends only: header + 2 rows, no migration rewrite in between
    assert p.stat().st_size > first_size
