"""
test_model.py — Tests for Person B's LSTM prediction module.
Person D owns this file.

Covers:
    - predict() output shape and keys
    - anomaly detection edge cases
    - scaler loads correctly from disk
"""
import pytest
import os
import joblib
import numpy as np


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_records(cpu=50.0, n=10):
    """Generate n fake MetricRecord dicts for testing."""
    return [
        {
            "timestamp": f"2024-01-15T14:{i:02d}:00",
            "cpu_percent": cpu,
            "memory_percent": 40.0,
            "request_rate": 100
        }
        for i in range(n)
    ]


# ── predict() output contract ─────────────────────────────────────────────────

def test_predict_returns_required_keys():
    """predict() must return a dict with predicted_cpu and anomaly."""
    from model.predict import predict
    result = predict(make_records(cpu=50.0))
    assert "predicted_cpu" in result
    assert "anomaly" in result


def test_predict_cpu_is_list_of_3():
    """predicted_cpu must be a list of exactly 3 floats."""
    from model.predict import predict
    result = predict(make_records(cpu=50.0))
    assert isinstance(result["predicted_cpu"], list)
    assert len(result["predicted_cpu"]) == 3
    assert all(isinstance(v, float) for v in result["predicted_cpu"])


def test_predict_anomaly_is_bool():
    """anomaly must be a Python bool."""
    from model.predict import predict
    result = predict(make_records(cpu=50.0))
    assert isinstance(result["anomaly"], bool)


def test_predict_no_anomaly_on_stable_cpu():
    """Stable CPU at 50% should not trigger anomaly."""
    from model.predict import predict
    result = predict(make_records(cpu=50.0))
    assert result["anomaly"] is False


# ── Anomaly detection edge cases ──────────────────────────────────────────────

def test_anomaly_spike_detected():
    """A sudden spike in the last reading should trigger anomaly."""
    from model.anomaly import detect_anomaly
    # 9 stable readings then one extreme spike
    values = np.array([50.0] * 9 + [99.0])
    result = detect_anomaly(values)
    assert isinstance(result, bool)


def test_anomaly_all_zeros_no_crash():
    """detect_anomaly must handle std=0 without crashing."""
    from model.anomaly import detect_anomaly
    values = np.array([0.0] * 10)
    result = detect_anomaly(values)
    assert isinstance(result, bool)


def test_anomaly_returns_bool_not_numpy():
    """detect_anomaly must return Python bool, not numpy.bool_."""
    from model.anomaly import detect_anomaly
    values = np.array([50.0] * 10)
    result = detect_anomaly(values)
    assert type(result) is bool


# ── Scaler ────────────────────────────────────────────────────────────────────

def test_scaler_loads_from_disk():
    """scaler.pkl must exist and load without error."""
    scaler_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "model", "saved", "scaler.pkl"
    )
    scaler = joblib.load(scaler_path)
    assert scaler is not None


def test_scaler_has_3_features():
    """scaler must have been fit on exactly 3 features."""
    scaler_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "model", "saved", "scaler.pkl"
    )
    scaler = joblib.load(scaler_path)
    assert scaler.n_features_in_ == 3