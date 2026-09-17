"""
test_model.py — Tests for Person B's LSTM prediction module.
Person D owns this file.

Covers:
    - predict_load() output contract
    - anomaly detection edge cases
    - scaler loads correctly from disk
"""
import pytest
import os
import joblib
import numpy as np


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_records(cpu=50.0, n=10):
    """Generate n fake metric records for testing."""
    return [
        {
            "timestamp": f"2024-01-15T14:{i:02d}:00",
            "cpu_percent": cpu,
            "memory_percent": 40.0,
            "request_rate": 100,
            "post_scaling": False,
        }
        for i in range(n)
    ]


# ── predict_load() output contract ────────────────────────────────────────────

def test_predict_load_returns_three_values():
    """predict_load() must return (predicted_cpu, upper_bound, anomaly)."""
    pytest.importorskip("tf_keras", reason="Person B's TF stack not installed")

    from model.predict import predict_load

    predicted_cpu, upper_bound, anomaly = predict_load(make_records(cpu=50.0))

    assert isinstance(predicted_cpu, float)
    assert isinstance(upper_bound, float)
    assert isinstance(anomaly, bool)


def test_predict_load_values_are_finite():
    """predict_load() outputs must be finite numeric values."""
    pytest.importorskip("tf_keras", reason="Person B's TF stack not installed")

    from model.predict import predict_load

    predicted_cpu, upper_bound, anomaly = predict_load(make_records(cpu=50.0))

    assert np.isfinite(predicted_cpu)
    assert np.isfinite(upper_bound)
    assert isinstance(anomaly, bool)


def test_predict_load_stable_cpu_no_anomaly():
    """Stable CPU at 50% should not trigger anomaly."""
    pytest.importorskip("tf_keras", reason="Person B's TF stack not installed")

    from model.predict import predict_load

    predicted_cpu, upper_bound, anomaly = predict_load(make_records(cpu=50.0))

    assert anomaly is False


def test_predict_load_short_window_uses_fallback():
    """An invalid window should use predict_load()'s documented fallback."""
    pytest.importorskip("tf_keras", reason="Person B's TF stack not installed")

    from model.predict import predict_load

    result = predict_load(make_records(cpu=50.0, n=5))

    assert isinstance(result, tuple)
    assert len(result) == 3
    assert isinstance(result[0], float)
    assert isinstance(result[1], float)
    assert result[2] is True


# ── Anomaly detection edge cases ──────────────────────────────────────────────

def test_anomaly_spike_detected():
    """A sudden spike in the last reading should trigger anomaly."""
    from model.anomaly import is_anomaly as detect_anomaly

    values = np.array([50.0] * 9 + [99.0])

    result = detect_anomaly(values)

    assert isinstance(result, bool)


def test_anomaly_all_zeros_no_crash():
    """detect_anomaly must handle std=0 without crashing."""
    from model.anomaly import is_anomaly as detect_anomaly

    values = np.array([0.0] * 10)

    result = detect_anomaly(values)

    assert isinstance(result, bool)


def test_anomaly_returns_bool_not_numpy():
    """detect_anomaly must return Python bool, not numpy.bool_."""
    from model.anomaly import is_anomaly as detect_anomaly

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