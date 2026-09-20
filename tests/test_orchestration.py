"""
tests/test_orchestration.py
Person D — integration tests for main.py's orchestration glue.

Tests ONLY:
  - Person D's normalise_signal() and execute()
  - Person B's predict_load() return contract
  - Person C's DecisionEngine.evaluate(anomaly_flag=...) contract

No sys.modules stubbing — main.py imports its dependencies lazily,
so importing main is safe with real modules present.

Run:
    pytest tests/test_orchestration.py -v
"""
import os
import pytest
from unittest.mock import patch, MagicMock

import main


# ── Person D: normalise_signal ──────────────────────────────────────────────
@pytest.mark.parametrize("raw,expected", [
    ("scale_up",         "scale_up"),
    ("scale_down",       "scale_down"),
    ("hold",             "hold"),
    ("hold_cooldown",    "hold"),
    ("hold_max_reached", "hold"),
    ("hold_min_reached", "hold"),
])
def test_normalise_signal(raw, expected):
    assert main.normalise_signal(raw) == expected


# ── Person D: execute() accepts every signal ────────────────────────────────
@pytest.mark.parametrize("signal", ["scale_up", "scale_down", "hold"])
def test_execute_accepts_signal(signal):
    main.execute(signal)   # must not raise


#Person B's predict_load() return contract
def test_predict_load_contract_shape():
    """
    Person B Semester-2 contract:
    predict_load() returns
    (predicted_cpu, upper_bound, anomaly_flag).
    """
    pytest.importorskip("tensorflow", reason="TF stack not installed")
    pytest.importorskip("prophet", reason="Prophet not installed")

    from model.predict import predict_load

    window = [
        {
            "timestamp": f"2026-09-20T10:{i // 2:02d}:{(i % 2) * 30:02d}",
            "cpu_percent": 40.0,
            "memory_percent": 40.0,
            "request_rate": 10,
            "post_scaling": False,
        }
        for i in range(10)
    ]

    predicted_cpu, upper_bound, anomaly_flag = predict_load(window)

    assert isinstance(predicted_cpu, float)
    assert isinstance(upper_bound, float)
    assert isinstance(anomaly_flag, bool)


# ── Person C contract: engine.evaluate accepts anomaly_flag kwarg ───────────
def test_engine_accepts_anomaly_flag_kwarg():
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "config.yaml"
    )
    with patch("docker.from_env") as mock_docker:
        mock_client = MagicMock()
        mock_docker.return_value = mock_client
        mock_client.images.pull.return_value = None
        mock_client.containers.list.return_value = []

        from decision.engine import DecisionEngine
        engine = DecisionEngine(config_path=config_path)
        signal = engine.evaluate(50.0, anomaly_flag=True)
        assert signal == "scale_up"