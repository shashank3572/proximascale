"""
tests/test_orchestration.py
Person D — integration tests for main.py's orchestration glue.

Tests ONLY:
  - Person D's normalise_signal() and execute()
  - Person B's predict() return contract
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


# ── Person B contract: predict() returns dict with the right keys ───────────
# ── Person B contract: predict() returns dict with the right keys ───────────
def test_predict_contract_shape():
    """
    Skips cleanly if Person B's ML stack (tf_keras / TF) isn't
    installed in this environment. Runs when it is.
    """
    pytest.importorskip("tf_keras", reason="Person B's TF stack not installed")
    from model.predict import predict
    result = predict([{"cpu": 40}] * 10)
    assert "predicted_cpu" in result
    assert "anomaly" in result
    assert isinstance(result["predicted_cpu"], list)
    assert isinstance(result["anomaly"], bool)


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