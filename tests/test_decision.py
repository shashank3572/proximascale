"""
tests/test_decision.py
Pytest test suite for DecisionEngine.evaluate().

Docker is fully mocked — these tests run without Docker Desktop.
Run with:  pytest tests/test_decision.py -v
"""
import os
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Fixture: build a DecisionEngine with Docker completely mocked out
# ---------------------------------------------------------------------------
@pytest.fixture
def engine():
    """
    Returns a DecisionEngine instance where:
    - docker.from_env() is patched → no real Docker connection
    - client.images.pull() is a no-op
    - containers.list() returns [] by default (0 running workers)
    - containers.run() is a no-op
    """
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "config.yaml"
    )

    with patch("docker.from_env") as mock_docker:
        mock_client = MagicMock()
        mock_docker.return_value = mock_client

        # Default: no containers running → scale_up is always possible
        mock_client.containers.list.return_value = []
        mock_client.containers.run.return_value = MagicMock()

        from decision.engine import DecisionEngine
        eng = DecisionEngine(config_path=config_path)

        # Reset hysteresis so cooldown never blocks tests
        eng.hysteresis.last_action_time = 0

        yield eng


# ---------------------------------------------------------------------------
# Test 1: high CPU → scale_up
# ---------------------------------------------------------------------------
def test_high_cpu_triggers_scale_up(engine):
    """cpu=90 is above upper_threshold (75) → must return scale_up"""
    result = engine.evaluate(predicted_cpu=90.0, anomaly_flag=False)
    assert result == "scale_up", f"Expected 'scale_up', got '{result}'"


# ---------------------------------------------------------------------------
# Test 2: low CPU → scale_down
# ---------------------------------------------------------------------------
def test_low_cpu_triggers_scale_down(engine):
    """cpu=10 is below lower_threshold (30) → must return scale_down."""
    # Need >min_containers workers so scale_down is actually allowed
    engine.actuator._workers = lambda: [MagicMock(), MagicMock()]
    engine.hysteresis.last_action_time = 0   # ensure no cooldown

    result = engine.evaluate(predicted_cpu=10.0, upper_bound=None, anomaly_flag=False)

    assert result == "scale_down", f"Expected 'scale_down', got '{result}'"
    assert engine.last_reason == "cpu_low"


# ---------------------------------------------------------------------------
# Test 3: mid-range CPU → hold
# ---------------------------------------------------------------------------
def test_mid_cpu_triggers_hold(engine):
    """cpu=50 is between thresholds (30–75) → must return hold"""
    result = engine.evaluate(predicted_cpu=50.0, anomaly_flag=False)
    assert result == "hold", f"Expected 'hold', got '{result}'"


# ---------------------------------------------------------------------------
# Test 4: anomaly_flag=True → scale_up regardless of CPU value
# ---------------------------------------------------------------------------
def test_anomaly_flag_forces_scale_up_regardless_of_cpu(engine):
    """
    cpu=10 would normally trigger scale_down.
    But anomaly_flag=True must bypass thresholds and force scale_up.
    """
    result = engine.evaluate(predicted_cpu=10.0, anomaly_flag=True)
    assert result == "scale_up", (
        f"Expected 'scale_up' due to anomaly flag, got '{result}'"
    )


# ---------------------------------------------------------------------------
# Test 5 (bonus): cooldown is respected after an action
# ---------------------------------------------------------------------------
def test_cooldown_returns_hold_with_reason(engine):
    """After a real scale action, subsequent calls inside cooldown return
    'hold' (locked 3-value contract) with last_reason == 'cooldown'."""
    r1 = engine.evaluate(predicted_cpu=90.0, upper_bound=None, anomaly_flag=False)
    assert r1 == "scale_up"
    assert engine.last_reason == "cpu_high"

    r2 = engine.evaluate(predicted_cpu=90.0, upper_bound=None, anomaly_flag=False)
    assert r2 == "hold"
    assert engine.last_reason == "cooldown"

def test_upper_bound_triggers_scale_up_even_when_mean_is_safe(engine):
    # mean 50 is below static upper 75, but upper_bound 85 > threshold
    action = engine.evaluate(predicted_cpu=50.0, upper_bound=85.0, anomaly_flag=False)
    assert action == "scale_up"


def test_upper_bound_not_triggered_when_low(engine):
    # warm the window so adaptive bounds are ~50 ± 5
    for _ in range(15):
        engine.evaluate(predicted_cpu=50.0, upper_bound=50.0, anomaly_flag=False)
    engine.hysteresis.last_action_time = 0       # clear cooldown if any
    action = engine.evaluate(predicted_cpu=50.0, upper_bound=60.0, anomaly_flag=False)
    assert action == "hold"


def test_scaling_event_written_on_action(tmp_path, monkeypatch, engine):
    import json
    import decision.scaling_log as slog
    monkeypatch.setattr(slog, "_LOG_PATH", tmp_path / "events.json")

    engine.evaluate(predicted_cpu=95.0, upper_bound=99.0, anomaly_flag=False)

    events = json.loads((tmp_path / "events.json").read_text())
    assert events[-1]["action"] == "scale_up"
    assert {"action", "timestamp", "expires_after_steps"} <= events[-1].keys()

def test_anomaly_bypasses_cooldown_then_starts_fresh_cooldown(engine):
    """
    Full sequence under the locked 3-value contract:
      1. High CPU            → scale_up, starts cooldown
      2. High CPU again      → hold (reason=cooldown)
      3. Anomaly while cool  → scale_up (bypasses cooldown)
      4. High CPU again      → hold (anomaly re-armed the clock)
    """
    # Step 1 — normal scale_up
    r1 = engine.evaluate(predicted_cpu=90.0, upper_bound=None, anomaly_flag=False)
    assert r1 == "scale_up"
    assert engine.last_reason == "cpu_high"

    # Step 2 — cooldown is now armed
    r2 = engine.evaluate(predicted_cpu=90.0, upper_bound=None, anomaly_flag=False)
    assert r2 == "hold"
    assert engine.last_reason == "cooldown"

    # Step 3 — anomaly ignores cooldown, fires scale_up, re-arms the clock
    r3 = engine.evaluate(predicted_cpu=10.0, upper_bound=None, anomaly_flag=True)
    assert r3 == "scale_up"
    assert engine.last_reason == "anomaly"

    # Step 4 — must be cooling down again from step 3
    r4 = engine.evaluate(predicted_cpu=90.0, upper_bound=None, anomaly_flag=False)
    assert r4 == "hold"
    assert engine.last_reason == "cooldown"

import json

def test_no_event_logged_when_actuator_is_none(tmp_path, monkeypatch, engine):
    """If actuator is unavailable, evaluate() returns 'hold' and does NOT log."""
    import decision.scaling_log as slog
    monkeypatch.setattr(slog, "_LOG_PATH", tmp_path / "events.json")

    engine.actuator = None
    result = engine.evaluate(predicted_cpu=90.0, upper_bound=None, anomaly_flag=False)

    assert result == "hold"
    assert engine.last_reason == "cpu_high_no_actuator"
    assert not (tmp_path / "events.json").exists()


def test_no_event_logged_when_at_max_replicas(tmp_path, monkeypatch, engine):
    """If scale_up returns False (max reached), do NOT log, do NOT arm cooldown."""
    import decision.scaling_log as slog
    monkeypatch.setattr(slog, "_LOG_PATH", tmp_path / "events.json")

    engine.actuator.scale_up = lambda: False
    result = engine.evaluate(predicted_cpu=90.0, upper_bound=None, anomaly_flag=False)

    assert result == "hold"
    assert engine.last_reason == "max_reached"
    assert not (tmp_path / "events.json").exists()
    assert not engine.hysteresis.is_cooling_down()   # cooldown NOT armed


def test_no_event_logged_when_actuator_raises(tmp_path, monkeypatch, engine):
    """An SDK exception must be caught, treated as hold, and NOT logged."""
    import decision.scaling_log as slog
    monkeypatch.setattr(slog, "_LOG_PATH", tmp_path / "events.json")

    def boom():
        raise RuntimeError("docker daemon unreachable")
    engine.actuator.scale_up = boom

    result = engine.evaluate(predicted_cpu=90.0, upper_bound=None, anomaly_flag=False)

    assert result == "hold"
    assert engine.last_reason == "cpu_high_error"
    assert not (tmp_path / "events.json").exists()