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
    """cpu=10 is below lower_threshold (30) → must return scale_down"""
    # Give the engine 2 fake running containers so min_containers (1) is not hit

    c1, c2 = MagicMock(), MagicMock()
    c1.name = "proximascale_1"
    c2.name = "proximascale_2"
    engine.actuator.client.containers.list.return_value = [c1, c2]
    engine.hysteresis.last_action_time = 0

    result = engine.evaluate(predicted_cpu=10.0, anomaly_flag=False)
    assert result == "scale_down", f"Expected 'scale_down', got '{result}'"
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
def test_cooldown_returns_hold_cooldown(engine):
    """After a scale action, the next call within cooldown must return hold_cooldown"""
    # First call — triggers scale_up and records time
    engine.evaluate(predicted_cpu=90.0, anomaly_flag=False)

    # Immediately call again — should be inside cooldown window
    result = engine.evaluate(predicted_cpu=90.0, anomaly_flag=False)
    assert result == "hold_cooldown", (
        f"Expected 'hold_cooldown' during cooldown window, got '{result}'"
    )
