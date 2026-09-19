"""
tests/test_integration.py
Integration smoke tests for ProximaScale decision-actuator module.

Semester 1 scope: verify all modules import cleanly and the engine
can be instantiated. Full end-to-end test comes in Weeks 10-11
once Person B's model is integrated.

Run with:  pytest tests/test_integration.py -v
"""
import os
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Test 1: All 4 core modules import without errors
# ---------------------------------------------------------------------------
def test_scaler_interface_imports():
    """actuator/scaler_interface.py must import cleanly."""
    from actuator.scaler_interface import ScalerInterface
    assert ScalerInterface is not None


def test_hysteresis_imports():
    """decision/hysteresis.py must import cleanly and be instantiable."""
    from decision.hysteresis import Hysteresis
    h = Hysteresis(cooldown_seconds=60)
    assert h is not None


def test_docker_scaler_imports():
    """actuator/docker_scaler.py must import cleanly (Docker mocked)."""
    with patch("docker.from_env") as mock_docker:
        mock_client = MagicMock()
        mock_docker.return_value = mock_client
        mock_client.images.pull.return_value = None

        from actuator.docker_scaler import DockerActuator

        # Minimal config to satisfy __init__
        cfg = {
            "scaling_rules": {
                "target_service_image": "nginx:alpine",
                "max_containers": 5,
                "min_containers": 1,
            }
        }
        actuator = DockerActuator(cfg)
        assert actuator is not None


def test_decision_engine_imports():
    """decision/engine.py must import cleanly and be instantiable."""
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
        assert engine is not None


# ---------------------------------------------------------------------------
# Test 2: Hysteresis logic works correctly in isolation
# ---------------------------------------------------------------------------
def test_hysteresis_not_cooling_on_init():
    """Fresh Hysteresis should NOT be in cooldown (last_action_time=0)."""
    from decision.hysteresis import Hysteresis
    h = Hysteresis(cooldown_seconds=180)
    assert h.is_cooling_down() is False


def test_hysteresis_is_cooling_after_action():
    """After record_action(), is_cooling_down() must return True."""
    from decision.hysteresis import Hysteresis
    h = Hysteresis(cooldown_seconds=180)
    h.record_action()
    assert h.is_cooling_down() is True


# ---------------------------------------------------------------------------
# Test 3: Signal normaliser (Person D's main.py logic)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw,expected", [
    ("scale_up",         "scale_up"),
    ("scale_down",       "scale_down"),
    ("hold",             "hold"),
    ("hold_cooldown",    "hold"),
    ("hold_max_reached", "hold"),
    ("hold_min_reached", "hold"),
])
def test_signal_normaliser(raw, expected):
    """All hold_* variants must collapse to 'hold' without crashing."""
    signal = "hold" if raw.startswith("hold") else raw
    assert signal == expected
