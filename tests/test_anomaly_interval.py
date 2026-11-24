"""Regression tests for the 2026-11-23 fix: one spike, one anomaly scale-up.

Before: the anomaly branch bypassed the cooldown unconditionally, so a
sustained 100% CPU spike re-fired scale_up on EVERY poll. The config_demo.yaml
live run produced three containers in 70 s from a single 4-minute spike.

After: anomaly scale-ups are spaced by `anomaly_min_interval_seconds`, measured
from the last SUCCESSFUL scale_up of any reason (anomaly, upper_bound_risk,
cpu_high, current_cpu_high). While the interval has not elapsed the engine
returns "hold" with last_reason == "anomaly_interval". Unset/0 disables the
check, so config.yaml (report numbers) is behaviourally unchanged.
"""
import os
import time
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def isolated_scaling_log(tmp_path, monkeypatch):
    """Keep these tests from appending to the real data/scaling_events.json."""
    import decision.scaling_log as slog
    monkeypatch.setattr(slog, "_LOG_PATH", tmp_path / "scaling_events.json")


def _engine(**cfg):
    """DecisionEngine with Docker mocked out; never in cooldown, never scaled."""
    with patch("docker.from_env") as mock_docker:
        client = MagicMock()
        mock_docker.return_value = client
        client.containers.list.return_value = []      # 0 workers -> scale_up allowed
        client.containers.run.return_value = MagicMock()

        from decision.engine import DecisionEngine
        eng = DecisionEngine(config={
            "cooldown_seconds": 60,
            "tick_seconds": 3,
            "max_containers": 10,
            "min_containers": 0,
            "use_anomaly_bypass": True,
            "use_upper_bound_risk": True,
            **cfg,
        })
        eng.hysteresis.last_action_time = 0
        eng.hysteresis.last_scale_up_time = 0
        return eng


# ── Hysteresis: the spacing clock ────────────────────────────────────────────
def test_hysteresis_tracks_scale_ups_only():
    from decision.hysteresis import Hysteresis
    h = Hysteresis(cooldown_seconds=60)

    assert h.scale_up_interval_elapsed(300) is True    # nothing has scaled up yet
    h.record_action("scale_up")
    assert h.scale_up_interval_elapsed(300) is False
    h.record_action("scale_down")
    assert h.scale_up_interval_elapsed(300) is False   # untouched by a scale_down


def test_interval_of_zero_or_none_always_elapsed():
    from decision.hysteresis import Hysteresis
    h = Hysteresis(cooldown_seconds=60)
    h.record_action("scale_up")
    assert h.scale_up_interval_elapsed(0) is True
    assert h.scale_up_interval_elapsed(None) is True


# ── Engine: the actual behaviour ─────────────────────────────────────────────
def test_sustained_anomaly_scales_up_only_once():
    """The live-demo storm: 100% CPU for many polls must not add a container
    per poll."""
    eng = _engine(anomaly_min_interval_seconds=300)

    assert eng.evaluate(predicted_cpu=50.0, anomaly_flag=True) == "scale_up"
    assert eng.last_reason == "anomaly"

    for _ in range(4):                       # every later poll of the same spike
        assert eng.evaluate(predicted_cpu=100.0, anomaly_flag=True) == "hold"
        assert eng.last_reason == "anomaly_interval"


def test_interval_is_keyed_off_any_scale_up():
    """A cpu_high scale-up starts the clock just like an anomaly one does."""
    eng = _engine(anomaly_min_interval_seconds=300)

    assert eng.evaluate(predicted_cpu=90.0, upper_bound=None, anomaly_flag=False) == "scale_up"
    assert eng.last_reason == "cpu_high"

    assert eng.evaluate(predicted_cpu=100.0, anomaly_flag=True) == "hold"
    assert eng.last_reason == "anomaly_interval"


def test_anomaly_fires_again_once_the_interval_elapses():
    eng = _engine(anomaly_min_interval_seconds=300)
    eng.evaluate(predicted_cpu=50.0, anomaly_flag=True)

    long_ago = time.time() - 301
    eng.hysteresis.last_scale_up_time = long_ago
    eng.hysteresis.last_action_time = long_ago

    assert eng.evaluate(predicted_cpu=100.0, anomaly_flag=True) == "scale_up"
    assert eng.last_reason == "anomaly"


def test_scale_down_does_not_defer_the_next_anomaly():
    eng = _engine(anomaly_min_interval_seconds=300)
    eng.hysteresis.record_action("scale_down")

    assert eng.evaluate(predicted_cpu=100.0, anomaly_flag=True) == "scale_up"
    assert eng.last_reason == "anomaly"


def test_interval_unset_keeps_the_old_anomaly_bypass():
    """config.yaml does not set the key: anomaly still bypasses the cooldown."""
    eng = _engine()

    assert eng.evaluate(predicted_cpu=50.0, anomaly_flag=True) == "scale_up"
    assert eng.last_reason == "anomaly"
    assert eng.evaluate(predicted_cpu=50.0, anomaly_flag=True) == "scale_up"
    assert eng.last_reason == "anomaly"


def test_demo_config_enables_the_spacing():
    import yaml
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config_demo.yaml",
    )
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    assert cfg["anomaly_min_interval_seconds"] == 300
