"""decision_replay.py (evaluation D1): the replay harness and its scoring.

No TensorFlow needed: the replay drives the real DecisionEngine with a
replica counter and a simulated clock, and scoring is plain arithmetic.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))

import decision_replay as R  # noqa: E402


def _trace(predicted, start="2026-10-14T00:00:00"):
    stamps = pd.date_range(start, periods=len(predicted), freq="30s")
    return pd.DataFrame({"ts_t": stamps.strftime("%Y-%m-%dT%H:%M:%S"), "pred": predicted})


def _policy(**overrides):
    return R.Policy("test", "pred", None, None, overrides, "")


def test_replica_counter_respects_bounds():
    counter = R.ReplicaCounter(1, 3)
    assert counter.scale_down() is False and counter.replicas == 1
    assert [counter.scale_up() for _ in range(3)] == [True, True, False]
    assert counter.replica_count() == 3


def test_simulated_clock_makes_180s_cooldown_six_decisions():
    result = R.replay(_policy(), _trace([95.0] * 20), {"cooldown_seconds": 180})
    assert list(np.flatnonzero(result["signals"] == "scale_up")) == [0, 6, 12, 18]
    assert set(result["reasons"][1:6]) == {"cooldown"}
    assert result["replicas"][-1] == 5


def test_no_cooldown_scales_every_decision_until_max_replicas():
    result = R.replay(_policy(cooldown_seconds=0), _trace([95.0] * 20),
                      {"min_containers": 1, "max_containers": 10})
    assert list(np.flatnonzero(result["signals"] == "scale_up")) == list(range(9))
    assert set(result["reasons"][9:]) == {"max_reached"}


def test_replay_writes_no_scaling_log(tmp_path, monkeypatch):
    import decision.scaling_log as slog
    log_path = tmp_path / "events.json"
    monkeypatch.setattr(slog, "_LOG_PATH", log_path)
    result = R.replay(_policy(), _trace([95.0] * 8 + [10.0] * 12), {"cooldown_seconds": 60})
    actions = np.isin(result["signals"], ["scale_up", "scale_down"])
    assert actions.sum() > 2 and len(result["actions"]) == actions.sum()
    assert not log_path.exists()


def test_score_classifies_spikes_false_alarms_and_reversals():
    cpu = np.full(400, 40.0)
    for onset in (50, 150, 250):
        cpu[onset:onset + 15] = 80.0
    t_rows = np.arange(9, 390)
    signals = np.full(len(t_rows), "hold", dtype=object)
    reasons = np.full(len(t_rows), "in_band", dtype=object)
    for row, action, reason in [
        (48, "scale_up", "cpu_high"),            # spike 1: 60 s ahead
        (54, "scale_up", "cpu_high"),            # during spike 1, so not false
        (60, "scale_down", "cpu_low"),           # 6 rows after an up: a 5-min reversal
        (80, "scale_up", "upper_bound_risk"),    # CPU stays low for 5 min: false
        (152, "scale_up", "anomaly"),            # spike 2: reacted 60 s after onset
    ]:                                           # spike 3 (onset 250): nothing, missed
        signals[row - 9], reasons[row - 9] = action, reason
    replicas = 1 + np.cumsum(signals == "scale_up") - np.cumsum(signals == "scale_down")

    metrics, spikes = R.score(signals, reasons, replicas, t_rows, cpu, [50, 150, 250], 30)

    assert [s["outcome"] for s in spikes] == ["anticipated", "reacted", "missed"]
    assert spikes[0]["lead_s"] == 60 and spikes[1]["delay_s"] == 60
    assert spikes[0]["capacity_up_before_onset"] and not spikes[2]["capacity_up_before_onset"]
    assert (metrics["ups"], metrics["downs"]) == (4, 1)
    by_reason = (metrics["ups_cpu_high"], metrics["ups_upper_bound"], metrics["ups_anomaly"])
    assert by_reason == (2, 1, 1)
    assert metrics["reversals_5min"] == 1      # up@54 -> down@60; down@60 -> up@80 is 20 rows
    assert metrics["false_ups"] == 1           # only the scale-up at row 80
    assert (metrics["anticipated"], metrics["reacted"], metrics["missed"]) == (1, 1, 1)


def test_score_separates_spikes_met_at_max_replicas_from_misses():
    cpu = np.full(200, 40.0)
    cpu[100:115] = 80.0
    t_rows = np.arange(9, 190)
    holds = np.full(len(t_rows), "hold", dtype=object)
    for replicas, outcome in [(10, "at_max"), (4, "missed")]:
        metrics, spikes = R.score(holds, holds, np.full(len(t_rows), replicas), t_rows, cpu,
                                  [100], 30, max_replicas=10)
        assert spikes[0]["outcome"] == outcome
        assert metrics["time_at_max_pct"] == (100.0 if replicas == 10 else 0.0)
