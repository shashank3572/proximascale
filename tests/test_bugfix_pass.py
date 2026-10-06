"""Regression tests for the 2026-09-22 bugfix pass.

Covers the four live-observed bugs:
  A. Collector polluted the frozen 90k training metrics.csv with live rows,
     so the model's input windows crossed a synthetic-to-live discontinuity
     (predicted ~57% CPU while the container idled at 0.02%).
  B. z-score anomaly detection fired "anomaly" scale-ups on an idle container
     (0.35% CPU vs a 0.00-0.05% baseline = z > 20).
  C. SHAP background contained the explained window itself, so attribution
     collapsed to the same ~77/23/0 split regardless of what drove the decision.
  D. Dashboard lead_time read 0 whenever the proactive scale-up kept the
     container below the threshold for the whole visible window.
"""
import builtins
import csv
import os
import sys
import types
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

# ── A. Training data stays frozen; live rows go to the live file ────────────


def test_collector_appends_to_live_file_not_training_csv(monkeypatch):
    pytest.importorskip("docker")
    import monitoring.collector as collector
    import monitoring.storage as storage

    seen = {}
    monkeypatch.setattr(collector, "get_container_cpu_percent", lambda: 12.5)
    monkeypatch.setattr(collector, "get_container_memory_percent", lambda: 40.0)
    monkeypatch.setattr(collector, "collect_request_count", lambda: 3)
    monkeypatch.setattr(collector, "is_post_scaling", lambda: False)
    monkeypatch.setattr(collector.time, "sleep", lambda s: (_ for _ in ()).throw(StopIteration))

    def fake_append_row(record, path=None):
        seen["path"] = path
        raise StopIteration

    monkeypatch.setattr(collector, "append_row", fake_append_row)

    with pytest.raises(StopIteration):
        collector.collect_metrics()

    live = storage.csv_path()
    training = storage.csv_path(training=True)
    assert os.path.basename(live) == "metrics_live.csv"
    assert os.path.basename(training) == "metrics.csv"
    assert seen["path"] == live, "collector must append to the LIVE metrics file"


def test_read_last_n_defaults_to_live_file(monkeypatch, tmp_path):
    import monitoring.storage as storage

    monkeypatch.setenv("PROXIMASCALE_LIVE_METRICS_PATH", str(tmp_path / "live.csv"))
    assert storage.csv_path() == str(tmp_path / "live.csv")
    # Training path is unaffected by the env override.
    assert storage.csv_path(training=True).endswith("metrics.csv")
    assert storage.csv_path(training=True) == storage.TRAINING_METRICS_PATH

    # Append + read go to the live file; the training file is untouched
    # (asserted via open() monitoring, not by reading the real CSV).
    real_open = builtins.open
    opened = []

    def tracking_open(file, *a, **k):
        opened.append(str(file))
        return real_open(file, *a, **k)

    monkeypatch.setattr(builtins, "open", tracking_open)
    storage.append_row(
        types.SimpleNamespace(
            timestamp="2026-09-22T10:00:00", cpu_percent=1.0,
            memory_percent=2.0, request_rate=3, post_scaling=False,
        )
    )
    rows = storage.read_last_n(1)
    assert len(rows) == 1 and rows[-1].cpu_percent == 1.0
    assert not any(f.endswith("collected\\metrics.csv") or
                   f.endswith("collected/metrics.csv") for f in opened)


# ── B. Idle jitter can never be an anomaly ──────────────────────────────────


def test_idle_jitter_is_not_anomaly():
    from model.anomaly import is_anomaly

    # Real observed live sequence: baseline 0.00-0.16, one jittery 0.35.
    # z was ~1.8-20 depending on the window -- either way it fired scale-ups.
    idle = [0.0] * 8 + [0.02, 0.35]
    assert is_anomaly(idle) is False


def test_real_spike_still_flagged():
    from model.anomaly import is_anomaly

    warm = [float(v) for v in (list(range(20, 30)) + [95.0])]
    assert is_anomaly(warm) is True


def test_min_cpu_floor_respects_explicit_override():
    from model.anomaly import is_anomaly

    # Jitter above the 1e-3 std floor so the z-score itself is computable.
    idle = [0.0, 0.02, 0.0, 0.03, 0.01, 0.0, 0.02, 0.01, 0.0, 0.35]
    # With the floor removed, the relative z-score flags it again.
    assert is_anomaly(idle, min_cpu_percent=0.0) is True
    assert is_anomaly(idle, min_cpu_percent=5.0) is False


def test_flat_baseline_jitter_returns_zero_z():
    from model.anomaly import rolling_zscore

    # 0.00 vs 0.02 on an idle container must not produce a huge z-score.
    assert rolling_zscore([0.0, 0.0, 0.0, 0.02]) == 0.0


# ── C. SHAP background comes from the training distribution ────────────────


def test_compute_shap_background_is_training_distribution(monkeypatch, tmp_path):
    pytest.importorskip("tensorflow")
    import numpy as np
    import pandas as pd

    import main
    import model.predict as P

    # A tiny standalone CSV so the test does not depend on the 90k file.
    hint = pd.DataFrame({
        "timestamp": [f"2026-09-19T{h:02d}:00:00" for h in range(24)] * 25,
        "cpu_percent": [float(10 + (h % 24)) for h in range(24)] * 25,
        "memory_percent": [40.0] * 600,
        "request_rate": [100.0] * 600,
    })
    hint_path = tmp_path / "hint.csv"
    hint.to_csv(hint_path, index=False)
    monkeypatch.setattr(P, "TRAINING_DATA_HINT_PATH", hint_path)

    windows_seen = []

    def spy_prepare(window):
        df = pd.DataFrame([r.to_dict() if hasattr(r, "to_dict") else r for r in window])
        windows_seen.append(len(df))
        # Mimic the real scaler's row output: one (n_rows, 3) scaled array.
        return np.zeros((len(df), 3), dtype="float32"), df["cpu_percent"].values

    monkeypatch.setattr(P, "prepare_scaled_window", spy_prepare)
    monkeypatch.setattr(P, "get_shap_artifacts", lambda: ("fake-model", None))

    captured = {}

    class FakeExplainer:
        def __init__(self, model, background, nsamples=256):
            bg = np.asarray(background)
            captured["background_shape"] = bg.shape

        def explain(self, x):
            return {"CPU%": 60.0, "Memory%": 30.0, "Request Rate": 10.0}

    import dashboard.shap_explain as se
    monkeypatch.setattr(se, "ShapExplainer", FakeExplainer)

    records = [
        types.SimpleNamespace(
            to_dict=lambda i=i: {
                "timestamp": f"2026-09-22T10:{i:02d}:00",
                "cpu_percent": 50.0, "memory_percent": 40.0,
                "request_rate": 100, "post_scaling": False,
            }
        )
        for i in range(10)
    ]
    out = main._compute_shap(records)

    assert out == {"CPU%": 60.0, "Memory%": 30.0, "Request Rate": 10.0}
    # Background = a stack of WINDOW_SIZE-length training windows PLUS the
    # current window -- shape (N, 10, 3) with N > 1 (the old set was exactly 2
    # loose points, which let the explainer trivially copy the input).
    assert captured["background_shape"][1:] == (10, 3)
    assert captured["background_shape"][0] > 2
    assert windows_seen[0] == 10  # last prepare call is the explained window


def test_compute_shap_falls_back_when_hint_missing(monkeypatch, tmp_path):
    pytest.importorskip("tensorflow")
    import numpy as np

    import main
    import model.predict as P

    monkeypatch.setattr(P, "TRAINING_DATA_HINT_PATH", tmp_path / "missing.csv")

    def fake_prepare(window):
        n = len(window)
        return np.zeros((n, 3), dtype="float32"), np.zeros(n)

    monkeypatch.setattr(P, "prepare_scaled_window", fake_prepare)
    monkeypatch.setattr(P, "get_shap_artifacts", lambda: ("fake-model", None))

    captured = {}

    class FakeExplainer:
        def __init__(self, model, background, nsamples=256):
            captured["background_rows"] = np.asarray(background).shape[0]

        def explain(self, x):
            return {"CPU%": 70.0, "Memory%": 20.0, "Request Rate": 10.0}

    import dashboard.shap_explain as se
    monkeypatch.setattr(se, "ShapExplainer", FakeExplainer)

    records = [
        types.SimpleNamespace(
            to_dict=lambda i=i: {
                "timestamp": f"2026-09-22T10:{i:02d}:00",
                "cpu_percent": 50.0, "memory_percent": 40.0,
                "request_rate": 100, "post_scaling": False,
            }
        )
        for i in range(10)
    ]
    assert main._compute_shap(records) is not None
    # Fallback = old [current window, neutral] 2-point background.
    assert captured["background_rows"] == 2


# ── D. Lead time stays meaningful for proactive scale-ups ──────────────────


def _df_from_rows(rows):
    import pandas as pd
    return pd.DataFrame(rows)


def test_lead_time_counts_back_to_last_breach_when_no_future_breach():
    from dashboard.live_plot import lead_time

    t0 = datetime(2026, 9, 22, 14, 0, 0)
    rows = [
        # breach before the scale-up, then nothing above threshold after
        {"timestamp": t0, "actual_cpu": 90.0, "signal": "hold"},
        {"timestamp": t0 + timedelta(seconds=30), "actual_cpu": 40.0, "signal": "scale_up"},
        {"timestamp": t0 + timedelta(seconds=60), "actual_cpu": 35.0, "signal": "hold"},
    ]
    # Old behaviour returned 0.0 here; now it measures back to the breach
    # the scale-up preempted.
    assert lead_time(_df_from_rows(rows), 75.0) == 30.0


def test_lead_time_forward_reading_unchanged():
    from dashboard.live_plot import lead_time

    t0 = datetime(2026, 9, 22, 14, 0, 0)
    rows = [
        {"timestamp": t0, "actual_cpu": 40.0, "signal": "scale_up"},
        {"timestamp": t0 + timedelta(seconds=90), "actual_cpu": 80.0, "signal": "hold"},
    ]
    assert lead_time(_df_from_rows(rows), 75.0) == 90.0


def test_lead_time_zero_only_when_no_breach_ever():
    from dashboard.live_plot import lead_time

    t0 = datetime(2026, 9, 22, 14, 0, 0)
    rows = [
        {"timestamp": t0, "actual_cpu": 40.0, "signal": "scale_up"},
        {"timestamp": t0 + timedelta(seconds=30), "actual_cpu": 45.0, "signal": "hold"},
    ]
    assert lead_time(_df_from_rows(rows), 75.0) == 0.0
