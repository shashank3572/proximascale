"""Residual-stacking architecture checks.

final prediction = Prophet forecast + LSTM residual forecast (ADDITION),
with the counterfactual correction applied to post-scaling rows first.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))

from counterfactual import apply_counterfactual_correction  # noqa: E402


# ── Counterfactual correction (no TF needed) ────────────────────────────────
def _df(cpu, flags):
    return pd.DataFrame({"cpu_percent": cpu, "post_scaling": flags})


def test_unflagged_rows_pass_through_unchanged():
    out = apply_counterfactual_correction(_df([10.0, 20.0, 30.0], [False] * 3))
    assert out["cpu_percent_corrected"].tolist() == [10.0, 20.0, 30.0]


def test_flagged_rows_blend_with_last_pre_scaling_value():
    # baseline = 80 (last unflagged reading); flagged actual = 40
    out = apply_counterfactual_correction(
        _df([70.0, 80.0, 40.0, 40.0], [False, False, True, True]))
    expected = 0.6 * 40.0 + 0.4 * 80.0
    assert out["cpu_percent_corrected"].iloc[2] == pytest.approx(expected)
    assert out["cpu_percent_corrected"].iloc[3] == pytest.approx(expected)
    assert out["cpu_percent_corrected"].iloc[:2].tolist() == [70.0, 80.0]


def test_flagged_from_first_row_falls_back_to_actual():
    out = apply_counterfactual_correction(_df([40.0, 45.0], [True, True]))
    assert out["cpu_percent_corrected"].tolist() == [40.0, 45.0]


def test_input_dataframe_is_not_mutated():
    df = _df([70.0, 40.0], [False, True])
    apply_counterfactual_correction(df)
    assert "cpu_percent_corrected" not in df.columns


# ── Stacking arithmetic (needs TF only to import model.predict) ─────────────
pytest.importorskip("tensorflow", reason="TF stack not installed")
pytest.importorskip("prophet", reason="Prophet not installed")

import model.predict as P  # noqa: E402


def _window(n=10):
    return [{"timestamp": f"2026-09-20T10:{i // 2:02d}:{(i % 2) * 30:02d}",
             "cpu_percent": 50.0, "memory_percent": 40.0, "request_rate": 100,
             "post_scaling": False} for i in range(n)]


def test_final_prediction_is_prophet_plus_residual(monkeypatch):
    monkeypatch.setattr(P, "_load_artifacts", lambda: None)
    monkeypatch.setattr(P, "_scaler", object(), raising=False)
    monkeypatch.setattr(P, "_model", object(), raising=False)
    monkeypatch.setattr(P, "_residual_scaler", object(), raising=False)
    monkeypatch.setattr(P, "_prophet_model", object(), raising=False)
    monkeypatch.setattr(P, "scale_features", lambda df, s: np.zeros((P.WINDOW_SIZE, 3)))
    monkeypatch.setattr(P, "get_prophet_fitted", lambda m, ts: [10.0, 20.0, 30.0])
    monkeypatch.setattr(
        P, "predict_with_uncertainty",
        lambda *a, **k: {"mean": [1.0, 2.0, 3.0], "upper_bound": [2.0, 4.0, 6.0]})

    predicted, upper, anomaly = P.predict_load(_window())

    assert predicted == 30.0 + 3.0          # farthest step: Prophet + residual mean
    assert upper == 30.0 + 6.0              # Prophet + residual upper bound
    assert anomaly is False


# ── Held-out regression: stacking must beat Prophet alone ───────────────────
def test_stacked_beats_prophet_only_on_held_out_tail():
    """Guards against a stale/mismatched artifact or a broken residual path.
    Uses the chronological last 10% of metrics.csv (training used the first 80%)."""
    df = pd.read_csv(ROOT / "data" / "collected" / "metrics.csv")
    n = len(df)
    start = int(n * 0.9)
    idx = list(range(start, n - P.HORIZON - 1, max(1, (n - start) // 20)))[:20]

    err_stacked, err_prophet = [], []
    for i in idx:
        window = df.iloc[i - P.WINDOW_SIZE + 1:i + 1][
            ["timestamp", "cpu_percent", "memory_percent", "request_rate", "post_scaling"]
        ].to_dict("records")
        predicted, _, _ = P.predict_load(window, n_passes=5)
        truth = df["cpu_percent"].iloc[i + P.HORIZON]
        future = [pd.Timestamp(window[-1]["timestamp"])
                  + pd.Timedelta(seconds=P.SAMPLING_INTERVAL_SECONDS) * k
                  for k in range(1, P.HORIZON + 1)]
        prophet_only = P.get_prophet_fitted(P._prophet_model, future)[-1]
        err_stacked.append(abs(predicted - truth))
        err_prophet.append(abs(prophet_only - truth))

    assert np.mean(err_stacked) < np.mean(err_prophet)
