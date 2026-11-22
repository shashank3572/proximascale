"""cf_ablation.py (evaluation D2) and timing.py's summary: no TensorFlow needed."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))

import cf_ablation as C  # noqa: E402
import timing  # noqa: E402  (TensorFlow is imported inside main() only)


def _dump(pred_shift, n=40):
    regime = np.array(["normal"] * 30 + ["post_scaling"] * 10)
    in_window = np.zeros(n, dtype=bool)
    in_window[25:] = True
    true = np.full(n, 50.0)
    actual = np.where(regime == "post_scaling", 40.0, 50.0)
    pred = 50.0 + np.where(in_window, pred_shift, 0.0)
    return pd.DataFrame({"seed": 0, "t": np.arange(n), "actual_t3": actual, "true_cpu_t3": true,
                         "regime_t3": regime, "post_scaling_in_window": in_window,
                         "anomaly_flag": False, "predicted_load": pred, "univariate_t3": pred})


def test_compare_reports_paired_difference_within_post_scaling_rows():
    on = {0: _dump(0.0), 1: _dump(0.0)}
    off = {0: _dump(-4.0), 1: _dump(-4.0)}       # without the correction: 4 points low
    table, seeds, flips = C.compare(on, off)
    assert seeds == [0, 1] and flips == 0
    row = table[(table["rows"] == "target is post-scaling") & (table["vs"] == "true load")
                & (table["method"] == C.METHODS["predicted_load"])].iloc[0]
    assert (row["n"], row["rmse_on"], row["rmse_off"]) == (10, 0.0, 4.0)
    assert row["d_rmse"] == pytest.approx(-4.0) and row["bias_off"] == pytest.approx(-4.0)
    untouched = table[(table["rows"] == "all") & (table["vs"] == "recorded")].iloc[0]
    assert untouched["rmse_on"] == pytest.approx(np.sqrt(10 * 100 / 40))


def test_compare_refuses_dumps_of_different_windows():
    shifted = _dump(0.0)
    shifted["t"] += 1
    with pytest.raises(ValueError, match="not the same windows"):
        C.compare({0: _dump(0.0)}, {0: shifted})


def test_timing_summary():
    s = timing.summarize([0.1, 0.2, 0.3, 0.4], interval_s=30)
    assert s["calls"] == 4 and s["mean_ms"] == pytest.approx(250.0)
    assert s["max_ms"] == pytest.approx(400.0)
    assert s["pct_of_interval"] == pytest.approx(100 * 0.25 / 30)
