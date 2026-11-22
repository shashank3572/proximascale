"""train_variants.py / variant_eval.py (evaluation D4, C1-C3, C6 and the
model search): the data plumbing, without TensorFlow. That the inference
path matches predict_load() is enforced when variant_eval.py runs: it must
reproduce the committed models' prediction dump before it scores anything."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))

import train_variants as T  # noqa: E402
import variant_eval as V  # noqa: E402
from counterfactual import apply_counterfactual_correction  # noqa: E402


def test_make_windows_targets_follow_inputs():
    inputs = np.arange(20, dtype=float)[:, None]
    X, y = T.make_windows(inputs, np.arange(20, dtype=float) * 10, window=4, horizon=3)
    assert X.shape == (14, 4, 1) and y.shape == (14, 3)
    np.testing.assert_array_equal(X[0, :, 0], [0, 1, 2, 3])
    np.testing.assert_array_equal(y[0], [40, 50, 60])
    np.testing.assert_array_equal(y[-1], [170, 180, 190])     # never past the end


def test_period_windows_match_make_windows():
    t = T.period_t_rows(0, 20, window=4)
    rows = T.window_rows(t, 4)
    np.testing.assert_array_equal(rows, T.make_windows(np.arange(20)[:, None],
                                                       np.zeros(20), window=4)[0][:, :, 0])
    assert t[-1] + 3 == 19                                     # last target is the last row


def test_validation_is_the_latest_tenth_in_time_order():
    df = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=100, freq="30s")})
    fit, val = T.split_fit_val(df)
    assert (len(fit), len(val)) == (90, 10)
    assert fit["timestamp"].max() < val["timestamp"].min()


def test_variant_specs():
    assert set(V.GRID_2X2) <= set(T.VARIANTS)
    assert T.VARIANTS["hybrid"]["inputs"] == list(T.FEATURE_COLS)
    assert T.VARIANTS["hybrid"]["target"] == "residual"
    assert T.VARIANTS["hybrid_cf_off"]["correct"] is False
    assert all(T.VARIANTS[v]["correct"] for v in V.GRID_2X2)     # the 2x2 all use the correction
    for name, spec in T.VARIANTS.items():
        assert set(spec) == set(T.DEFAULTS) | {"inputs"}, name
        assert name in V.LABELS


def test_build_inputs_channels():
    cpu = np.arange(10.0, 30.0)
    arrays = {"yhat": np.arange(20.0) * 0.5, "memory_percent": np.full(20, 40.0),
              "request_rate": np.arange(20.0) * 10}
    rows = T.window_rows(np.array([9, 12]), 5)
    X = T.build_inputs(["cpu_percent", "resid", "yhat", "yhat_f3", "d_cpu", "d_req",
                        "memory_percent"], cpu[rows], rows, arrays)
    assert X.shape == (2, 5, 7)
    np.testing.assert_array_equal(X[0, :, 0], cpu[5:10])
    np.testing.assert_array_equal(X[0, :, 1], cpu[5:10] - arrays["yhat"][5:10])
    np.testing.assert_array_equal(X[0, :, 3], arrays["yhat"][8:13])        # 3 steps ahead
    np.testing.assert_array_equal(X[0, :, 4], [0, 1, 1, 1, 1])             # within-window deltas
    np.testing.assert_array_equal(X[0, :, 5], [0, 10, 10, 10, 10])
    per_row = T.row_channels(["cpu_percent", "resid", "yhat_f3", "d_cpu"], cpu, arrays,
                             np.arange(1, 15))
    np.testing.assert_array_equal(per_row[0], [11.0, 11.0 - 0.5, 2.0, 1.0])


def _full(n=80):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="30s"),
        "cpu_percent": np.round(50 + rng.normal(0, 5, n), 2),
        "memory_percent": np.round(40 + rng.normal(0, 2, n), 2),
        "request_rate": np.round(500 + rng.normal(0, 30, n), 1),
        "post_scaling": False})
    df.loc[40:46, "post_scaling"] = True
    return df


def test_frame_inputs_are_predict_load_windows():
    df = _full()
    frame = V.Frame(df, 30, 80, {"committed": np.arange(80, dtype=float)})
    spec = {**T.DEFAULTS, "inputs": list(T.FEATURE_COLS)}
    corrected = frame.inputs(spec)
    raw = frame.inputs({**spec, "correct": False})
    assert corrected.shape == (len(frame.t), 10, 3)
    assert frame.t_abs[0] == 39 and frame.t[0] == 9          # same windows as eval_dump
    for j in (0, 5, 12):                         # windows before, overlapping, after a run
        t = frame.t_abs[j]
        window = df.iloc[t - 9:t + 1]
        expected = apply_counterfactual_correction(window)["cpu_percent_corrected"].to_numpy()
        np.testing.assert_allclose(corrected[j, :, 0], expected, atol=1e-9)
        np.testing.assert_array_equal(raw[j, :, 0], window["cpu_percent"].to_numpy())
    long = frame.inputs({**spec, "window": 20})
    np.testing.assert_array_equal(long[0, :, 1], df["memory_percent"].to_numpy()[20:40])
    np.testing.assert_array_equal(frame.prophet(), frame.t_abs + 3)
    np.testing.assert_array_equal(frame.actual, df["cpu_percent"].to_numpy()[frame.t_abs + 3])


def test_ensemble_pools_passes():
    a = {"det": np.array([1.0]), "passes": np.array([[1.0], [3.0]])}
    b = {"det": np.array([3.0]), "passes": np.array([[5.0], [7.0]])}
    out = V.pool([a, b])
    assert out["mc"][0] == 4.0 and out["det"][0] == 2.0
    assert out["sigma"][0] == np.std([1.0, 3.0, 5.0, 7.0])


def test_blend_weight_is_chosen_on_validation():
    frame = V.Frame(_full(), 30, 80, {"committed": np.arange(80, dtype=float)})
    w, table = V.tune_blend(frame, frame.actual.copy())     # C1 perfect on validation
    assert w == 1.0 and len(table) == len(V.BLEND_GRID)


def test_lead_differences_are_paired_over_onsets():
    rng = np.random.default_rng(0)
    leads = {"hybrid": [np.array([60.0, 60.0, 30.0]), np.array([60.0, 30.0, 30.0])],
             "univariate": [np.array([30.0, 30.0, 30.0]), np.array([30.0, 30.0, 0.0])]}
    (row,) = V.lead_differences(leads, "hybrid", rng, n_boot=2000)
    assert row["versus"] == "univariate"
    assert row["d_lead_s"] == np.mean([30.0, 15.0, 15.0])
    assert 15.0 <= row["d_lead_lo"] <= row["d_lead_s"] <= row["d_lead_hi"] <= 30.0
