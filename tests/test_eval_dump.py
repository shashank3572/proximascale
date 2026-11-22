"""eval_dump.py (evaluation Step 0) must match predict_load() and be reproducible.

The dump batches predict_load()'s pipeline over every test window; these
checks pin the batched copy to the real function, so the evaluation tables
cannot drift from what the live system computes.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))

pytest.importorskip("tensorflow", reason="TF stack not installed")
pytest.importorskip("prophet", reason="Prophet not installed")

import eval_dump as D  # noqa: E402
import predict  # noqa: E402

N_PASSES = 5


@pytest.fixture(scope="module")
def data():
    df = D.load_csv(D.DATA_CSV)
    if len(df) != D.gen.N_ROWS:
        pytest.skip(f"needs the {D.gen.N_ROWS}-row synthetic metrics.csv "
                    "the artifacts were trained on")
    _, test = D.chronological_split(df, D.ev.TRAIN_FRAC)
    predict._load_artifacts()
    return df, test


def _first_post_scaling_run(test):
    flags = test["post_scaling"].to_numpy(dtype=bool)
    return int(np.flatnonzero(flags[1:] & ~flags[:-1])[0]) + 1


def _interesting_windows(test):
    """First and last windows, two around a spike onset, and two overlapping a
    post-scaling run -- one starting inside it, where the correction has no
    pre-scaling baseline and must fall back to the raw reading."""
    run_start = _first_post_scaling_run(test)
    onset = D.ev._detect_spike_onsets(test["cpu_percent"].to_numpy(), D.ev.SCALE_THRESHOLD)[0]
    n_windows = len(test) - D.WINDOW_SIZE - D.HORIZON + 1
    picks = [0, run_start - 5, run_start + 2,
             onset - D.WINDOW_SIZE - 1, onset - D.WINDOW_SIZE + 1, n_windows - 1]
    return sorted({i for i in picks if 0 <= i < n_windows})


class _CyclingDropout:
    """Deterministic stand-in for MC Dropout: the k-th of every n calls adds
    0.01*k to the real dropout-off output, so the mean and spread are fixed
    and identical whether windows run one at a time or in one batch."""

    def __init__(self, model, n_passes):
        self.model, self.n_passes, self.calls = model, n_passes, 0

    def __call__(self, x, training=False):
        k = self.calls % self.n_passes
        self.calls += 1
        return self.model(x, training=False) + 0.01 * k


def test_batched_hybrid_matches_predict_load(data, monkeypatch):
    _, test = data
    windows = D.build_windows(test, _interesting_windows(test))
    real_model = predict._model
    monkeypatch.setattr(predict, "_model", _CyclingDropout(real_model, N_PASSES))
    expected = [predict.predict_load(w, n_passes=N_PASSES) for w in windows]

    X, anomaly = D.preprocess_windows(windows, predict._scaler, *D.anomaly_settings())
    prophet_fc = D.prophet_forecast(predict._prophet_model, windows)
    mean, std, upper = D.mc_hybrid(_CyclingDropout(real_model, N_PASSES), X,
                                   predict._residual_scaler, prophet_fc, N_PASSES)

    assert np.all(std > 0)  # the sigma path is exercised, not just the mean
    for j, (load, bound, flag) in enumerate(expected):
        assert D.clamp(mean[j, -1]) == pytest.approx(load, abs=1e-3)
        assert D.clamp(upper[j, -1]) == pytest.approx(bound, abs=1e-3)
        assert anomaly[j] == flag
    # Prophet-alone column agrees with evaluate.py's own batched helper
    np.testing.assert_allclose(prophet_fc[:, -1],
                               D.ev._prophet_alone_predict_batch(predict._prophet_model, windows),
                               atol=1e-9)


def test_same_seed_reproduces_and_seeds_differ(data):
    _, test = data
    windows = D.build_windows(test, range(32))
    X, _ = D.preprocess_windows(windows, predict._scaler, *D.anomaly_settings())
    no_prophet = np.zeros((len(windows), D.HORIZON))

    def run(seed):
        model = D.load_lstm(D.MODEL_PATH, seed)
        return D.mc_hybrid(model, X, predict._residual_scaler, no_prophet, N_PASSES)[0]

    first, again, other = run(0), run(0), run(1)
    np.testing.assert_array_equal(first, again)
    assert not np.allclose(first, other)


def test_counterfactual_off_changes_only_flagged_cpu(data):
    _, test = data
    i = _first_post_scaling_run(test) - 5
    on = D.build_windows(test, [i])
    off = D.build_windows(test, [i], counterfactual=False)
    x_on, _ = D.preprocess_windows(on, predict._scaler, *D.anomaly_settings())
    x_off, _ = D.preprocess_windows(off, predict._scaler, *D.anomaly_settings())

    flagged = np.array([row["post_scaling"] for row in on[0]])
    assert flagged.any() and not flagged.all()
    assert not np.allclose(x_on[0, flagged, 0], x_off[0, flagged, 0])
    np.testing.assert_array_equal(x_on[0, ~flagged], x_off[0, ~flagged])
    np.testing.assert_array_equal(x_on[0, :, 1:], x_off[0, :, 1:])  # memory, request rate
