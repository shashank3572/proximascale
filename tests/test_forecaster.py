"""The v2 model (model/forecaster.py): the live path predict_load() takes and
eval_dump.py --bundle's batched path must build the same inputs, and the
pooled seed ensemble must be what predict_load() returns."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))

pytest.importorskip("tensorflow")
import predict as P  # noqa: E402

BUNDLE = P.BUNDLE_DIR                                   # the bundle predict.py serves
if not (BUNDLE / "config.json").exists():
    pytest.skip("stacked bundle not installed", allow_module_level=True)

from forecaster import StackedForecaster  # noqa: E402
from preprocessing import load_csv  # noqa: E402

DATA = ROOT / "data" / "collected" / "metrics.csv"


@pytest.fixture(scope="module")
def setup():
    df = load_csv(DATA)
    P._load_artifacts()
    fc = StackedForecaster(BUNDLE, P._prophet_model, seed=0)
    post = np.flatnonzero(df["post_scaling"].to_numpy(dtype=bool))
    run = post[post > 72000][0]                       # first post-scaling run in the test split
    t_abs = np.array([72100, run - 5, run + 3, run + 40, len(df) - 4])
    return df, fc, t_abs


def test_live_and_batched_inputs_match(setup):
    df, fc, t_abs = setup
    X_batch, prophet_batch = fc.history_inputs(df, t_abs)
    assert X_batch.shape == (len(t_abs), fc.window, len(fc.inputs))
    for j, t in enumerate(t_abs):
        live, prophet_live, _ = fc.window_inputs(df.iloc[t - fc.window + 1:t + 1])
        np.testing.assert_allclose(live, X_batch[j], atol=1e-9)
        np.testing.assert_allclose(prophet_live, prophet_batch[j], atol=1e-9)


def test_predict_load_serves_v2_with_30_readings(setup, monkeypatch):
    # The model's ABSOLUTE output is what the batched evaluation path computes.
    # Live calibration (model/predict.py) re-anchors the level on the measured
    # CPU and is a separate layer, covered by tests/test_remediation.py.
    monkeypatch.setenv("PROXIMASCALE_LIVE_CALIBRATION", "off")
    df, fc, t_abs = setup
    cols = ["timestamp", "cpu_percent", "memory_percent", "request_rate", "post_scaling"]
    t = int(t_abs[0])
    window = df.iloc[t - 29:t + 1][cols].to_dict("records")
    X, prophet = fc.history_inputs(df, [t])
    many = fc.forecast(X, prophet, n_passes=600)       # MC mean, tight
    predicted, upper, anomaly = P.predict_load(window)
    assert isinstance(predicted, float) and isinstance(anomaly, bool)
    sigma = float(many["sigma"][0, -1])
    assert abs(predicted - many["mean"][0, -1]) < 2 * sigma          # 30 passes vs 600
    assert upper >= predicted
    # 10-29 readings: the original model, unchanged
    legacy = P.predict_load(window[-10:], n_passes=5)
    assert legacy != (predicted, upper, anomaly) and len(legacy) == 3
    assert P.HISTORY_SIZE == 30 and P.V2_WINDOW == 30


def test_pool_counts_every_member(setup):
    _, fc, t_abs = setup
    assert len(fc.members) == 3
    X, prophet = fc.history_inputs(setup[0], t_abs[:2])
    out = fc.forecast(X, prophet, n_passes=30)
    assert out["mean"].shape == (2, 3)
    np.testing.assert_allclose(out["upper"], fc.upper(out["mean"], out["sigma"]))
    assert np.all(out["upper"] > out["mean"])
