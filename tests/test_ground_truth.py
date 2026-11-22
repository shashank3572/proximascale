"""ground_truth.py: the seeded generator, replayed in memory, must reproduce
metrics.csv and label every row. No TensorFlow needed."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))

import generate_realistic_data as gen  # noqa: E402
from ground_truth import generator_ground_truth  # noqa: E402
from preprocessing import load_csv  # noqa: E402


@pytest.fixture(scope="module")
def df():
    data = load_csv(ROOT / "data" / "collected" / "metrics.csv")
    if len(data) != gen.N_ROWS:
        pytest.skip(f"needs the {gen.N_ROWS}-row synthetic metrics.csv")
    return data


def test_ground_truth_matches_generator_structure(df):
    truth = generator_ground_truth(df)  # raises unless metrics.csv is the seeded output
    n = gen.N_SPIKES
    counts = truth["regime"].value_counts()
    assert truth["spike_start"].sum() == n
    assert counts["ramp"] == n * gen.SPIKE_RAMP
    assert counts["hold"] == n * gen.SPIKE_HOLD
    assert counts["decay"] == n * gen.SPIKE_DECAY
    assert counts["post_scaling"] == n * gen.POST_SCALING_LEN

    post = (truth["regime"] == "post_scaling").to_numpy()
    recorded = df["cpu_percent"].to_numpy()
    true_cpu = truth["true_cpu"].to_numpy()
    np.testing.assert_allclose(recorded[post], np.clip(true_cpu[post] * 0.8, 5.0, 98.0), atol=0.01)
    np.testing.assert_array_equal(true_cpu[~post], recorded[~post])


def test_seasonal_cpu_is_the_noiseless_base_load(df):
    truth = generator_ground_truth(df)
    normal = (truth["regime"] == "normal").to_numpy()
    noise = df["cpu_percent"].to_numpy()[normal] - truth["seasonal_cpu"].to_numpy()[normal]
    assert abs(noise.mean()) < 0.05
    assert 2.9 < noise.std() < 3.05          # the generator's N(0, 3^2), slightly clipped at 5%
    starts = np.flatnonzero(truth["spike_start"].to_numpy())
    assert np.all(truth["regime"].to_numpy()[starts] == "ramp")
