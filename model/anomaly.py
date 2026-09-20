"""
ProximaScale - Phase 6: Anomaly Detection
---------------------------------------------
Flags a CPU reading as an anomaly (spike) using a rolling Z-score: how many
standard deviations the latest reading sits above the recent baseline.

Directional by design -- this flags upward SPIKES (z > threshold), not dips.
A sudden drop in CPU isn't something autoscaling needs to react to the same
way a surge is, so it isn't flagged here. If you want both directions later,
swap `z > threshold` for `abs(z) > threshold` in is_anomaly().
"""

import os
import numpy as np
import yaml

THRESHOLD = 2.5   # fallback default if config.yaml has no override
ROLLING_WINDOW = 10  # matches the LSTM's WINDOW_SIZE, for consistency

_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"
)

def _resolve_config_path() -> str:
    return os.environ.get("PROXIMASCALE_CONFIG_PATH", _DEFAULT_CONFIG_PATH)

def _configured_threshold(default: float = THRESHOLD) -> float:
    try:
        with open(_resolve_config_path(), "r") as f:
            return float(yaml.safe_load(f).get("anomaly_z_threshold", default))
    except (OSError, ValueError, AttributeError, yaml.YAMLError):
        return default

def rolling_zscore(values):
    """
    values: array-like of raw (unscaled) cpu_percent readings, latest LAST.
    Baseline = all but the latest reading. Returns the z-score of the latest
    reading relative to that baseline, or 0.0 if there isn't enough history
    or the baseline has zero variance (can't divide by a zero std).
    """
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return 0.0
    baseline, latest = values[:-1], values[-1]
    mean, std = baseline.mean(), baseline.std()
    if std == 0:
        return 0.0
    return float((latest - mean) / std)


def is_anomaly(cpu_values, threshold=None):
    """
    cpu_values: array-like of raw cpu_percent readings, latest LAST.
    threshold: explicit override; if None, reads anomaly_z_threshold from
               config.yaml (falling back to THRESHOLD).
    Returns True if the latest reading is an upward spike (z > threshold).
    """
    if threshold is None:
        threshold = _configured_threshold()
    try:
        z = rolling_zscore(cpu_values)
        return bool(z > threshold)
    except Exception as e:
        print(f"🧠 [ProximaScale] ERROR in is_anomaly: {e}")
        return False  # never let a bad anomaly check crash the caller


def detect_anomalies_series(cpu_series, window_size=ROLLING_WINDOW, threshold=None):
    """
    Batch version for offline evaluation (Phase 10): slides a window_size-long
    baseline across a full series and flags each point as anomalous or not.
    cpu_series: 1D array-like of raw cpu_percent values, in time order.
    threshold:  explicit override; if None, resolved from config.yaml once.
    Returns a numpy bool array, same length as cpu_series (the first
    window_size points are always False -- not enough history yet to judge).
    """
    if threshold is None:
        threshold = _configured_threshold()
    cpu_series = np.asarray(cpu_series, dtype=float)
    flags = np.zeros(len(cpu_series), dtype=bool)
    for i in range(window_size, len(cpu_series)):
        recent = cpu_series[i - window_size: i + 1]  # baseline + point, latest last
        flags[i] = is_anomaly(recent, threshold)
    return flags


if __name__ == "__main__":
    from pathlib import Path
    from preprocessing import load_csv

    THIS_DIR = Path(__file__).parent
    csv_path = THIS_DIR.parent / "data" / "collected" / "metrics.csv"
    df = load_csv(csv_path)
    cpu_series = df["cpu_percent"].values

    rng = np.random.default_rng(42)
    baseline = list(50 + rng.normal(0, 2, 10))
    normal_reading = baseline + [51.0]
    assert is_anomaly(normal_reading) is False, "Normal reading incorrectly flagged"
    print("🧠 [ProximaScale] Normal reading correctly NOT flagged")

    spike_reading = baseline + [95.0]
    assert is_anomaly(spike_reading) is True, "Obvious spike NOT flagged"
    print("🧠 [ProximaScale] Obvious spike correctly flagged")

    flags = detect_anomalies_series(cpu_series)
    flagged_indices = np.where(flags)[0]
    print(f"🧠 [ProximaScale] Flagged {len(flagged_indices)} anomalous points out of {len(cpu_series)}")
    print(f"🧠 [ProximaScale] Flagged indices: {list(flagged_indices)}")

    print("🧠 [ProximaScale] Anomaly detection self-test PASSED.")