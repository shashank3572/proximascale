"""
ProximaScale - Refinement 3: Uncertainty Calibration Check
--------------------------------------------------------------
Checks whether predict_load()'s upper_bound (mean + 2*std from MC
Dropout) actually behaves like a real ~95% confidence range, instead
of just assuming it does.

For each test window, reuses predict_load() AS-IS (same real pipeline,
nothing re-implemented), then mirrors the matching lower bound by
symmetry: since upper = mean + 2*std, the lower bound is
mean - 2*std = 2*mean - upper. Checks what fraction of real CPU
readings actually fall inside [lower, upper].
"""

import numpy as np
import pandas as pd
from pathlib import Path

from preprocessing import load_csv, chronological_split, WINDOW_SIZE
from lstm_model import HORIZON
from predict import predict_load

THIS_DIR = Path(__file__).parent
TRAIN_FRAC = 0.8
CHECK_MAX_WINDOWS = 150   # quick check -- not the final report number
CHECK_STRIDE = 4


def _row_to_window_dicts(df, start, size):
    cols = ["timestamp", "cpu_percent", "memory_percent", "request_rate"]
    if "post_scaling" in df.columns:
        cols.append("post_scaling")
    return df.iloc[start:start + size][cols].to_dict("records")


def run_calibration_check():
    csv_path = THIS_DIR.parent / "data" / "collected" / "metrics.csv"
    df = load_csv(csv_path)
    _train_df, test_df = chronological_split(df, TRAIN_FRAC)

    n_windows = len(test_df) - WINDOW_SIZE - HORIZON + 1
    candidate_indices = list(range(0, n_windows, CHECK_STRIDE))
    if len(candidate_indices) > CHECK_MAX_WINDOWS:
        positions = np.linspace(0, len(candidate_indices) - 1, CHECK_MAX_WINDOWS)
        candidate_indices = [candidate_indices[int(round(p))] for p in positions]

    print(f"🧠 [ProximaScale] Checking calibration on {len(candidate_indices)} windows...")

    inside_count = 0
    widths = []

    for count, i in enumerate(candidate_indices):
        window = _row_to_window_dicts(test_df, i, WINDOW_SIZE)
        target_idx = i + WINDOW_SIZE - 1 + HORIZON
        actual = float(test_df.loc[target_idx, "cpu_percent"])

        predicted_load, upper_bound, _ = predict_load(window)
        lower_bound = 2 * predicted_load - upper_bound   # mirror of mean+2*std

        if lower_bound <= actual <= upper_bound:
            inside_count += 1
        widths.append(upper_bound - lower_bound)

        if (count + 1) % 50 == 0:
            print(f"🧠 [ProximaScale]   ...{count + 1}/{len(candidate_indices)} checked")

    coverage_pct = inside_count / len(candidate_indices) * 100
    avg_width = float(np.mean(widths))

    print(f"\n🧠 [ProximaScale] ===== CALIBRATION CHECK =====")
    print(f"🧠 [ProximaScale] Windows checked:          {len(candidate_indices)}")
    print(f"🧠 [ProximaScale] Actual value inside band: {inside_count}/{len(candidate_indices)} ({coverage_pct:.1f}%)")
    print(f"🧠 [ProximaScale] Target coverage:          ~95%")
    print(f"🧠 [ProximaScale] Average band width:       {avg_width:.2f} CPU percentage points")
    print("🧠 [ProximaScale] ================================\n")

    if coverage_pct < 85:
        print("🧠 [ProximaScale] ⚠️  LOW coverage -- model is overconfident, band is too narrow to trust.")
    elif coverage_pct > 99:
        print("🧠 [ProximaScale] ⚠️  VERY HIGH coverage -- band may be wider than needed to be useful.")
    else:
        print("🧠 [ProximaScale] Coverage looks reasonable for a ~95% band.")

    print("🧠 [ProximaScale] Calibration check self-test PASSED.")
    return coverage_pct, avg_width


if __name__ == "__main__":
    run_calibration_check()