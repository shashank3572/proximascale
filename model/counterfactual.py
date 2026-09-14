"""
ProximaScale - Phase 7: Counterfactual Correction
------------------------------------------------------
THE ORIGINAL NOVELTY. When the system proactively scales resources ahead of
a predicted spike, the CPU/memory readings collected AFTER that scaling
event are no longer a clean signal of "what the load actually was" -- they
reflect load PLUS the effect of your own intervention. Training on that
masked signal teaches the model to under-react to the very spikes it's
supposed to catch (the system's own correctness poisons its future training
data).

This module corrects post-scaling records by blending the actual observed
value back toward a "counterfactual baseline" -- our best estimate of what
the metric would have looked like WITHOUT the scaling event -- so the
training signal isn't entirely masked by the intervention:

    corrected = 0.6 * actual + 0.4 * pre_scaling_baseline

The 0.6/0.4 split is deliberate, not a full reversion: scaling DID have a
real effect, so fully reverting to the pre-scaling baseline (weight 1.0)
would erase real information. Blending keeps most of the observed signal
while pulling back some of what the intervention hid.
"""

import numpy as np
import pandas as pd

ACTUAL_WEIGHT = 0.6
BASELINE_WEIGHT = 0.4


def compute_pre_scaling_baseline(df, value_col="cpu_percent", flag_col="post_scaling"):
    """
    For every contiguous run of post_scaling=True rows, the counterfactual
    baseline is the last value_col reading from BEFORE that run started,
    held constant across the whole run (our best guess of "what it would
    have kept doing without the intervention").

    df: dataframe in time order, with columns [value_col, flag_col].
    Returns a pandas Series, same length/index as df -- NaN where flag_col
    is False, or where post_scaling=True from the very first row (no prior
    value exists yet to use as a baseline).
    """
    flags = df[flag_col].values
    values = df[value_col].values
    baseline = np.full(len(df), np.nan)

    last_pre_scaling_value = None
    for i in range(len(df)):
        if flags[i]:
            if last_pre_scaling_value is not None:
                baseline[i] = last_pre_scaling_value
        else:
            last_pre_scaling_value = values[i]

    return pd.Series(baseline, index=df.index)


def apply_counterfactual_correction(df, value_col="cpu_percent", flag_col="post_scaling",
                                      actual_weight=ACTUAL_WEIGHT, baseline_weight=BASELINE_WEIGHT):
    """
    Returns a NEW dataframe with an added f"{value_col}_corrected" column.
    Rows where flag_col is False are copied through unchanged (nothing was
    suppressed there). Rows where flag_col is True get:
        corrected = actual_weight * actual + baseline_weight * pre_scaling_baseline
    Rows with flag_col True but no prior baseline (edge case: scaling was
    already active at the very start of the data) fall back to the raw
    actual value -- there's nothing to blend against yet.
    """
    try:
        df = df.copy()
        baseline = compute_pre_scaling_baseline(df, value_col, flag_col)

        corrected = df[value_col].astype(float).copy()
        has_baseline = df[flag_col] & baseline.notna()
        corrected[has_baseline] = (
            actual_weight * df.loc[has_baseline, value_col]
            + baseline_weight * baseline[has_baseline]
        )

        df[f"{value_col}_corrected"] = corrected
        return df
    except Exception as e:
        print(f"🧠 [ProximaScale] ERROR in apply_counterfactual_correction: {e}")
        raise


if __name__ == "__main__":
    # metrics.csv doesn't have a post_scaling column yet -- that gets added
    # once the live system is actually scaling things. This self-test builds
    # a small example by hand to prove the math instead.
    #
    # Story: CPU climbs toward 80, the system scales up at row 5, and scaling
    # successfully suppresses CPU back down to ~55 afterward. Uncorrected,
    # the model would see "CPU calmly sitting at 55" right where a real spike
    # was happening -- correction should pull those rows back up toward
    # something between 55 and the pre-scaling trend.
    demo = pd.DataFrame({
        "cpu_percent":  [50, 55, 62, 70, 80,   55, 54, 56, 53],
        "post_scaling": [False, False, False, False, False, True, True, True, True],
    })

    result = apply_counterfactual_correction(demo)
    print("🧠 [ProximaScale] Counterfactual correction demo:")
    print(result[["cpu_percent", "post_scaling", "cpu_percent_corrected"]].to_string(index=False))

    pre_scaling_rows = ~demo["post_scaling"]
    assert (result.loc[pre_scaling_rows, "cpu_percent_corrected"]
            == result.loc[pre_scaling_rows, "cpu_percent"]).all(), \
        "Pre-scaling rows should be unchanged"

    # Row 5 (first post-scaling row): baseline = row 4's value = 80
    # corrected = 0.6*55 + 0.4*80 = 65.0
    expected_row5 = 0.6 * 55 + 0.4 * 80
    actual_row5 = result.loc[5, "cpu_percent_corrected"]
    assert abs(actual_row5 - expected_row5) < 1e-9, \
        f"Row 5 correction wrong: got {actual_row5}, expected {expected_row5}"

    print(f"🧠 [ProximaScale] Row 5 check: corrected={actual_row5:.2f}, expected={expected_row5:.2f}")
    print("🧠 [ProximaScale] Counterfactual correction self-test PASSED.")