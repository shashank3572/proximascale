"""
ProximaScale - Ground truth for the synthetic metrics.csv
----------------------------------------------------------
generate_realistic_data.py writes metrics.csv from a seeded RNG but keeps
no record of which rows belong to a spike, or of what CPU would have been
without its post-scaling drop. The evaluation needs both: regime-stratified
error and the counterfactual check (plan M3, D3) and the anomaly detector's
precision and recall against known spike starts (D3).

generate() would overwrite metrics.csv, so its RNG calls are replayed here,
in the same order, in memory. The rebuilt CPU column must match metrics.csv
exactly, or this stops. No TensorFlow or Prophet needed.
"""

import os
import sys
from datetime import timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import generate_realistic_data as gen  # noqa: E402  constants only -- generate() is never called


def generator_ground_truth(full_df):
    """Per-row ground truth for metrics.csv, rebuilt from the seeded generator.

    Returns a DataFrame aligned with full_df's rows:
        regime       : normal, ramp, hold, decay or post_scaling
        spike_start  : True on the first ramp row of each generated spike
        true_cpu     : the CPU the generator produced BEFORE its post-scaling
                       0.8x drop, i.e. the load the counterfactual correction
                       tries to recover (the recorded CPU on all other rows)
        seasonal_cpu : the generator's noiseless base load (35 + daily sine +
                       weekly sine). On normal rows the recorded CPU is this
                       plus N(0, 3^2) noise, so it is the forecast of an oracle
                       that knows the generator but not its noise.

    Raises ValueError unless the rebuilt timestamps, CPU column and
    post_scaling flags all match full_df exactly.
    """
    n = gen.N_ROWS
    if len(full_df) != n:
        raise ValueError(f"metrics.csv has {len(full_df)} rows, but the seed-{gen.SEED} "
                         f"generator writes {n}; restore it with "
                         "`git checkout -- data/collected/metrics.csv`")

    rng = np.random.default_rng(gen.SEED)
    times = [gen.START_TIME + timedelta(seconds=gen.INTERVAL_SECONDS * i) for i in range(n)]
    hour_of_day = np.array([t.hour + t.minute / 60 + t.second / 3600 for t in times])
    day_of_week = np.array([t.weekday() for t in times])

    segment_size = n // gen.N_SPIKES
    spike_starts = []
    for i in range(gen.N_SPIKES):
        earliest = i * segment_size + 20
        latest = i * segment_size + segment_size - gen.SPIKE_LEN - gen.POST_SCALING_LEN - 20
        if latest <= earliest:
            latest = earliest + 1
        spike_starts.append(int(rng.integers(earliest, latest)))
    spike_peak = {s: rng.uniform(78, 92) for s in spike_starts}

    noise_cpu = rng.normal(0, 3, n)
    daily_pattern = 20 * np.sin(2 * np.pi * (hour_of_day - 9) / 24)
    weekly_pattern = 8 * np.sin(2 * np.pi * day_of_week / 7)
    seasonal = 35 + daily_pattern + weekly_pattern
    base_cpu = seasonal + noise_cpu

    cpu = base_cpu.copy()
    regime = np.full(n, "normal", dtype=object)
    ramp, hold = gen.SPIKE_RAMP, gen.SPIKE_HOLD
    for s in spike_starts:
        peak, base_at_start = spike_peak[s], base_cpu[s]
        for k in range(gen.SPIKE_LEN):
            idx = s + k
            if idx >= n:
                continue
            if k < ramp:
                frac = (k + 1) / ramp
                cpu[idx] = base_at_start + frac * (peak - base_at_start)
                regime[idx] = "ramp"
            elif k < ramp + hold:
                cpu[idx] = peak + rng.normal(0, 1)
                regime[idx] = "hold"
            else:
                frac = (k - ramp - hold + 1) / gen.SPIKE_DECAY
                cpu[idx] = peak - frac * (peak - base_cpu[idx])
                regime[idx] = "decay"
        for k in range(gen.POST_SCALING_LEN):
            idx = s + gen.SPIKE_LEN + k
            if idx < n:
                cpu[idx] = base_cpu[idx] * 0.8
                regime[idx] = "post_scaling"

    recorded = full_df["cpu_percent"].to_numpy(dtype=float)
    post = regime == "post_scaling"
    same_times = np.array_equal(pd.DatetimeIndex(times).as_unit("s").asi8,
                                pd.DatetimeIndex(full_df["timestamp"]).as_unit("s").asi8)
    same_cpu = np.allclose(np.round(np.clip(cpu, 5.0, 98.0), 2), recorded, rtol=0, atol=1e-9)
    same_flags = np.array_equal(post, full_df["post_scaling"].to_numpy(dtype=bool))
    if not (same_times and same_cpu and same_flags):
        raise ValueError(f"metrics.csv does not match generate_realistic_data.py "
                         f"(seed {gen.SEED}): timestamps {same_times}, cpu {same_cpu}, "
                         f"post_scaling {same_flags}. "
                         "Restore it with `git checkout -- data/collected/metrics.csv`.")

    true_cpu = np.where(post, np.round(np.clip(base_cpu, 5.0, 98.0), 2), recorded)
    return pd.DataFrame({
        "regime": regime.astype(str),
        "spike_start": np.isin(np.arange(n), spike_starts),
        "true_cpu": true_cpu,
        "seasonal_cpu": seasonal,
    })
