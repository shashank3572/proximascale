#!/usr/bin/env python3
"""
generate_varied_data.py - training data with varied spike shapes
-----------------------------------------------------------------
generate_realistic_data.py gives every spike the same shape: a 90 s ramp
(3 readings), a 7-minute hold, a 90 s decay, one spike every ~4 hours. A
model trained on it learns that one shape: on spikes that ramp more slowly
(results/unseen, "shifted") it forecasts them late.

This generator keeps EVERYTHING else identical (the same 31 days and
timestamps, daily and weekly pattern, noise, memory leak, request rate and
the 5-minute post-scaling dip), so the committed Prophet still fits it, but
draws each spike's shape at random:

    ramp     1-8 readings   (30 s to 4 min to reach the peak)
    hold     4-16 readings  (2 to 8 minutes at the peak)
    decay    1-8 readings   (30 s to 4 min back down)
    peak     75-95 % CPU    (was 78-92)
    gap      150-800 readings between spike starts (75 min to 6.7 h; was a
             fixed ~4 h grid), so spikes come in bursts and in quiet spells

    python model/generate_varied_data.py
Writes data/collected/metrics_varied.csv (90,000 rows, same columns as metrics.csv).
"""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 1001
N_ROWS = 90000
INTERVAL_SECONDS = 30
START_TIME = datetime(2026, 9, 19, 0, 0, 0)          # the same timestamps as metrics.csv
RAMP = (1, 8)
HOLD = (4, 16)
DECAY = (1, 8)
PEAK = (75.0, 95.0)
GAP = (150, 800)
POST_SCALING_LEN = 10
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "collected" / "metrics_varied.csv"


def generate(seed=SEED, n_rows=N_ROWS, start_time=START_TIME, output_path=OUTPUT_PATH,
             ramp=RAMP, hold=HOLD, decay=DECAY, peak=PEAK, gap=GAP):
    rng = np.random.default_rng(seed)
    timestamps = [start_time + timedelta(seconds=INTERVAL_SECONDS * i) for i in range(n_rows)]
    hour_of_day = np.array([t.hour + t.minute / 60 + t.second / 3600 for t in timestamps])
    day_of_week = np.array([t.weekday() for t in timestamps])

    # ---- spikes: random shapes and gaps, never overlapping a post-scaling dip
    spikes = []
    s = int(rng.integers(20, gap[1]))
    while True:
        r, h, d = (int(rng.integers(lo, hi + 1)) for lo, hi in (ramp, hold, decay))
        if s + r + h + d + POST_SCALING_LEN + 20 >= n_rows:
            break
        spikes.append((s, r, h, d, float(rng.uniform(*peak))))
        s = s + max(r + h + d + POST_SCALING_LEN + 20, int(rng.integers(gap[0], gap[1] + 1)))

    in_spike = np.zeros(n_rows, dtype=bool)
    post_scaling = np.zeros(n_rows, dtype=bool)
    for s, r, h, d, _ in spikes:
        in_spike[s:s + r + h + d] = True
        post_scaling[s + r + h + d:s + r + h + d + POST_SCALING_LEN] = True

    # ---- CPU: identical base process to generate_realistic_data.py
    noise_cpu = rng.normal(0, 3, n_rows)
    daily = 20 * np.sin(2 * np.pi * (hour_of_day - 9) / 24)
    weekly = 8 * np.sin(2 * np.pi * day_of_week / 7)
    base_cpu = 35 + daily + weekly + noise_cpu
    cpu = base_cpu.copy()
    for s, r, h, d, pk in spikes:
        base_at_start = base_cpu[s]
        for k in range(r):
            cpu[s + k] = base_at_start + (k + 1) / r * (pk - base_at_start)
        for k in range(h):
            cpu[s + r + k] = pk + rng.normal(0, 1)
        for k in range(d):
            idx = s + r + h + k
            cpu[idx] = pk - (k + 1) / d * (pk - base_cpu[idx])
        post = s + r + h + d
        cpu[post:post + POST_SCALING_LEN] = base_cpu[post:post + POST_SCALING_LEN] * 0.8
    cpu = np.clip(cpu, 5.0, 98.0)

    # ---- memory and request rate: as generate_realistic_data.py
    leak = np.zeros(n_rows)
    level = 0.0
    for i in range(n_rows):
        level += 0.0015
        if post_scaling[i] and (i == 0 or not post_scaling[i - 1]):
            level *= 0.85
        leak[i] = level
    memory = np.clip(40 + 0.5 * (cpu - 30) + rng.normal(0, 2, n_rows) + leak, 20.0, 95.0)
    daytime = np.where((hour_of_day >= 9) & (hour_of_day <= 21), 200, 0)
    request_rate = np.clip(100 + 8 * cpu + daytime + np.where(in_spike, 300, 0)
                           + rng.normal(0, 40, n_rows), 0.0, 1500.0)

    df = pd.DataFrame({
        "timestamp": [t.strftime("%Y-%m-%dT%H:%M:%S") for t in timestamps],
        "cpu_percent": np.round(cpu, 2),
        "memory_percent": np.round(memory, 2),
        "request_rate": np.round(request_rate, 1),
        "post_scaling": post_scaling,
    })
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    ramps = np.array([sp[1] for sp in spikes])
    print(f"[DataGen] {len(df)} rows, {len(spikes)} spikes (ramp {ramps.min()}-{ramps.max()} "
          f"readings, mean {ramps.mean():.1f}), {int(post_scaling.sum())} post-scaling rows "
          f"-> {output_path}")
    return df, spikes


if __name__ == "__main__":
    generate()
