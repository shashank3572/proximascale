#!/usr/bin/env python3
"""
generate_realistic_data.py

Synthetic container-metrics generator for an autoscaling ML project.
Produces data/collected/metrics.csv with:
  - realistic daily/weekly CPU seasonality (learnable by Prophet)
  - sharp, non-overlapping CPU spikes, 1 per ~500 rows (learnable by an LSTM)
  - memory correlated with CPU plus a slow leak that resets after scale events
  - request_rate correlated with CPU and time-of-day
  - a post_scaling flag marking the ~10 rows right after each spike, where
    CPU is artificially depressed (the "actuator responded" counterfactual)

Note on the daily-pattern formula: a raw 20*sin(2*pi*hour/24) peaks at
hour=6, not 3 PM. To actually satisfy "peaks ~3 PM, low ~3 AM" (the
stated intent), the sine is phase-shifted by -9 hours below.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

SEED = 42
N_ROWS = 90000  # ~31 days at 30s intervals — enough span for Prophet to learn daily/weekly cycles
INTERVAL_SECONDS = 30
START_TIME = datetime(2026, 9, 19, 0, 0, 0)  # current date

# Keep the same spike density as the 20k-row version (1 per ~500 rows)
N_SPIKES = N_ROWS // 500  # 180 spikes at N_ROWS=90000
# Ramp/hold/decay shape, widened vs. the minimal 3/5/4 split so enough rows
# clear the 75% threshold to satisfy the density requirements below, while
# staying within the spec's stated 10-20 row spike-duration range.
SPIKE_RAMP = 3
SPIKE_HOLD = 14
SPIKE_DECAY = 3
SPIKE_LEN = SPIKE_RAMP + SPIKE_HOLD + SPIKE_DECAY  # 20 rows = 10 minutes
POST_SCALING_LEN = 10  # rows (5 minutes)

OUTPUT_PATH = Path(__file__).parent.parent / "data" / "collected" / "metrics.csv"

def generate():
    rng = np.random.default_rng(SEED)

    print(f"🧠 [DataGen] Generating {N_ROWS} rows of synthetic metrics...")

    # ---- Timestamps ----
    timestamps = [START_TIME + timedelta(seconds=INTERVAL_SECONDS * i) for i in range(N_ROWS)]
    hour_of_day = np.array([t.hour + t.minute / 60 + t.second / 3600 for t in timestamps])
    day_of_week = np.array([t.weekday() for t in timestamps])  # 0=Mon .. 6=Sun

    span_days = (timestamps[-1] - timestamps[0]).total_seconds() / 86400
    print(f"🧠 [DataGen] Timestamp span: {span_days:.2f} days")

    # ---- Spike placement (non-overlapping, one per ~500-row segment) ----
    segment_size = N_ROWS // N_SPIKES
    spike_starts = []
    for i in range(N_SPIKES):
        seg_start = i * segment_size
        seg_end = seg_start + segment_size
        earliest = seg_start + 20
        latest = seg_end - SPIKE_LEN - POST_SCALING_LEN - 20
        if latest <= earliest:
            latest = earliest + 1
        start = int(rng.integers(earliest, latest))
        spike_starts.append(start)

    spike_peak = {s: rng.uniform(78, 92) for s in spike_starts}

    # Per-row spike/post-scaling bookkeeping
    spike_state = np.full(N_ROWS, -1)  # spike start index this row belongs to, else -1
    post_scaling = np.zeros(N_ROWS, dtype=bool)

    for s in spike_starts:
        for k in range(SPIKE_LEN):
            idx = s + k
            if idx < N_ROWS:
                spike_state[idx] = s
        post_start = s + SPIKE_LEN
        for k in range(POST_SCALING_LEN):
            idx = post_start + k
            if idx < N_ROWS:
                post_scaling[idx] = True

    # ---- CPU ----
    noise_cpu = rng.normal(0, 3, N_ROWS)
    daily_pattern = 20 * np.sin(2 * np.pi * (hour_of_day - 9) / 24)  # peak ~15:00, trough ~03:00
    weekly_pattern = 8 * np.sin(2 * np.pi * day_of_week / 7)
    base_cpu = 35 + daily_pattern + weekly_pattern + noise_cpu

    cpu = base_cpu.copy()

    for s in spike_starts:
        peak = spike_peak[s]
        base_at_start = base_cpu[s]
        for k in range(SPIKE_LEN):
            idx = s + k
            if idx >= N_ROWS:
                continue
            if k < SPIKE_RAMP:  # ramp up over 3 rows
                frac = (k + 1) / SPIKE_RAMP
                cpu[idx] = base_at_start + frac * (peak - base_at_start)
            elif k < SPIKE_RAMP + SPIKE_HOLD:  # hold at peak for 5 rows
                cpu[idx] = peak + rng.normal(0, 1)
            else:  # decay back to baseline over 4 rows
                decay_k = k - SPIKE_RAMP - SPIKE_HOLD + 1
                frac = decay_k / SPIKE_DECAY
                target = base_cpu[idx]
                cpu[idx] = peak - frac * (peak - target)

        post_start = s + SPIKE_LEN
        for k in range(POST_SCALING_LEN):
            idx = post_start + k
            if idx >= N_ROWS:
                continue
            cpu[idx] = base_cpu[idx] * 0.8  # ~20% artificial drop from added capacity

    cpu = np.clip(cpu, 5.0, 98.0)

    # ---- Memory (correlated with CPU, slow leak, relief after scale events) ----
    leak = np.zeros(N_ROWS)
    leak_level = 0.0
    for i in range(N_ROWS):
        leak_level += 0.0015  # slow drift = simulated memory leak
        if post_scaling[i] and (i == 0 or not post_scaling[i - 1]):
            leak_level *= 0.85  # drop 15% at the start of each scale event
        leak[i] = leak_level

    noise_mem = rng.normal(0, 2, N_ROWS)
    memory = 40 + 0.5 * (cpu - 30) + noise_mem + leak
    memory = np.clip(memory, 20.0, 95.0)

    # ---- Request rate (correlated with CPU + time of day, spikes during CPU spikes) ----
    daytime_boost = np.where((hour_of_day >= 9) & (hour_of_day <= 21), 200, 0)
    spike_boost = np.where(spike_state >= 0, 300, 0)
    noise_req = rng.normal(0, 40, N_ROWS)
    request_rate = 100 + 8 * cpu + daytime_boost + spike_boost + noise_req
    request_rate = np.clip(request_rate, 0.0, 1500.0)

    # ---- Assemble ----
    df = pd.DataFrame({
        "timestamp": [t.strftime("%Y-%m-%dT%H:%M:%S") for t in timestamps],
        "cpu_percent": np.round(cpu, 2),
        "memory_percent": np.round(memory, 2),
        "request_rate": np.round(request_rate, 1),
        "post_scaling": post_scaling,
    })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"🧠 [DataGen] Wrote {len(df)} rows to {OUTPUT_PATH}")

    # ---- Data quality checks ----
    rows_above_75 = int((cpu > 75).sum())
    rows_above_85 = int((cpu > 85).sum())
    post_scaling_count = int(post_scaling.sum())

    print(f"🧠 [DataGen] Total rows: {len(df)}")
    print(f"🧠 [DataGen] CPU min={cpu.min():.2f} max={cpu.max():.2f} "
          f"mean={cpu.mean():.2f} std={cpu.std():.2f}")
    print(f"🧠 [DataGen] Rows with CPU > 75%: {rows_above_75}")
    print(f"🧠 [DataGen] Rows with CPU > 85%: {rows_above_85}")
    print(f"🧠 [DataGen] Rows with post_scaling=True: {post_scaling_count}")
    print(f"🧠 [DataGen] request_rate min={request_rate.min():.2f} "
          f"max={request_rate.max():.2f} mean={request_rate.mean():.2f}")

    above = cpu > 75
    detected_spikes = int(np.sum(above[1:] & ~above[:-1]) + (1 if above[0] else 0))
    print(f"🧠 [DataGen] Detected spikes (contiguous CPU>75 runs): {detected_spikes}")

    # Thresholds scale with N_ROWS/N_SPIKES so density stays constant regardless of dataset size
    min_rows_above_75 = int(0.025 * N_ROWS)   # ~613/20000 in the reference run
    min_post_scaling = int(0.9 * N_SPIKES * POST_SCALING_LEN)

    assert cpu.std() > 8, "CPU std too low — not enough variance"
    assert cpu.max() > 85, "CPU max too low — spikes not present"
    assert rows_above_75 > min_rows_above_75, "Not enough high-CPU rows for spike learning"
    assert post_scaling_count > min_post_scaling, "Not enough post_scaling counterfactual rows"

    print("🧠 [DataGen] All data quality assertions passed ✅")


if __name__ == "__main__":
    generate()
