"""
ProximaScale - Synthetic Dataset Generator (stand-in for Person A's data)
--------------------------------------------------------------------------
Person A's real collected dataset is only 82 rows at 60s intervals --
too short for a meaningful train/test split or for Prophet to see any
daily pattern. This generates a synthetic replacement that:

  - Matches Person A's locked schema (a.md) exactly: timestamp,
    cpu_percent, memory_percent, request_rate, post_scaling.
  - Uses a 30s polling interval to match what predict.py already
    assumes. NOTE: this differs from a.md's "poll interval: 60
    seconds" -- flag this to your team.
  - Spans multiple SIMULATED days, each with a repeating daily
    traffic "bump" (mirrors the lunch/dinner-rush pattern a.md cites
    for Swiggy/Zomato), so Prophet has actual day-to-day structure
    to learn instead of just noise.
  - Makes cpu_percent respond to request_rate with a short lag +
    noise, instead of an unrelated pure random walk -- this is
    exactly why your Multivariate LSTM lost to Reactive last time:
    the old dummy data had zero learnable structure by construction.
  - Injects sharp spikes (Locust spike_load.py-style) and, after
    each one crosses SCALE_THRESHOLD, simulates the actuator firing:
    cpu_percent is artificially suppressed for ~5 minutes and tagged
    post_scaling=True, so Phase 7's counterfactual correction
    finally has real work to do (it was a no-op before).

This is a development/evaluation placeholder, not real data --
swap in Person A's actual CSV the moment more of it exists. Same
path, same schema, zero downstream code changes needed.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Config -- tune these
# ---------------------------------------------------------------------------
INTERVAL_SECONDS = 30
NUM_DAYS = 5
ROWS_PER_DAY = 400                       # 400 * 30s = ~3h20m "active" window/day
TOTAL_ROWS = NUM_DAYS * ROWS_PER_DAY     # 2000

START_DATE = datetime(2024, 1, 15, 9, 0, 0)

BASELINE_CPU = 15.0
BASELINE_MEM = 30.0
BASELINE_REQ = 20.0

SPIKES_PER_DAY = 1
SCALE_THRESHOLD = 75.0          # keep this in sync with evaluate.py's SCALE_THRESHOLD
POST_SCALING_STEPS = 10         # ~5 min at 30s polling, matching a.md's "5 cycles"

RNG_SEED = 42


def _daily_traffic_curve(n_rows, rng):
    """Quiet -> lunch/dinner-rush-style surge -> quiet, plus AR(1) noise."""
    t = np.linspace(0, 1, n_rows)
    surge = np.exp(-((t - 0.6) ** 2) / (2 * 0.08 ** 2))
    curve = BASELINE_REQ + surge * 220
    noise = np.zeros(n_rows)
    for i in range(1, n_rows):
        noise[i] = 0.85 * noise[i - 1] + rng.normal(0, 6)
    return np.clip(curve + noise, 2, None)


def _inject_spikes(request_rate, rng, n_spikes):
    """Overlay Locust spike_load.py-style sharp bursts."""
    n = len(request_rate)
    spike_starts = rng.choice(np.arange(20, n - 60), size=n_spikes, replace=False)
    for start in sorted(spike_starts):
        ramp_len = int(rng.integers(6, 12))
        hold_len = int(rng.integers(8, 16))
        for i in range(ramp_len):
            idx = start + i
            if idx < n:
                request_rate[idx] += (i / ramp_len) * 300
        for i in range(hold_len):
            idx = start + ramp_len + i
            if idx < n:
                request_rate[idx] += 300 + rng.normal(0, 15)
    return request_rate, sorted(spike_starts)


def generate_dataset():
    rng = np.random.default_rng(RNG_SEED)
    all_rows = []
    cpu_prev = BASELINE_CPU

    for day in range(NUM_DAYS):
        day_start = START_DATE + timedelta(days=day)
        request_rate = _daily_traffic_curve(ROWS_PER_DAY, rng)
        request_rate, _ = _inject_spikes(request_rate, rng, SPIKES_PER_DAY)

        cpu = np.zeros(ROWS_PER_DAY)
        mem = np.zeros(ROWS_PER_DAY)
        post_scaling = np.zeros(ROWS_PER_DAY, dtype=bool)
        scaling_cooldown = 0

        for i in range(ROWS_PER_DAY):
            lag_idx = max(0, i - 2)
            target_cpu = BASELINE_CPU + (request_rate[lag_idx] / 300) * 75
            target_cpu += rng.normal(0, 2.5)

            if scaling_cooldown > 0:
                target_cpu *= 0.45          # actuator "just fired" -- artificial drop
                post_scaling[i] = True
                scaling_cooldown -= 1

            cpu_val = float(np.clip(0.7 * cpu_prev + 0.3 * target_cpu, 1, 100))
            cpu[i] = cpu_val
            cpu_prev = cpu_val
            mem[i] = float(np.clip(BASELINE_MEM + cpu_val * 0.35 + rng.normal(0, 3), 5, 95))

            if cpu_val > SCALE_THRESHOLD and scaling_cooldown == 0:
                scaling_cooldown = POST_SCALING_STEPS

        timestamps = [day_start + timedelta(seconds=INTERVAL_SECONDS * i) for i in range(ROWS_PER_DAY)]
        for i in range(ROWS_PER_DAY):
            all_rows.append({
                "timestamp": timestamps[i].strftime("%Y-%m-%dT%H:%M:%S"),
                "cpu_percent": round(float(cpu[i]), 2),
                "memory_percent": round(float(mem[i]), 2),
                "request_rate": round(float(request_rate[i]), 1),
                "post_scaling": bool(post_scaling[i]),
            })

    return pd.DataFrame(all_rows)


if __name__ == "__main__":
    df = generate_dataset()

    out_path = Path(__file__).parent.parent / "data" / "collected" / "metrics.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"🧠 [ProximaScale] Generated {len(df)} rows across {NUM_DAYS} simulated days")
    print(f"🧠 [ProximaScale] Saved -> {out_path}")
    print(f"🧠 [ProximaScale] CPU range: {df['cpu_percent'].min():.1f}% - {df['cpu_percent'].max():.1f}%")
    print(f"🧠 [ProximaScale] post_scaling=True rows: {int(df['post_scaling'].sum())} "
          f"({df['post_scaling'].sum() / len(df) * 100:.1f}%)")
    print(f"🧠 [ProximaScale] Rows above {SCALE_THRESHOLD}%: {int((df['cpu_percent'] > SCALE_THRESHOLD).sum())}")

    assert len(df) == TOTAL_ROWS
    assert set(df.columns) == {"timestamp", "cpu_percent", "memory_percent", "request_rate", "post_scaling"}
    assert df["post_scaling"].sum() > 0, "No post_scaling rows -- correction will be a no-op again"
    assert (df["cpu_percent"] > SCALE_THRESHOLD).sum() > 0, "No spikes above threshold -- lead time will read 0 again"
    print("🧠 [ProximaScale] Synthetic dataset self-test PASSED.")