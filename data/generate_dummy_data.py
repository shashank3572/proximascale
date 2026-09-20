"""
ProximaScale - Phase 0: Dummy Data Generator
---------------------------------------------
Generates synthetic system metrics (cpu_percent, memory_percent, request_rate)
as a bounded random walk with 3 injected sharp spikes, so the pipeline
(preprocessing -> Prophet -> LSTM residual -> stacking -> uncertainty) can be built and
tested before real collected data exists.

Usage:
    python data/generate_dummy_data.py
"""

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

# --- Config ---
N_ROWS = 2000
START_TIME = datetime(2026, 1, 1, 0, 0, 0)
INTERVAL_SECONDS = 30       # one reading every 30s
SPIKE_COUNT = 3
SPIKE_DURATION = 15         # rows a spike lasts (ramp + hold + decay)
SEED = 42

random.seed(SEED)

# Paths resolved relative to this file -- never hardcode absolute paths
THIS_DIR = Path(__file__).parent
OUTPUT_DIR = THIS_DIR / "collected"
OUTPUT_FILE = OUTPUT_DIR / "metrics.csv"


def random_walk_series(n, start, step_std, low, high):
    """Bounded random walk."""
    vals = [start]
    for _ in range(n - 1):
        nxt = vals[-1] + random.gauss(0, step_std)
        nxt = max(low, min(high, nxt))
        vals.append(nxt)
    return vals


def inject_spikes(series, n_rows, n_spikes, duration, peak_range):
    """Injects n_spikes sharp ramp-hold-decay spikes at spaced-out locations."""
    series = series[:]
    max_start = n_rows - duration - 1
    gap = max_start // (n_spikes + 1)
    spike_starts = [
        gap * (i + 1) + random.randint(-gap // 4, gap // 4) for i in range(n_spikes)
    ]

    for start in spike_starts:
        peak = random.uniform(*peak_range)
        base = series[start]
        ramp = duration // 3
        hold = duration // 3
        decay = duration - ramp - hold
        for i in range(ramp):
            series[start + i] = base + (peak - base) * (i / ramp)
        for i in range(hold):
            series[start + ramp + i] = peak + random.uniform(-2, 2)
        for i in range(decay):
            series[start + ramp + hold + i] = peak - (peak - base) * (i / decay)

    return series, spike_starts


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cpu = random_walk_series(N_ROWS, start=25.0, step_std=1.5, low=5.0, high=60.0)
    mem = random_walk_series(N_ROWS, start=40.0, step_std=1.0, low=15.0, high=70.0)
    req = random_walk_series(N_ROWS, start=50.0, step_std=4.0, low=0.0, high=200.0)

    cpu, cpu_spikes = inject_spikes(cpu, N_ROWS, SPIKE_COUNT, SPIKE_DURATION, (85.0, 98.0))
    # Memory + request rate rise alongside CPU spikes (correlated load event,
    # not independent noise -- this matters for the multivariate LSTM later)
    for s in cpu_spikes:
        for i in range(SPIKE_DURATION):
            idx = s + i
            if idx < N_ROWS:
                mem[idx] = min(95.0, mem[idx] + 20 + random.uniform(-3, 3))
                req[idx] = min(500.0, req[idx] + 150 + random.uniform(-10, 10))

    rows = []
    t = START_TIME
    for i in range(N_ROWS):
        rows.append(
            {
                "timestamp": t.isoformat(),
                "cpu_percent": round(cpu[i], 2),
                "memory_percent": round(mem[i], 2),
                "request_rate": round(req[i], 2),
            }
        )
        t += timedelta(seconds=INTERVAL_SECONDS)

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["timestamp", "cpu_percent", "memory_percent", "request_rate"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"🧠 [ProximaScale] Generated {len(rows)} rows -> {OUTPUT_FILE}")
    print(f"🧠 [ProximaScale] Spikes injected at row indices: {cpu_spikes}")
    print(f"🧠 [ProximaScale] CPU range: {min(cpu):.1f}% - {max(cpu):.1f}%")
    print(f"🧠 [ProximaScale] Memory range: {min(mem):.1f}% - {max(mem):.1f}%")
    print(f"🧠 [ProximaScale] Request rate range: {min(req):.1f} - {max(req):.1f}")


if __name__ == "__main__":
    main()