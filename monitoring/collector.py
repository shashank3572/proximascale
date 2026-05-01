"""
collector.py — Polls CPU%, memory%, and request_rate every POLL_INTERVAL seconds.
Stores each reading via storage.append_row().

Usage (standalone):
    python -m monitoring.collector

Usage (from code):
    from monitoring.collector import collect_metrics
    records = collect_metrics(window=50)   # returns last 50 records as list of dicts
"""
import time
import psutil
import requests
from datetime import datetime

from monitoring.schema import MetricRecord
from monitoring import storage

POLL_INTERVAL = 10       # seconds between each sample
import os
APP_METRICS_URL = os.environ.get("APP_METRICS_URL", "http://localhost:5000/metrics")


def _fetch_request_rate() -> int:
    """
    Calls the Flask /metrics endpoint to get request_rate.
    Returns 0 if the app is unreachable (collector can start before the app).
    """
    try:
        resp = requests.get(APP_METRICS_URL, timeout=3)
        resp.raise_for_status()
        return int(resp.json().get("request_rate", 0))
    except Exception:
        return 0


def collect_metrics(window: int = None) -> list:
    """
    Returns the last `window` MetricRecord dicts from storage.
    If window is None, returns ALL stored records.
    This is the function Person B's code can call for the latest data.
    """
    n = window if window is not None else 999_999
    records = storage.read_last_n(n)
    return [r.to_dict() for r in records]


def run_collector():
    """
    Infinite polling loop. Call this to actively collect + persist metrics.
    Runs forever — launch in a thread or as a standalone process.
    """
    print(f"[collector] Starting. Polling every {POLL_INTERVAL}s → {storage.CSV_PATH}")
    while True:
        cpu    = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory().percent
        rate   = _fetch_request_rate()
        ts     = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

        record = MetricRecord(
            timestamp=ts,
            cpu_percent=round(cpu, 2),
            memory_percent=round(memory, 2),
            request_rate=rate,
        )
        storage.append_row(record)
        print(f"[collector] {ts} | CPU: {cpu}% | MEM: {memory}% | REQ/min: {rate}")

        time.sleep(POLL_INTERVAL - 1)   # -1 because cpu_percent(interval=1) already took 1s


if __name__ == "__main__":
    run_collector()
