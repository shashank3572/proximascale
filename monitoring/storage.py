"""
storage.py — CSV persistence layer for MetricRecord objects.
Thread-safe append and read operations.
"""
import csv
import os
import threading
from monitoring.schema import MetricRecord

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "collected", "metrics.csv")
_HEADER = ["timestamp", "cpu_percent", "memory_percent", "request_rate"]
_lock = threading.Lock()


def _ensure_header(path: str):
    """Write CSV header if the file is new or empty."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(_HEADER)


def append_row(record: MetricRecord, path: str = CSV_PATH):
    """Append a single MetricRecord as a CSV row. Thread-safe."""
    with _lock:
        _ensure_header(path)
        with open(path, "a", newline="") as f:
            csv.writer(f).writerow([
                record.timestamp,
                record.cpu_percent,
                record.memory_percent,
                record.request_rate,
            ])


def read_last_n(n: int, path: str = CSV_PATH) -> list:
    """
    Return the last n MetricRecord objects from the CSV.
    Skips the header row automatically.
    """
    with _lock:
        if not os.path.exists(path):
            return []
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)

    # rows[0] is header if file was created by append_row
    data_rows = [r for r in rows if r and r[0] != "timestamp"]
    tail = data_rows[-n:] if n < len(data_rows) else data_rows
    return [MetricRecord.from_csv_row(r) for r in tail]


def row_count(path: str = CSV_PATH) -> int:
    """Returns number of data rows (excluding header)."""
    records = read_last_n(999_999, path)
    return len(records)
