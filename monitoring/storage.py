"""
storage.py — CSV persistence layer for MetricRecord objects.
Thread-safe append and read operations.

TWO files, two purposes (do not merge them again):
    metrics.csv       -- FROZEN 90k-row synthetic TRAINING dataset. Prophet,
                         the LSTM and both scalers were fit on it; rerunning
                         training/evaluation on a mutated copy silently
                         invalidates the saved artifacts.
    metrics_live.csv  -- LIVE ring buffer the collector appends to and the
                         real loop reads from. Once it holds >WINDOW_SIZE
                         rows only the newest WINDOW_SIZE rows are ever used.

Mixing the two used to append live idle rows after the synthetic spike
history, so the model's input windows spanned a synthetic 60% CPU row and a
real 0.03% CPU row -- predictions and SHAP attributions from such windows
are meaningless (and idle jitter inside them tripped anomaly scale-ups).
Set PROXIMASCALE_LIVE_METRICS_PATH to relocate the live file.
"""
import csv
import os
import threading
from monitoring.schema import MetricRecord

DATA_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "collected",
)

TRAINING_METRICS_PATH = os.path.join(DATA_DIR, "metrics.csv")
LIVE_METRICS_PATH = os.path.join(DATA_DIR, "metrics_live.csv")

# Kept as the default read/append target for backward compatibility with
# callers/tests that pass an explicit path= argument; production code should
# go through csv_path() (training vs live is resolved per call site).
CSV_PATH = TRAINING_METRICS_PATH


def csv_path(training: bool = False) -> str:
    """Path for the caller's purpose. Training/evaluation always uses the
    frozen metrics.csv; the collector and the real loop use the live file
    (PROXIMASCALE_LIVE_METRICS_PATH overrides its location)."""
    if training:
        return TRAINING_METRICS_PATH
    return os.environ.get("PROXIMASCALE_LIVE_METRICS_PATH") or LIVE_METRICS_PATH

_HEADER = [
    "timestamp",
    "cpu_percent",
    "memory_percent",
    "request_rate",
    "post_scaling"
]

_lock = threading.Lock()


def _ensure_header(path: str):
    """Write CSV header if the file is new or empty."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(_HEADER)


def append_row(record: MetricRecord, path: str = None):
    """Append a single MetricRecord as a CSV row. Thread-safe.

    path=None (the default) appends to the LIVE file. Passing no path must
    never land in the frozen training metrics.csv."""
    path = path or csv_path()
    with _lock:
        _ensure_header(path)
        with open(path, "a", newline="") as f:
            csv.writer(f).writerow([
                record.timestamp,
                record.cpu_percent,
                record.memory_percent,
                record.request_rate,
                record.post_scaling,
            ])


def read_last_n(n: int, path: str = None) -> list:
    """
    Return the last n MetricRecord objects from the CSV.
    Skips the header row automatically.

    path=None (the default) reads the LIVE file -- the real control loop must
    never accidentally ingest the frozen training data. Pass
    csv_path(training=True) to read the frozen 90k training dataset.
    """
    path = path or csv_path()
    with _lock:
        if not os.path.exists(path):
            return []

        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)

    data_rows = [r for r in rows if r and r[0] != "timestamp"]
    tail = data_rows[-n:] if n < len(data_rows) else data_rows

    return [MetricRecord.from_csv_row(r) for r in tail]


def row_count(path: str = None) -> int:
    """Returns number of data rows (excluding header) of the LIVE file."""
    records = read_last_n(999_999, path=path)
    return len(records)