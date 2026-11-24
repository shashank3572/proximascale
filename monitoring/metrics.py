"""Thread-safe request counter for the workload app.

The counter is the source of the `request_rate` metric the collector stores.
Two properties matter and both were broken before (audit C4):

1. ATOMICITY. Read-and-reset used to be two statements on two connections
   with no transaction, so a request arriving between them was either lost or
   double counted. `reset_request_count()` now does both inside one
   BEGIN IMMEDIATE transaction, and `increment_request_count()` is a single
   atomic UPDATE.

2. WINDOW TRACKING. The counter alone cannot say over what period it
   accumulated, so the collector could not distinguish "12 requests in 30 s"
   from "12 requests in 90 s" after a timeout. The module now records when the
   current window started and reports its length alongside the count.

Definitions (documented once, used everywhere -- see docs/METRICS.md):
    request_count         cumulative requests since the last reset
                          (monotonic until reset; survives until the app
                          process restarts, because it lives in SQLite)
    window_seconds        wall-clock seconds since the previous reset
    requests_per_second   request_count / window_seconds
    request_rate (CSV)    request_count for exactly one collector poll window
                          -- requests per POLL, not per second
"""
import os
import sqlite3
import threading
import time
from pathlib import Path

DB_PATH = Path(
    os.environ.get("PROXIMASCALE_METRICS_DB")
    or Path(__file__).resolve().parent.parent / "data" / "collected" / "metrics.db"
)

# Serialises read-modify-write cycles between request threads.
_LOCK = threading.RLock()
_initialised = False
_window_started = None
_last_window_seconds = None


def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    # isolation_level=None => autocommit; explicit BEGIN/COMMIT where needed.
    conn = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def initialize(conn=None):
    global _initialised
    if _initialised:
        return
    owned = conn is None
    conn = conn or _connect()
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS request_counter "
            "(id INTEGER PRIMARY KEY CHECK (id = 1), count INTEGER NOT NULL)"
        )
        conn.execute(
            "INSERT OR IGNORE INTO request_counter (id, count) VALUES (1, 0)"
        )
        _initialised = True
    finally:
        if owned:
            conn.close()


def increment_request_count():
    global _window_started
    with _LOCK:
        conn = _connect()
        try:
            initialize(conn)
            conn.execute("UPDATE request_counter SET count = count + 1 WHERE id = 1")
        finally:
            conn.close()
        if _window_started is None:
            _window_started = time.time()


def get_request_count():
    with _LOCK:
        conn = _connect()
        try:
            initialize(conn)
            row = conn.execute(
                "SELECT count FROM request_counter WHERE id = 1"
            ).fetchone()
            return int(row[0]) if row else 0
        finally:
            conn.close()


def reset_request_count_window():
    """Atomically read-and-reset the counter. Returns
    {"request_count", "window_seconds", "requests_per_second"}.

    `window_seconds` is measured on the app side, so it reflects the real
    interval the count covers even when the collector's poll was delayed or
    retried. It is None on the very first reset after a process start (no
    previous window to measure).
    """
    global _window_started, _last_window_seconds
    now = time.time()
    with _LOCK:
        conn = _connect()
        try:
            initialize(conn)
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT count FROM request_counter WHERE id = 1"
            ).fetchone()
            count = int(row[0]) if row else 0
            conn.execute("UPDATE request_counter SET count = 0 WHERE id = 1")
            conn.execute("COMMIT")
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            conn.close()

        window = None if _window_started is None else max(0.0, now - _window_started)
        _window_started = now
        _last_window_seconds = window
        return {
            "request_count": count,
            "window_seconds": None if window is None else round(window, 3),
            "requests_per_second":
                None if not window else round(count / window, 3),
        }


def reset_request_count():
    """Backward-compatible read-and-reset: returns just the count."""
    return reset_request_count_window()["request_count"]
