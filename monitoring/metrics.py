import sqlite3
from pathlib import Path

DB_PATH = Path("data/collected/metrics.db")


def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)


def initialize():
    with _connect() as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS request_counter "
            "(id INTEGER PRIMARY KEY CHECK (id = 1), count INTEGER NOT NULL)"
        )
        connection.execute(
            "INSERT OR IGNORE INTO request_counter (id, count) VALUES (1, 0)"
        )


def increment_request_count():
    initialize()
    with _connect() as connection:
        connection.execute(
            "UPDATE request_counter SET count = count + 1 WHERE id = 1"
        )


def get_request_count():
    initialize()
    with _connect() as connection:
        row = connection.execute(
            "SELECT count FROM request_counter WHERE id = 1"
        ).fetchone()
        return row[0]


def reset_request_count():
    initialize()
    with _connect() as connection:
        row = connection.execute(
            "SELECT count FROM request_counter WHERE id = 1"
        ).fetchone()

        count = row[0]

        connection.execute(
            "UPDATE request_counter SET count = 0 WHERE id = 1"
        )

        return count