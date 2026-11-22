import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "collected" / "metrics.db"


@contextmanager
def _connect():
    """Yield a connection and always close it.

    `with sqlite3.connect(...) as connection:` only commits/rolls back -- it
    does NOT close the connection, so every counter call leaked a handle (and
    on Windows left the metrics.db file locked for other processes).
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)
    try:
        yield connection
    finally:
        connection.close()


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