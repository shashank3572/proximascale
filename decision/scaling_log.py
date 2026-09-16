# decision/scaling_log.py
"""
Append-only scaling event log. Consumed by Person B's counterfactual
correction to reason about which scaling decisions were made and when
they "expired".
"""
import json
import os
import threading
from pathlib import Path

_LOCK = threading.Lock()
_LOG_PATH = Path(os.getenv("SCALING_LOG_PATH", "data/scaling_events.json"))


def _ensure_parent() -> None:
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def log_scaling_event(event: dict) -> None:
    """
    Atomically append one event to scaling_events.json.
    File is a JSON array; created on first write.
    Required keys: action, timestamp, expires_after_steps.
    """
    required = {"action", "timestamp", "expires_after_steps"}
    missing = required - event.keys()
    if missing:
        raise ValueError(f"scaling event missing keys: {missing}")

    with _LOCK:
        _ensure_parent()
        if _LOG_PATH.exists() and _LOG_PATH.stat().st_size > 0:
            with _LOG_PATH.open("r") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
        else:
            data = []

        data.append(event)

        tmp = _LOG_PATH.with_suffix(".json.tmp")
        with tmp.open("w") as f:
            json.dump(data, f, indent=2)
        tmp.replace(_LOG_PATH)


def read_scaling_events() -> list:
    if not _LOG_PATH.exists():
        return []
    with _LOG_PATH.open("r") as f:
        return json.load(f)


def clear_scaling_events() -> None:
    with _LOCK:
        _ensure_parent()
        with _LOG_PATH.open("w") as f:
            json.dump([], f)