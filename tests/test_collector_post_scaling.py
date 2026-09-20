"""Tests for monitoring.collector.is_post_scaling (array-format scaling log)."""
import json
import time

import pytest

import decision.scaling_log as slog
import monitoring.collector as collector


@pytest.fixture(autouse=True)
def tmp_log(tmp_path, monkeypatch):
    p = tmp_path / "scaling_events.json"
    monkeypatch.setattr(slog, "_LOG_PATH", p)
    monkeypatch.setattr(collector, "_tick_seconds", lambda default=3.0: 3.0)
    return p


def _write(path, events):
    path.write_text(json.dumps(events))


def test_import_does_not_need_docker_daemon():
    assert collector._docker_client is None


def test_missing_file_is_false():
    assert collector.is_post_scaling() is False


def test_empty_array_is_false(tmp_log):
    _write(tmp_log, [])
    assert collector.is_post_scaling() is False


def test_fresh_event_is_true(tmp_log):
    _write(tmp_log, [{"action": "scale_up", "timestamp": time.time(),
                      "expires_after_steps": 60}])
    assert collector.is_post_scaling() is True


def test_expired_event_is_false(tmp_log):
    # window = 60 * 3s = 180s; event is 200s old
    _write(tmp_log, [{"action": "scale_up", "timestamp": time.time() - 200,
                      "expires_after_steps": 60}])
    assert collector.is_post_scaling() is False


def test_window_is_seconds_not_minutes(tmp_log):
    # Regression: old code used steps*60 -> 1h window. 10 min old must be False.
    _write(tmp_log, [{"action": "scale_down", "timestamp": time.time() - 600,
                      "expires_after_steps": 60}])
    assert collector.is_post_scaling() is False


def test_uses_latest_event(tmp_log):
    now = time.time()
    _write(tmp_log, [
        {"action": "scale_up", "timestamp": now - 1000, "expires_after_steps": 60},
        {"action": "scale_down", "timestamp": now - 5, "expires_after_steps": 60},
    ])
    assert collector.is_post_scaling() is True


def test_malformed_entry_is_false(tmp_log):
    _write(tmp_log, [{"action": "scale_up"}])
    assert collector.is_post_scaling() is False


def test_corrupt_json_is_false(tmp_log):
    tmp_log.write_text("{not json")
    assert collector.is_post_scaling() is False
