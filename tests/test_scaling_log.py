import json
import os
import pytest
from pathlib import Path
import decision.scaling_log as slog


@pytest.fixture(autouse=True)
def tmp_log(tmp_path, monkeypatch):
    p = tmp_path / "scaling_events.json"
    monkeypatch.setattr(slog, "_LOG_PATH", p)
    yield p


def test_append_creates_file_with_array(tmp_log):
    slog.log_scaling_event({"action": "scale_up", "timestamp": 1.0,
                            "expires_after_steps": 60})
    data = json.loads(tmp_log.read_text())
    assert isinstance(data, list) and len(data) == 1
    assert data[0]["action"] == "scale_up"


def test_multiple_appends_accumulate(tmp_log):
    for i in range(3):
        slog.log_scaling_event({"action": "scale_down", "timestamp": float(i),
                                "expires_after_steps": 60})
    assert len(json.loads(tmp_log.read_text())) == 3


def test_missing_required_key_raises(tmp_log):
    with pytest.raises(ValueError):
        slog.log_scaling_event({"action": "scale_up"})