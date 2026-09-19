"""
test_monitoring.py — Tests for Person A's monitoring module.
Person D owns this file.

Covers:
    - MetricRecord schema validation
    - collect_metrics() return format
    - storage append / read round-trip
"""
import pytest
import os
import tempfile
from unittest.mock import patch
from monitoring.schema import MetricRecord
import monitoring.storage as storage


# ── MetricRecord schema ───────────────────────────────────────────────────────

def test_metric_record_fields():
    """MetricRecord must contain all 4 required fields."""
    record = MetricRecord(
        timestamp="2024-01-15T14:32:00",
        cpu_percent=67.4,
        memory_percent=52.1,
        request_rate=143
    )
    assert record.timestamp == "2024-01-15T14:32:00"
    assert record.cpu_percent == 67.4
    assert record.memory_percent == 52.1
    assert record.request_rate == 143


def test_metric_record_to_dict():
    """to_dict() must return all 4 keys with correct types."""
    record = MetricRecord(
        timestamp="2024-01-15T14:32:00",
        cpu_percent=67.4,
        memory_percent=52.1,
        request_rate=143
    )
    d = record.to_dict()
    assert set(d.keys()) == {"timestamp", "cpu_percent", "memory_percent", "request_rate"}
    assert isinstance(d["cpu_percent"], float)
    assert isinstance(d["request_rate"], int)


def test_metric_record_from_csv_row():
    """from_csv_row() must parse a list of strings into a MetricRecord."""
    row = ["2024-01-15T14:32:00", "67.4", "52.1", "143"]
    record = MetricRecord.from_csv_row(row)
    assert record.cpu_percent == 67.4
    assert record.request_rate == 143


def test_metric_record_from_dict():
    """from_dict() must reconstruct a MetricRecord from a dict."""
    d = {
        "timestamp": "2024-01-15T14:32:00",
        "cpu_percent": 67.4,
        "memory_percent": 52.1,
        "request_rate": 143
    }
    record = MetricRecord.from_dict(d)
    assert record.cpu_percent == 67.4
    assert record.memory_percent == 52.1


def test_metric_record_round_trip():
    """to_dict() → from_dict() must produce identical values."""
    original = MetricRecord(
        timestamp="2024-01-15T14:32:00",
        cpu_percent=55.0,
        memory_percent=40.0,
        request_rate=200
    )
    recovered = MetricRecord.from_dict(original.to_dict())
    assert recovered.cpu_percent == original.cpu_percent
    assert recovered.request_rate == original.request_rate


# ── Storage round-trip ────────────────────────────────────────────────────────

def test_storage_append_and_read():
    """Append a record to CSV then read it back — values must match."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode='w') as f:
        tmp_path = f.name

    try:
        record = MetricRecord(
            timestamp="2024-01-15T14:32:00",
            cpu_percent=72.1,
            memory_percent=54.3,
            request_rate=201
        )
        storage.append_row(record, path=tmp_path)
        rows = storage.read_last_n(1, path=tmp_path)
        assert len(rows) == 1
        assert rows[0].cpu_percent == 72.1
        assert rows[0].request_rate == 201
    finally:
        os.unlink(tmp_path)


def test_storage_read_last_n():
    """read_last_n(3) must return exactly 3 most recent rows."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode='w') as f:
        tmp_path = f.name

    try:
        for i in range(5):
            storage.append_row(MetricRecord(
                timestamp=f"2024-01-15T14:3{i}:00",
                cpu_percent=float(10 + i),
                memory_percent=50.0,
                request_rate=100 + i
            ), path=tmp_path)
        rows = storage.read_last_n(3, path=tmp_path)
        assert len(rows) == 3
        assert rows[-1].cpu_percent == 14.0
    finally:
        os.unlink(tmp_path)