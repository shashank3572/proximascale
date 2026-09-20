"""Tests for main.run_real_loop wiring (storage -> predict_load -> engine)."""
import sys
import types
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

import main
from monitoring.schema import MetricRecord


class _StopLoop(Exception):
    pass


def _records(n=10, age_seconds=0):
    newest = datetime.now() - timedelta(seconds=age_seconds)
    return [
        MetricRecord(
            timestamp=(newest - timedelta(seconds=30 * (n - 1 - i))).isoformat(),
            cpu_percent=50.0, memory_percent=40.0, request_rate=100,
            post_scaling=False,
        )
        for i in range(n)
    ]


@pytest.fixture
def harness(monkeypatch, tmp_path):
    """Patch storage.read_last_n, model.predict.predict_load, time.sleep."""
    fake_predict = types.ModuleType("model.predict")
    fake_predict.predict_load = MagicMock(return_value=(60.0, 70.0, False))
    monkeypatch.setitem(sys.modules, "model.predict", fake_predict)

    import monitoring.storage as storage
    reader = MagicMock()
    monkeypatch.setattr(storage, "read_last_n", reader)

    def _stop(_):
        raise _StopLoop
    monkeypatch.setattr(main.time, "sleep", _stop)

    events = tmp_path / "events.csv"
    monkeypatch.setattr(main, "EVENTS_PATH", str(events))

    engine = MagicMock()
    engine.evaluate.return_value = "hold"
    engine.last_reason = "in_band"
    engine.actuator.replica_count.return_value = 2
    return types.SimpleNamespace(predict=fake_predict.predict_load,
                                 reader=reader, engine=engine, events=events)


def test_fresh_window_calls_predict_and_engine(harness):
    harness.reader.return_value = _records()
    with pytest.raises(_StopLoop):
        main.run_real_loop(harness.engine, poll_interval=30)
    harness.reader.assert_called_once_with(10)
    harness.predict.assert_called_once()
    harness.engine.evaluate.assert_called_once_with(60.0, 70.0, anomaly_flag=False)


def test_short_window_skips_prediction(harness):
    harness.reader.return_value = _records(n=4)
    with pytest.raises(_StopLoop):
        main.run_real_loop(harness.engine, poll_interval=30)
    harness.predict.assert_not_called()
    harness.engine.evaluate.assert_not_called()


def test_stale_data_skips_prediction(harness):
    harness.reader.return_value = _records(age_seconds=600)
    with pytest.raises(_StopLoop):
        main.run_real_loop(harness.engine, poll_interval=30)
    harness.predict.assert_not_called()
    harness.engine.evaluate.assert_not_called()


def test_unparseable_timestamp_is_stale():
    rec = _records()[-1]
    rec.timestamp = "not-a-date"
    assert main._is_stale(rec, 90) is True


def test_loop_writes_dashboard_event_row(harness):
    import csv
    harness.reader.return_value = _records()
    with pytest.raises(_StopLoop):
        main.run_real_loop(harness.engine, poll_interval=30)
    rows = list(csv.DictReader(open(harness.events)))
    assert len(rows) == 1
    r = rows[0]
    assert list(r.keys()) == main.EVENT_COLUMNS
    assert r["signal"] == "hold" and r["replicas"] == "2"
    assert float(r["actual_cpu"]) == 50.0 and float(r["predicted_cpu"]) == 60.0
    assert r["anomaly_flag"] == "False"


def test_skipped_cycles_write_no_event(harness):
    harness.reader.return_value = _records(age_seconds=600)
    with pytest.raises(_StopLoop):
        main.run_real_loop(harness.engine, poll_interval=30)
    assert not harness.events.exists()


def test_log_event_never_raises(tmp_path):
    bad = tmp_path / "file"
    bad.write_text("x")
    main.log_event({"signal": "hold"}, path=str(bad / "sub" / "events.csv"))  # parent is a file


def test_event_columns_match_dashboard():
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from dashboard.live_plot import generate_demo
    assert set(main.EVENT_COLUMNS) == set(generate_demo().columns)
