"""Tests for the audit remediation (H1-H8, C1-C5, M1-M3, M6).

Deliberately dependency-light: everything except the predict_load tests runs
without TensorFlow/Prophet, so CI can execute this suite in seconds.
"""
import threading
import time

import numpy as np
import pytest

import monitoring.collector as collector
import monitoring.metrics as metrics
from decision.adaptive_threshold import AdaptiveThreshold
from decision.engine import DecisionEngine


# ── H1: fallback contract ────────────────────────────────────────────────────
def test_predict_load_fallback_reports_no_anomaly():
    pytest.importorskip("tensorflow")
    pytest.importorskip("prophet")
    from model.predict import predict_load

    records = [{"timestamp": f"2026-01-01T00:00:{i:02d}", "cpu_percent": 50.0,
                "memory_percent": 40.0, "request_rate": 10,
                "post_scaling": False} for i in range(4)]
    predicted, upper, anomaly = predict_load(records)
    assert anomaly is False, "a degraded forecast must not claim a detection"
    assert 0.0 <= predicted <= 100.0 and 0.0 <= upper <= 100.0
    assert isinstance(predicted, float) and isinstance(upper, float)
    assert type(anomaly) is bool


# ── C2/C5: delta calibration ─────────────────────────────────────────────────
def test_delta_calibration_moves_absolute_forecast_onto_live_level(monkeypatch):
    import model.predict as P

    monkeypatch.delenv("PROXIMASCALE_LIVE_CALIBRATION", raising=False)
    P.reset_live_calibration()

    # The model believes "now" is 52% (synthetic baseline); the container idles.
    predicted, upper, offset = P._apply_live_calibration(55.0, 60.0, 52.0, 0.5)
    assert offset == pytest.approx(-51.5)
    assert predicted == pytest.approx(3.5)
    assert upper == pytest.approx(8.5)


def test_delta_calibration_has_a_jitter_deadband(monkeypatch):
    import model.predict as P

    monkeypatch.delenv("PROXIMASCALE_LIVE_CALIBRATION", raising=False)
    P.reset_live_calibration()

    P._apply_live_calibration(55.0, 60.0, 52.0, 10.0)        # offset -42
    _, _, offset = P._apply_live_calibration(55.0, 60.0, 52.0, 10.4)
    assert offset == pytest.approx(-42.0), "sub-1pp jitter must not shift the band"
    # ...but a real workload change applies immediately.
    _, _, offset = P._apply_live_calibration(55.0, 60.0, 52.0, 95.0)
    assert offset == pytest.approx(43.0)


def test_calibration_can_be_switched_off(monkeypatch):
    import model.predict as P

    monkeypatch.setenv("PROXIMASCALE_LIVE_CALIBRATION", "off")
    predicted, upper, offset = P._apply_live_calibration(55.0, 60.0, 52.0, 0.5)
    assert (predicted, upper, offset) == (55.0, 60.0, 0.0)


# ── C3: collector resilience ─────────────────────────────────────────────────
def test_cpu_fields_missing_returns_none_not_zero():
    """A missing measurement must be distinguishable from 0% CPU usage."""
    assert collector._cpu_percent_from_stats({"cpu_stats": {}}, {}) is None
    half = {"cpu_stats": {"cpu_usage": {"total_usage": 5}, "system_cpu_usage": 10}}
    assert collector._cpu_percent_from_stats(half, {"cpu_stats": {}}) is None


def test_zero_cpu_is_still_zero():
    a = {"cpu_stats": {"cpu_usage": {"total_usage": 0}, "system_cpu_usage": 100}}
    b = {"cpu_stats": {"cpu_usage": {"total_usage": 0}, "system_cpu_usage": 100}}
    assert collector._cpu_percent_from_stats(a, b) == 0.0


def test_stopped_container_raises_measurement_unavailable(monkeypatch):
    class Stopped:
        attrs = {"State": {"Status": "exited", "ExitCode": 137}}

        def reload(self):
            pass

    class FakeClient:
        class containers:
            @staticmethod
            def get(name):
                return Stopped()

    monkeypatch.setattr(collector, "_get_docker_client", lambda: FakeClient())
    with pytest.raises(collector.MeasurementUnavailable) as excinfo:
        collector._get_container()
    assert "not running" in str(excinfo.value)


def test_missing_container_raises_measurement_unavailable(monkeypatch):
    import docker

    class FakeClient:
        class containers:
            @staticmethod
            def get(name):
                raise docker.errors.NotFound("no such container")

    monkeypatch.setattr(collector, "_get_docker_client", lambda: FakeClient())
    with pytest.raises(collector.MeasurementUnavailable) as excinfo:
        collector._get_container()
    assert "does not exist" in str(excinfo.value)


def test_docker_daemon_error_does_not_propagate(monkeypatch):
    import docker

    def boom():
        raise docker.errors.DockerException("daemon down")

    monkeypatch.setattr(collector, "_get_docker_client", boom)
    with pytest.raises(collector.MeasurementUnavailable):
        collector._get_container()


def test_failed_cycle_writes_no_row_and_loop_survives(monkeypatch):
    """A stopped container must not crash the collector NOR write a fake row."""
    written = []
    monkeypatch.setattr(
        collector, "get_container_cpu_percent",
        lambda: (_ for _ in ()).throw(collector.MeasurementUnavailable("state=exited")))
    monkeypatch.setattr(collector, "append_row",
                        lambda record, path=None: written.append(record))

    collector.collect_metrics(max_cycles=2)
    assert written == [], "no measurement -> no row"


def test_backoff_is_bounded_and_never_negative():
    assert collector._next_sleep(True, 100.0, 0) == 0.0
    assert collector._next_sleep(True, 5.0, 0) == pytest.approx(
        collector.POLL_INTERVAL - 5.0)
    assert collector._next_sleep(False, 0.0, 10) <= collector.POLL_INTERVAL
    assert collector._next_sleep(False, 0.0, 1) >= collector._next_sleep(False, 0.0, 0)


# ── C4/H3: request counter ───────────────────────────────────────────────────
@pytest.fixture
def fresh_counter(monkeypatch, tmp_path):
    monkeypatch.setattr(metrics, "DB_PATH", tmp_path / "metrics.db")
    monkeypatch.setattr(metrics, "_initialised", False)
    monkeypatch.setattr(metrics, "_window_started", None)
    yield
    monkeypatch.setattr(metrics, "_initialised", False)
    monkeypatch.setattr(metrics, "_window_started", None)


def test_reset_returns_count_and_window(fresh_counter):
    started = time.time()
    for _ in range(3):
        metrics.increment_request_count()
    first = metrics.reset_request_count_window()
    assert first["request_count"] == 3
    # The window starts at the first observed request, so the first reset
    # already has a measured (very short) window rather than None.
    assert first["window_seconds"] is not None
    assert 0 <= first["window_seconds"] <= time.time() - started + 1

    time.sleep(0.02)
    metrics.increment_request_count()
    second = metrics.reset_request_count_window()
    assert second["request_count"] == 1
    assert second["window_seconds"] > 0
    assert second["requests_per_second"] is not None


def test_reset_is_atomic_under_concurrency(fresh_counter):
    """Every increment must be counted exactly once, whatever the thread
    interleaving (the old read-then-reset on separate connections lost races)."""
    def worker():
        for _ in range(25):
            metrics.increment_request_count()

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert metrics.get_request_count() == 100
    assert metrics.reset_request_count_window()["request_count"] == 100
    assert metrics.get_request_count() == 0


# ── C1/M1: adaptive band ─────────────────────────────────────────────────────
@pytest.mark.parametrize("level", [0.0, 5.0, 30.0, 50.0, 60.0, 75.0, 90.0, 100.0])
def test_band_is_always_valid(level):
    t = AdaptiveThreshold(min_samples=3)
    for _ in range(10):
        t.update(level)
    upper, lower = t.get_thresholds()
    assert 0.0 <= lower < upper <= 100.0
    assert upper - lower >= t.min_band - 1e-9


def test_idle_predictions_are_below_the_lower_threshold():
    """C1: with the old band (lower pinned at 45, forecasts stuck at ~52)
    scale-down was unreachable. An idle forecast must fall below the lower
    threshold now."""
    t = AdaptiveThreshold(window_size=30, min_samples=5, static_upper=75.0,
                          static_lower=30.0, floor_lower=15.0, ceil_lower=45.0)
    for _ in range(10):
        t.update(0.4)
    _, lower = t.get_thresholds()
    assert 0.4 < lower


def test_busy_band_still_triggers_scale_up():
    t = AdaptiveThreshold(window_size=30, min_samples=5)
    for _ in range(10):
        t.update(99.0)
    upper, _ = t.get_thresholds()
    assert 100.0 > upper, "a pegged container must still cross the trigger"


def test_nan_and_inf_are_ignored():
    t = AdaptiveThreshold(min_samples=2)
    t.update(float("nan"))
    t.update(float("inf"))
    t.update(10.0)
    assert len(t.window) == 1


def test_inverted_config_limits_are_repaired():
    t = AdaptiveThreshold(floor_upper=90.0, ceil_upper=40.0,
                          floor_lower=60.0, ceil_lower=20.0, min_band=10.0)
    for _ in range(20):
        t.update(50.0)
    upper, lower = t.get_thresholds()
    assert lower < upper


# ── engine behaviour ────────────────────────────────────────────────────────
def _engine(**overrides):
    cfg = {
        "min_containers": 0, "max_containers": 5, "cooldown_seconds": 0,
        "cpu_upper_threshold": 75.0, "cpu_lower_threshold": 30.0,
        "adaptive_min_samples": 1, "adaptive_window": 5,
        "use_anomaly_bypass": False, "use_upper_bound_risk": False,
        "scale_down_max_current_cpu": 75.0,
    }
    cfg.update(overrides)

    class Actuator:
        def __init__(self):
            self.calls = []
            self.n = 1

        def scale_up(self):
            self.calls.append("scale_up")
            return True

        def scale_down(self):
            self.calls.append("scale_down")
            self.n -= 1
            return True

        def replica_count(self):
            return self.n

    return DecisionEngine(config=cfg, actuator=Actuator())


def test_engine_can_actually_scale_down_on_idle():
    eng = _engine()
    signal = None
    for _ in range(6):
        signal = eng.evaluate(0.4, 0.9, anomaly_flag=False, current_cpu=0.4)
    assert signal == "scale_down"
    assert eng.last_reason == "cpu_low"


def test_engine_scales_up_on_high_load():
    eng = _engine()
    signal = eng.evaluate(97.0, 100.0, anomaly_flag=False, current_cpu=97.0)
    assert signal == "scale_up"
    assert eng.last_reason in {"cpu_high", "current_cpu_high"}


def test_engine_exposes_effective_thresholds():
    eng = _engine()
    eng.evaluate(50.0, 55.0, anomaly_flag=False, current_cpu=50.0)
    assert eng.last_thresholds["upper"] is not None
    assert eng.last_thresholds["lower"] < eng.last_thresholds["upper"]
    assert eng.last_thresholds["mode"] in {"static", "adaptive"}


def test_engine_rejects_inverted_thresholds():
    with pytest.raises(ValueError):
        _engine(cpu_upper_threshold=20.0, cpu_lower_threshold=80.0)


def test_engine_rejects_negative_cooldown():
    with pytest.raises(ValueError):
        _engine(cooldown_seconds=-1)
