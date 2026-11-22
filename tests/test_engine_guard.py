"""DecisionEngine's scale_down_max_current_cpu guard: no scale-down while the
current reading is still high, however low the forecast; off unless set.

The anomaly minimum-interval gate lives here too: the anomaly branch bypasses
cooldown, so one sustained spike used to add a worker on every poll (a demo run
fired two anomaly scale-ups 34 s apart).
"""
import time

from decision.engine import DecisionEngine


class Counter:
    def __init__(self):
        self.replicas = 3

    def scale_up(self):
        self.replicas += 1
        return True

    def scale_down(self):
        self.replicas -= 1
        return True


def _engine(**cfg):
    return DecisionEngine(config={"cooldown_seconds": 0, "adaptive_min_samples": 10 ** 9, **cfg},
                          actuator=Counter())


def test_guard_blocks_scale_down_while_cpu_is_high():
    engine = _engine(scale_down_max_current_cpu=60.0)
    assert engine.evaluate(20.0, current_cpu=90.0) == "hold"
    assert engine.last_reason == "scale_down_guard"
    assert engine.actuator.replicas == 3
    assert engine.evaluate(20.0, current_cpu=40.0) == "scale_down"


def test_guard_is_off_by_default_and_without_current_cpu():
    assert _engine().evaluate(20.0, current_cpu=90.0) == "scale_down"
    assert _engine(scale_down_max_current_cpu=60.0).evaluate(20.0) == "scale_down"


def test_branch_switches():
    on = _engine()
    assert on.evaluate(50.0, upper_bound=99.0) == "scale_up"
    off = _engine(use_upper_bound_risk=False, use_anomaly_bypass=False)
    assert off.evaluate(50.0, upper_bound=99.0) == "hold"
    assert off.evaluate(50.0, anomaly_flag=True) == "hold"
    assert _engine().evaluate(50.0, anomaly_flag=True) == "scale_up"


def test_guard_never_blocks_scale_up():
    engine = _engine(scale_down_max_current_cpu=60.0)
    assert engine.evaluate(95.0, current_cpu=90.0) == "scale_up"


def test_reactive_floor_scales_up_on_current_cpu():
    engine = _engine(scale_up_on_current_cpu=True)
    assert engine.evaluate(50.0, current_cpu=90.0) == "scale_up"
    assert engine.last_reason == "current_cpu_high"
    assert engine.evaluate(50.0, current_cpu=60.0) == "hold"       # in band, below threshold


def test_anomaly_scale_up_respects_min_interval():
    """One sustained spike must add one worker, not one per poll."""
    engine = _engine(anomaly_scale_up_min_interval=120)
    assert engine.evaluate(50.0, anomaly_flag=True) == "scale_up"
    assert engine.last_reason == "anomaly"
    assert engine.actuator.replicas == 4

    # Same spike, next poll: anomaly flag is still set, no second worker.
    assert engine.evaluate(50.0, anomaly_flag=True) == "hold"
    assert engine.last_reason == "anomaly_min_interval"
    assert engine.actuator.replicas == 4


def test_anomaly_scale_up_fires_again_after_min_interval():
    engine = _engine(anomaly_scale_up_min_interval=120)
    engine.evaluate(50.0, anomaly_flag=True)
    engine.last_scale_up_time = time.time() - 121

    assert engine.evaluate(50.0, anomaly_flag=True) == "scale_up"
    assert engine.last_reason == "anomaly"
    assert engine.actuator.replicas == 5


def test_anomaly_min_interval_zero_disables_the_gate():
    engine = _engine(anomaly_scale_up_min_interval=0)
    assert engine.evaluate(50.0, anomaly_flag=True) == "scale_up"
    assert engine.evaluate(50.0, anomaly_flag=True) == "scale_up"
    assert engine.last_reason == "anomaly"


def test_anomaly_interval_counts_from_a_cpu_high_scale_up():
    """Regression: cpu_high adds a worker, then an anomaly 30 s later used to
    add a second one inside the window; the interval must span both branches."""
    engine = _engine(anomaly_scale_up_min_interval=120)
    assert engine.evaluate(95.0, anomaly_flag=False) == "scale_up"
    assert engine.last_reason == "cpu_high"
    assert engine.actuator.replicas == 4

    assert engine.evaluate(95.0, anomaly_flag=True) == "hold"
    assert engine.last_reason == "anomaly_min_interval"
    assert engine.actuator.replicas == 4

    # Once the window from the cpu_high scale-up has passed, anomaly may fire.
    engine.last_scale_up_time = time.time() - 121
    assert engine.evaluate(95.0, anomaly_flag=True) == "scale_up"
    assert engine.last_reason == "anomaly"
    assert engine.actuator.replicas == 5


def test_scale_down_does_not_arm_the_anomaly_interval():
    """Only scale_ups arm the window; a scale_down then an anomaly is allowed."""
    engine = _engine(anomaly_scale_up_min_interval=120)
    engine.actuator.replicas = 5
    engine.evaluate(10.0, anomaly_flag=False)
    assert engine.last_reason == "cpu_low"
    assert engine.evaluate(50.0, anomaly_flag=True) == "scale_up"
    assert engine.last_reason == "anomaly"


def test_anomaly_min_interval_is_on_by_default_and_separate_from_cooldown():
    # _engine() has cooldown_seconds=0, so the gate below cannot be the
    # normal cooldown doing the work.
    engine = _engine()
    assert engine.anomaly_scale_up_min_interval > 0
    assert engine.evaluate(50.0, anomaly_flag=True) == "scale_up"
    assert engine.evaluate(50.0, anomaly_flag=True) == "hold"
    assert engine.last_reason == "anomaly_min_interval"


def test_anomaly_gate_does_not_block_normal_branches():
    engine = _engine(anomaly_scale_up_min_interval=120)
    engine.evaluate(50.0, anomaly_flag=True)
    assert engine.evaluate(50.0, anomaly_flag=True) == "hold"   # gated
    assert engine.evaluate(95.0, anomaly_flag=False) == "scale_up"
    assert engine.last_reason == "cpu_high"


def test_reactive_floor_is_off_by_default():
    assert _engine().evaluate(50.0, current_cpu=90.0) == "hold"
    assert _engine(scale_up_on_current_cpu=True).evaluate(50.0) == "hold"   # no reading given
