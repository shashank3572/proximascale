"""DecisionEngine's scale_down_max_current_cpu guard: no scale-down while the
current reading is still high, however low the forecast; off unless set."""
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


def test_reactive_floor_is_off_by_default():
    assert _engine().evaluate(50.0, current_cpu=90.0) == "hold"
    assert _engine(scale_up_on_current_cpu=True).evaluate(50.0) == "hold"   # no reading given
