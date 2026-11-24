# decision/adaptive_threshold.py
"""Adaptive CPU band for the decision engine.

The band is a rolling mean +/- k standard deviations of the recent forecast
series, clamped into an operator-controlled window:

    upper = clamp(mu + k*sigma, floor_upper, ceil_upper)
    lower = clamp(mu - k*sigma, floor_lower, ceil_lower)

Both bounds are then kept inside the physical 0-100 CPU% range and separated
by at least `min_band` percentage points so the band can never invert (a lower
bound above the upper bound would make scale_up and scale_down fire on the
same reading).

Why the clamps exist:
  * floor_upper stops a quiet period from dragging the scale-up trigger so low
    that noise crosses it continuously (the "scale-up storm").
  * ceil_lower/floor_lower stop a busy period from dragging the scale-down
    trigger so high that idle can never reach it (audit finding C1: with the
    old hardcoded ceil_lower=45 and a forecast series stuck at ~52, `lower`
    converged to 45 and a 49-57 prediction series never dipped below it).

Every limit is configurable from config.yaml (`adaptive_floor_upper`,
`adaptive_ceil_upper`, `adaptive_floor_lower`, `adaptive_ceil_lower`,
`adaptive_min_band`). The defaults below are fallbacks for a config that omits
them; the effective values are reported by describe()/last_state and logged
with every decision.

Mathematical relationship to the other inputs (documented contract):
    static thresholds    operator intent (cpu_upper_threshold /
                         cpu_lower_threshold); used verbatim until
                         `min_samples` observations exist.
    adaptive thresholds  the clamps applied to the rolling forecast statistics
                         above -- never a substitute for the measured CPU.
    forecast mean        the live-calibrated predicted CPU (model/predict.py),
                         i.e. current CPU + forecast delta, so it is on the
                         same scale as the measured CPU.
    forecast uncertainty the LSTM's MC-Dropout sigma; only used by the engine's
                         optional upper_bound branch, never here.
    actual CPU           the reference the forecast is calibrated against; it
                         does not enter the band directly.
"""
from collections import deque
import math
import statistics

CPU_MIN = 0.0
CPU_MAX = 100.0


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class AdaptiveThreshold:
    def __init__(
        self,
        window_size: int = 30,
        k: float = 1.5,
        min_samples: int = 10,
        static_upper: float = 75.0,
        static_lower: float = 30.0,
        floor_upper: float = 60.0,
        ceil_upper: float = 90.0,
        floor_lower: float = None,       # None -> static_lower (configurable)
        ceil_lower: float = 45.0,
        min_band: float = 10.0,
    ):
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if min_samples < 1:
            raise ValueError("min_samples must be >= 1")
        if k < 0:
            raise ValueError("k must be >= 0")

        self.window = deque(maxlen=int(window_size))
        self.k = float(k)
        self.min_samples = int(min_samples)
        self.static_upper = _clamp(float(static_upper), CPU_MIN, CPU_MAX)
        self.static_lower = _clamp(float(static_lower), CPU_MIN, CPU_MAX)

        self.floor_upper = _clamp(float(floor_upper), CPU_MIN, CPU_MAX)
        self.ceil_upper = _clamp(float(ceil_upper), CPU_MIN, CPU_MAX)
        self.floor_lower = _clamp(
            float(static_lower if floor_lower is None else floor_lower), CPU_MIN, CPU_MAX
        )
        self.ceil_lower = _clamp(float(ceil_lower), CPU_MIN, CPU_MAX)
        self.min_band = max(0.0, float(min_band))

        self._normalise_limits()

        self.last_state = {
            "mode": "static",
            "mu": None,
            "sigma": None,
            "upper": self.static_upper,
            "lower": self.static_lower,
            "samples": 0,
        }

    # ------------------------------------------------------------------ limits
    def _normalise_limits(self) -> None:
        """Repair limits that contradict each other instead of trusting config."""
        if self.floor_upper > self.ceil_upper:
            self.floor_upper, self.ceil_upper = self.ceil_upper, self.floor_upper
        if self.floor_lower > self.ceil_lower:
            self.floor_lower, self.ceil_lower = self.ceil_lower, self.floor_lower
        # The adaptive upper band must not sit below the scale-down trigger,
        # nor the lower band above the scale-up trigger.
        if self.ceil_upper <= self.floor_lower:
            self.ceil_upper = min(CPU_MAX, self.floor_lower + self.min_band)
        if self.floor_upper <= self.floor_lower:
            self.floor_lower = max(CPU_MIN, self.floor_upper - self.min_band)
        if self.static_lower > self.static_upper:
            self.static_lower, self.static_upper = self.static_upper, self.static_lower

    # ------------------------------------------------------------------ update
    def update(self, cpu: float) -> None:
        try:
            value = float(cpu)
        except (TypeError, ValueError):
            return
        if not math.isfinite(value):
            return
        self.window.append(_clamp(value, CPU_MIN, CPU_MAX))

    def get_thresholds(self) -> tuple:
        samples = len(self.window)

        if samples < self.min_samples:
            self.last_state = {
                "mode": "static", "mu": None, "sigma": None,
                "upper": self.static_upper, "lower": self.static_lower,
                "samples": samples,
            }
            return self.static_upper, self.static_lower

        mu = statistics.fmean(self.window)
        sigma = statistics.pstdev(self.window)      # 0.0 for a flat window

        upper = mu + self.k * sigma
        lower = mu - self.k * sigma

        # A flat (zero-variance) window would collapse the band to a point;
        # open it by min_band so the two triggers stay distinguishable.
        if (upper - lower) < self.min_band:
            upper = mu + self.min_band / 2.0
            lower = mu - self.min_band / 2.0

        upper = _clamp(_clamp(upper, self.floor_upper, self.ceil_upper), CPU_MIN, CPU_MAX)
        lower = _clamp(_clamp(lower, self.floor_lower, self.ceil_lower), CPU_MIN, CPU_MAX)

        # Final inversion guard (config could still make the clamps overlap).
        if lower >= upper:
            lower = max(CPU_MIN, upper - self.min_band)
            if lower >= upper:
                upper = min(CPU_MAX, lower + self.min_band)

        self.last_state = {
            "mode": "adaptive", "mu": mu, "sigma": sigma,
            "upper": upper, "lower": lower, "samples": samples,
        }
        return upper, lower

    def describe(self) -> dict:
        """Effective limits + last computed state, for logging/dashboards."""
        return {
            "mode": self.last_state.get("mode", "static"),
            "upper": self.last_state.get("upper"),
            "lower": self.last_state.get("lower"),
            "mu": self.last_state.get("mu"),
            "sigma": self.last_state.get("sigma"),
            "samples": self.last_state.get("samples", len(self.window)),
            "k": self.k,
            "min_band": self.min_band,
            "floor_upper": self.floor_upper,
            "ceil_upper": self.ceil_upper,
            "floor_lower": self.floor_lower,
            "ceil_lower": self.ceil_lower,
            "static_upper": self.static_upper,
            "static_lower": self.static_lower,
        }

    def reset(self) -> None:
        self.window.clear()
        self.last_state = {
            "mode": "static", "mu": None, "sigma": None,
            "upper": self.static_upper, "lower": self.static_lower, "samples": 0,
        }
