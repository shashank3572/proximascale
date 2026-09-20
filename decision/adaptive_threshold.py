# decision/adaptive_threshold.py  (edited)
from collections import deque
import statistics


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
        floor_lower: float = 15.0,
        ceil_lower: float = 45.0,
        min_band: float = 10.0,          # NEW
    ):
        self.window = deque(maxlen=window_size)
        self.k = k
        self.min_samples = min_samples
        self.static_upper = static_upper
        self.static_lower = static_lower
        self.floor_upper, self.ceil_upper = floor_upper, ceil_upper
        self.floor_lower, self.ceil_lower = floor_lower, ceil_lower
        self.min_band = min_band

    def update(self, cpu: float) -> None:
        self.window.append(float(cpu))

    def get_thresholds(self) -> tuple[float, float]:
        if len(self.window) < self.min_samples:
            return self.static_upper, self.static_lower

        mu = statistics.fmean(self.window)
        sigma = statistics.pstdev(self.window)

        upper = mu + self.k * sigma
        lower = mu - self.k * sigma

        # NEW: enforce a minimum band so steady inputs don't collapse to a point
        if (upper - lower) < self.min_band:
            upper = mu + self.min_band / 2.0
            lower = mu - self.min_band / 2.0

        upper = max(self.floor_upper, min(self.ceil_upper, upper))
        lower = max(self.static_lower, min(self.ceil_lower, lower))
        return upper, lower

    def reset(self) -> None:
        self.window.clear()