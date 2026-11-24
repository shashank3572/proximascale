import time


class Hysteresis:
    """
    Prevents rapid scale-up/scale-down oscillation by enforcing a
    cooldown period between consecutive scaling actions.
    """

    def __init__(self, cooldown_seconds: int = 180):
        self.cooldown_seconds = cooldown_seconds
        self.last_action_time: float = 0  # epoch seconds; 0 = never acted
        # Last successful scale_UP of ANY reason (anomaly, upper_bound_risk,
        # cpu_high, current_cpu_high). The anomaly branch bypasses the
        # cooldown, so it needs its own spacing clock -- see
        # `scale_up_interval_elapsed`.
        self.last_scale_up_time: float = 0  # epoch seconds; 0 = never scaled up

    def is_cooling_down(self) -> bool:
        """Returns True if we are still inside the cooldown window."""
        return (time.time() - self.last_action_time) < self.cooldown_seconds

    def seconds_remaining(self) -> int:
        """How many seconds are left in the current cooldown."""
        remaining = self.cooldown_seconds - (time.time() - self.last_action_time)
        return max(0, int(remaining))

    def seconds_since_last_scale_up(self) -> float:
        """Seconds since the last successful scale_up of any reason.

        `inf` when nothing has scaled up yet, so a first anomaly is never
        spaced out.
        """
        if self.last_scale_up_time <= 0:
            return float("inf")
        return time.time() - self.last_scale_up_time

    def scale_up_interval_elapsed(self, interval_seconds: float) -> bool:
        """True when the last scale_up is at least `interval_seconds` old.

        interval_seconds <= 0 disables the check (always elapsed).
        """
        if interval_seconds is None or interval_seconds <= 0:
            return True
        return self.seconds_since_last_scale_up() >= interval_seconds

    def record_action(self, action: str = None):
        """Call this immediately after a scale_up or scale_down succeeds.

        `action` is optional for backward compatibility; pass it so the
        scale-up spacing clock advances only on real scale-ups.
        """
        now = time.time()
        self.last_action_time = now
        if action == "scale_up":
            self.last_scale_up_time = now
