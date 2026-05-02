import time


class Hysteresis:
    """
    Prevents rapid scale-up/scale-down oscillation by enforcing a
    cooldown period between consecutive scaling actions.
    """

    def __init__(self, cooldown_seconds: int = 180):
        self.cooldown_seconds = cooldown_seconds
        self.last_action_time: float = 0  # epoch seconds; 0 = never acted

    def is_cooling_down(self) -> bool:
        """Returns True if we are still inside the cooldown window."""
        return (time.time() - self.last_action_time) < self.cooldown_seconds

    def seconds_remaining(self) -> int:
        """How many seconds are left in the current cooldown."""
        remaining = self.cooldown_seconds - (time.time() - self.last_action_time)
        return max(0, int(remaining))

    def record_action(self):
        """Call this immediately after a scale_up or scale_down succeeds."""
        self.last_action_time = time.time()
