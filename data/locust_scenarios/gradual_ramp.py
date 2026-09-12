"""
gradual_ramp.py — Slow, linear traffic increase then decrease.
Adds 10 users every 30 seconds up to 100, holds, then ramps back down.
Gives the LSTM a clean ramp profile distinct from spikes.
"""

from locust import HttpUser, task, between, LoadTestShape


class RampUser(HttpUser):
    wait_time = between(1, 2)

    @task(3)
    def load_light(self):
        self.client.get("/work/light")

    @task(1)
    def load_heavy(self):
        self.client.get("/work/heavy")


class GradualRampShape(LoadTestShape):
    """
    Ramp up: +10 users every 30s until 100 users
    Hold: 100 users for 2 minutes
    Ramp down: -10 users every 30s until 0
    Total: ~12 minutes
    """

    STEP_SIZE = 10
    STEP_DURATION = 30
    MAX_USERS = 100
    HOLD_DURATION = 120

    def tick(self):
        run_time = self.get_run_time()

        ramp_up_duration = (
            self.MAX_USERS // self.STEP_SIZE
        ) * self.STEP_DURATION

        hold_end = ramp_up_duration + self.HOLD_DURATION
        ramp_down_duration = ramp_up_duration
        total = hold_end + ramp_down_duration

        if run_time > total:
            return None

        if run_time <= ramp_up_duration:
            step = int(run_time // self.STEP_DURATION)
            users = min(
                (step + 1) * self.STEP_SIZE,
                self.MAX_USERS
            )
            return users, self.STEP_SIZE

        elif run_time <= hold_end:
            return self.MAX_USERS, self.STEP_SIZE

        else:
            elapsed_down = run_time - hold_end
            step = int(elapsed_down // self.STEP_DURATION)
            users = max(
                self.MAX_USERS - (step + 1) * self.STEP_SIZE,
                0
            )
            return max(users, 1), self.STEP_SIZE