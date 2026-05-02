"""
spike_load.py — Sudden traffic spike then drop.
Ramps from 1 to 100 users instantly, holds 2 minutes, drops back to 1.
This generates the spike pattern the LSTM needs to learn.

Run:
    locust -f data/locust_scenarios/spike_load.py --host=http://localhost:5000 \
           --headless --run-time 8m
"""
from locust import HttpUser, task, between, LoadTestShape


class SpikeUser(HttpUser):
    wait_time = between(0.5, 1.5)

    @task(3)
    def load_home(self):
        self.client.get("/")

    @task(1)
    def load_heavy(self):
        self.client.get("/heavy")


class SpikeShape(LoadTestShape):
    """
    Timeline (minutes):
      0:00 →  1:00  : 1 user   (pre-spike baseline)
      1:00 →  3:00  : 100 users (spike — instant ramp)
      3:00 →  5:00  : 1 user   (post-spike drop)
      5:00 →  6:00  : 100 users (second spike)
      6:00 →  8:00  : 1 user   (recovery)
    Two spikes in one run to give the LSTM more examples.
    """
    stages = [
        {"duration": 60,  "users": 1,   "spawn_rate": 1},
        {"duration": 180, "users": 100, "spawn_rate": 100},
        {"duration": 300, "users": 1,   "spawn_rate": 100},
        {"duration": 360, "users": 100, "spawn_rate": 100},
        {"duration": 480, "users": 1,   "spawn_rate": 100},
    ]

    def tick(self):
        run_time = self.get_run_time()
        for stage in self.stages:
            if run_time < stage["duration"]:
                return stage["users"], stage["spawn_rate"]
        return None  # stop the test
