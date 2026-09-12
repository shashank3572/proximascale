"""
spike_load.py — Sudden traffic spike then drop.
Creates repeated traffic spikes for ML training data.

Run:
    locust -f data/locust_scenarios/spike_load.py --host=http://localhost:5000 \
           --headless --run-time 8m
"""

from locust import HttpUser, task, between, LoadTestShape


class SpikeUser(HttpUser):
    wait_time = between(0.5, 1.5)

    @task(3)
    def load_light(self):
        self.client.get("/work/light")

    @task(1)
    def load_heavy(self):
        self.client.get("/work/heavy")


class SpikeShape(LoadTestShape):
    """
    Timeline:
      0:00 → 1:00 : 1 user
      1:00 → 3:00 : 100 users
      3:00 → 5:00 : 1 user
      5:00 → 6:00 : 100 users
      6:00 → 8:00 : 1 user
    """

    stages = [
        {"duration": 60, "users": 1, "spawn_rate": 1},
        {"duration": 180, "users": 100, "spawn_rate": 100},
        {"duration": 300, "users": 1, "spawn_rate": 100},
        {"duration": 360, "users": 100, "spawn_rate": 100},
        {"duration": 480, "users": 1, "spawn_rate": 100},
    ]

    def tick(self):
        run_time = self.get_run_time()

        for stage in self.stages:
            if run_time < stage["duration"]:
                return stage["users"], stage["spawn_rate"]

        return None