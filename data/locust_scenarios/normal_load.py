"""
normal_load.py — Steady baseline traffic.
Simulates ~10-30 users making requests at a normal, consistent pace.

Run:
    locust -f data/locust_scenarios/normal_load.py --host=http://localhost:5000 \
           --headless -u 20 -r 2 --run-time 30m
"""
from locust import HttpUser, task, between


class NormalUser(HttpUser):
    wait_time = between(1, 3)   # each user waits 1-3s between requests

    @task(3)
    def load_home(self):
        self.client.get("/")

    @task(1)
    def load_heavy(self):
        self.client.get("/heavy")
