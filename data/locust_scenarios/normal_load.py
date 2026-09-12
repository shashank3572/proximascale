"""
normal_load.py — Steady baseline traffic.
Simulates normal, consistent application traffic.

Run:
    locust -f data/locust_scenarios/normal_load.py --host=http://localhost:5000 \
           --headless -u 20 -r 2 --run-time 30m
"""

from locust import HttpUser, task, between


class NormalUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def light_work(self):
        self.client.get("/work/light")