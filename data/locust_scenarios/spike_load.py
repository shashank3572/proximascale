from locust import HttpUser, task, between


class SpikeUser(HttpUser):
    wait_time = between(0.1, 0.1)

    @task
    def load_heavy(self):
        self.client.get("/work/heavy")