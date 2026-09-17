from locust import HttpUser, task, between, LoadTestShape

class RampUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def load_light(self):
        self.client.get("/work/light")


class GradualRampShape(LoadTestShape):
    def tick(self):
        run_time = self.get_run_time()

        if run_time < 300:
            users = 2 + int((23 * run_time) / 300)
            return min(users, 25), 1

        return None