import csv
import json
import os
import time
from datetime import datetime

import docker

from monitoring.metrics import get_request_count, reset_request_count


CONTAINER_NAME = "proximascale-app"
FILE_PATH = "data/collected/metrics.csv"

docker_client = docker.from_env()


def get_container_cpu_percent():
    container = docker_client.containers.get(CONTAINER_NAME)
    stats = container.stats(stream=False)

    cpu_delta = (
        stats["cpu_stats"]["cpu_usage"]["total_usage"]
        - stats["precpu_stats"]["cpu_usage"]["total_usage"]
    )

    system_delta = (
        stats["cpu_stats"]["system_cpu_usage"]
        - stats["precpu_stats"]["system_cpu_usage"]
    )

    if system_delta <= 0 or cpu_delta < 0:
        return 0.0

    num_cpus = stats["cpu_stats"].get("online_cpus", 1)

    return round(
        min((cpu_delta / system_delta) * num_cpus * 100, 100.0),
        2
    )


def get_container_memory_percent():
    container = docker_client.containers.get(CONTAINER_NAME)
    stats = container.stats(stream=False)

    usage = stats["memory_stats"]["usage"]
    limit = stats["memory_stats"]["limit"]

    return round((usage / limit) * 100, 2)


def is_post_scaling():
    event_path = "data/scaling_events.json"

    if not os.path.exists(event_path):
        return False

    try:
        with open(event_path, "r") as file:
            event = json.load(file)

        elapsed = time.time() - event["timestamp"]
        return elapsed < (event["expires_after_steps"] * 60)

    except (json.JSONDecodeError, KeyError, TypeError):
        return False


def ensure_csv_header():
    os.makedirs(os.path.dirname(FILE_PATH), exist_ok=True)

    if not os.path.exists(FILE_PATH) or os.path.getsize(FILE_PATH) == 0:
        with open(FILE_PATH, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "timestamp",
                "cpu_percent",
                "memory_percent",
                "request_rate",
                "post_scaling"
            ])


def collect_metrics():
    ensure_csv_header()

    while True:
        cpu = get_container_cpu_percent()
        memory = get_container_memory_percent()
        request_rate = get_request_count()
        post_scaling = is_post_scaling()
        timestamp = datetime.now().isoformat()

        with open(FILE_PATH, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                timestamp,
                cpu,
                memory,
                request_rate,
                post_scaling
            ])

        print(
            f"{timestamp} | "
            f"Container CPU: {cpu}% | "
            f"Container Memory: {memory}% | "
            f"Requests: {request_rate} | "
            f"Post-scaling: {post_scaling}"
        )

        reset_request_count()
        time.sleep(60)


if __name__ == "__main__":
    collect_metrics()