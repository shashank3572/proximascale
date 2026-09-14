import json
import os
import time
from datetime import datetime

import docker

from monitoring.metrics import reset_request_count
from monitoring.schema import MetricRecord
from monitoring.storage import append_row


CONTAINER_NAME = "proximascale-app"


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


docker_client = docker.from_env()


def collect_metrics():
    while True:
        cpu = get_container_cpu_percent()
        memory = get_container_memory_percent()
        # reset_request_count() reads-and-clears the counter in a single
        # transaction, so a request that arrives mid-poll is never lost
        # or double-counted (unlike calling get_request_count() followed
        # by a separate reset_request_count()).
        request_rate = reset_request_count()
        post_scaling = is_post_scaling()
        timestamp = datetime.now().isoformat()

        record = MetricRecord(
            timestamp=timestamp,
            cpu_percent=cpu,
            memory_percent=memory,
            request_rate=request_rate,
            post_scaling=post_scaling,
        )
        append_row(record)

        print(
            f"{timestamp} | "
            f"Container CPU: {cpu}% | "
            f"Container Memory: {memory}% | "
            f"Requests: {request_rate} | "
            f"Post-scaling: {post_scaling}"
        )

        time.sleep(60)


if __name__ == "__main__":
    collect_metrics()
