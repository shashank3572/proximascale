import json
import logging
import os
import time
from datetime import datetime

import docker
import yaml

from decision.scaling_log import read_scaling_events
from monitoring.metrics import reset_request_count
from monitoring.schema import MetricRecord
from monitoring.storage import append_row

POLL_INTERVAL = 30       # seconds between each sample -- matches the model's
                         # trained sampling rate (model/evaluate.py SAMPLING_INTERVAL_SEC,
                         # model/prophet_model.py DEFAULT_FREQ). Do not drift these apart.


CONTAINER_NAME = "proximascale-app"

_docker_client = None


def _get_docker_client():
    """Create the Docker client on first use so importing this module
    does not require a running Docker daemon."""
    global _docker_client
    if _docker_client is None:
        _docker_client = docker.from_env()
    return _docker_client


def get_container_cpu_percent():
    container = _get_docker_client().containers.get(CONTAINER_NAME)
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
    container = _get_docker_client().containers.get(CONTAINER_NAME)
    stats = container.stats(stream=False)

    usage = stats["memory_stats"]["usage"]
    limit = stats["memory_stats"]["limit"]

    return round((usage / limit) * 100, 2)


_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"
)


def _tick_seconds(default: float = 3.0) -> float:
    """Engine tick length; expires_after_steps is counted in these."""
    try:
        with open(_CONFIG_PATH, "r") as f:
            return float(yaml.safe_load(f).get("tick_seconds", default))
    except (OSError, ValueError, AttributeError, yaml.YAMLError):
        return default


def is_post_scaling() -> bool:
    """True while the latest scale event's cooldown window is still active.

    Reads the JSON array written by decision/scaling_log.py.
    Window (s) = expires_after_steps * tick_seconds (== cooldown_seconds).
    """
    try:
        events = read_scaling_events()
    except (OSError, json.JSONDecodeError):
        return False
    if not events:
        return False
    try:
        last = max(events, key=lambda e: e["timestamp"])
        window = last["expires_after_steps"] * _tick_seconds()
        return (time.time() - last["timestamp"]) < window
    except (KeyError, TypeError):
        logging.warning("scaling_events.json has malformed entries; "
                        "treating as not post-scaling")
        return False


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

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    collect_metrics()