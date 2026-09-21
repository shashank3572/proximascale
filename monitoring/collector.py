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

CPU_SAMPLE_GAP = 2       # seconds between the two samples used for one CPU reading

CONTAINER_NAME = "proximascale-app"

# Where the Flask app exposes the request counter. The app runs inside a
# container, so its SQLite file (data/collected/metrics.db) is NOT shared with
# the host -- reading that file here always returned 0. The HTTP endpoint is
# the correct cross-boundary channel; the file is only a fallback for a
# host-run app.py (Option A in docs/setup_guide.md).
APP_HOST = os.getenv("PROXIMASCALE_APP_HOST", "localhost")
APP_PORT = int(os.getenv("PROXIMASCALE_APP_PORT", "5000"))
REQUEST_RESET_TIMEOUT = 3  # seconds

_docker_client = None


def _get_docker_client():
    """Create the Docker client on first use so importing this module
    does not require a running Docker daemon."""
    global _docker_client
    if _docker_client is None:
        _docker_client = docker.from_env()
    return _docker_client


def get_container_cpu_percent():
    """Container CPU% over CPU_SAMPLE_GAP seconds, from two consecutive stats.

    Why two samples: a single stats(stream=False) call compares total_usage
    against precpu_stats -- Docker's cached snapshot of the PREVIOUS call.
    When the previous call happened long ago (or is the first-ever call),
    precpu is stale/zero and the formula degenerates to 0% or 100%, which is
    exactly the flapping this collector used to log. Sampling twice and
    diffing the two fresh snapshots always yields a real window.
    """
    container = _get_docker_client().containers.get(CONTAINER_NAME)

    stats_a = container.stats(stream=False)
    time.sleep(CPU_SAMPLE_GAP)
    stats_b = container.stats(stream=False)

    return _cpu_percent_from_stats(stats_a, stats_b)


def _cpu_percent_from_stats(stats_a, stats_b) -> float:
    """CPU% between two stat snapshots. Separated from the I/O above for tests."""
    cpu_a = stats_a["cpu_stats"]["cpu_usage"]["total_usage"]
    sys_a = stats_a["cpu_stats"]["system_cpu_usage"]
    cpu_b = stats_b["cpu_stats"]["cpu_usage"]["total_usage"]
    sys_b = stats_b["cpu_stats"]["system_cpu_usage"]

    cpu_delta = cpu_b - cpu_a
    system_delta = sys_b - sys_a

    if system_delta <= 0 or cpu_delta < 0:
        return 0.0

    num_cpus = stats_b["cpu_stats"].get("online_cpus", 1)

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


def collect_request_count():
    """Requests that arrived during the CURRENT poll window, read-and-reset.

    Tries the Flask app's POST /request-rate/reset first (works across the
    container boundary); falls back to the local SQLite counter, which only
    matches the app when app.py runs on the host rather than in a container.

    Returns -1 when neither channel is reachable, so the CSV row visibly
    records 'requests unavailable' instead of a fake zero.
    """
    try:
        import requests
        resp = requests.post(
            f"http://{APP_HOST}:{APP_PORT}/request-rate/reset",
            timeout=REQUEST_RESET_TIMEOUT,
        )
        if resp.ok:
            return int(resp.json()["request_count"])
        logging.warning("reset endpoint returned HTTP %s; falling back to SQLite",
                        resp.status_code)
    except Exception as e:
        logging.warning("reset endpoint unreachable (%s); falling back to SQLite", e)

    try:
        return int(reset_request_count())
    except Exception as e:
        logging.warning("SQLite request counter unavailable: %s", e)
        return -1


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
        # collect_request_count() reads-and-clears the app's counter over the
        # HTTP boundary (SQLite fallback for a host-run app), so each CSV row
        # is the request volume of exactly one poll interval.
        request_rate = collect_request_count()
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