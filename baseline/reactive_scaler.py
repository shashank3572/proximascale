"""
baseline/reactive_scaler.py
Person D — Reactive baseline scaler (comparison group).

Purpose:
    Read CURRENT container CPU every 10s. If it crosses a static
    threshold (70%), scale up. If it drops well below, scale down.
    No ML. No prediction. This is the "respond after the fact"
    behaviour that ProximaScale is compared against.

Runs against the same Docker container as ProximaScale so the
two can be benchmarked under identical Locust spike scenarios.
"""
import time
import logging
import docker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [BASELINE] %(message)s"
)
logger = logging.getLogger(__name__)

CONTAINER_NAME = "proximascale-app"
SCALE_UP_THRESHOLD = 70.0
SCALE_DOWN_THRESHOLD = 30.0
POLL_INTERVAL = 10          # reactive polls much more frequently
MIN_REPLICAS = 1
MAX_REPLICAS = 5

client = docker.from_env()


def get_container_cpu(container) -> float:
    """Return container CPU% using the standard Docker delta formula."""
    stats = container.stats(stream=False)

    cpu_delta = (
        stats["cpu_stats"]["cpu_usage"]["total_usage"]
        - stats["precpu_stats"]["cpu_usage"]["total_usage"]
    )
    system_delta = (
        stats["cpu_stats"]["system_cpu_usage"]
        - stats["precpu_stats"]["system_cpu_usage"]
    )
    online_cpus = stats["cpu_stats"].get("online_cpus", 1)

    if system_delta > 0 and cpu_delta > 0:
        return (cpu_delta / system_delta) * online_cpus * 100.0
    return 0.0


def get_replicas() -> int:
    return len(client.containers.list(filters={"name": CONTAINER_NAME}))


def scale_to(n: int):
    """Scale the service to n replicas using docker-compose-style scale."""
    logger.info(f"Scaling {CONTAINER_NAME} → {n} replicas")
    # NOTE: with plain Docker SDK, replicas are managed by a
    # service/stack. If the team uses docker-compose, call:
    #   subprocess.run(["docker", "compose", "up", "-d",
    #                   "--scale", f"{CONTAINER_NAME}={n}"])
    # We keep the call site abstract so integration can adapt it.


def main():
    logger.info(
        f"Reactive baseline starting. "
        f"up>{SCALE_UP_THRESHOLD}% down<{SCALE_DOWN_THRESHOLD}% "
        f"poll={POLL_INTERVAL}s"
    )

    while True:
        try:
            containers = client.containers.list(
                filters={"name": CONTAINER_NAME}
            )
            if not containers:
                logger.warning(f"No container named {CONTAINER_NAME}")
                time.sleep(POLL_INTERVAL)
                continue

            cpu = get_container_cpu(containers[0])
            replicas = get_replicas()

            if cpu > SCALE_UP_THRESHOLD and replicas < MAX_REPLICAS:
                logger.info(
                    f"REACTIVE: CPU={cpu:.1f}% > {SCALE_UP_THRESHOLD}% "
                    f"— scaling up ({replicas} → {replicas + 1})"
                )
                scale_to(replicas + 1)

            elif cpu < SCALE_DOWN_THRESHOLD and replicas > MIN_REPLICAS:
                logger.info(
                    f"REACTIVE: CPU={cpu:.1f}% < {SCALE_DOWN_THRESHOLD}% "
                    f"— scaling down ({replicas} → {replicas - 1})"
                )
                scale_to(replicas - 1)

            else:
                logger.info(f"CPU={cpu:.1f}% replicas={replicas} — hold")

        except docker.errors.DockerException as e:
            logger.error(f"Docker error: {e}")

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()