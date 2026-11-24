"""Docker actuator: create and remove worker containers.

ARCHITECTURE -- read this before claiming anything about "traffic"
---------------------------------------------------------------
ProximaScale runs ONE application container (proximascale-app) that serves the
workload on :5000 and is the container the collector measures. Scaling adds
WORKER containers from the same image. In the default mode those workers are a
RESOURCE POOL:

    logical replicas   the number the engine is managing
    running containers containers this actuator created that are running
    healthy workers   running containers whose /health endpoint answers
    traffic-serving   workers actually receiving application requests

Pool mode does NOT install a load balancer, so workers only receive requests if
something sends them there. `worker_status()` reports `traffic_serving`
separately and honestly for exactly that reason -- a running worker is not
proof that it serves traffic. Set `worker_host_port_base` to publish base+i ->
5000 per worker and put those ports behind a proxy; see
docs/DEMO_TROUBLESHOOTING.md.

Replica counts are always derived from what Docker reports, never from an
internal counter, so a failed creation cannot desynchronise the fleet.
"""
import logging
import os
import time

import docker

from actuator.scaler_interface import ScalerInterface

logger = logging.getLogger(__name__)

MANAGED_LABEL = "proximascale.managed"
PREFIX_LABEL = "proximascale.prefix"
ROLE_LABEL = "proximascale.role"

# The worker image is the same Flask app (python app.py, /health on 5000), so
# this is a real readiness signal and not a placeholder.
_HEALTHCHECK = [
    "CMD", "python", "-c",
    "import urllib.request;"
    "urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=2).read()",
]


class DockerActuator(ScalerInterface):
    def __init__(self, config):
        super().__init__()
        # Flatten nested scaling_rules -> same trick as engine
        flat = dict(config)
        flat.update(config.get("scaling_rules", {}))

        # Lazy client: DecisionEngine construction must never fail when Docker
        # Desktop is not running (the engine degrades to *_no_actuator).
        self._client = None

        self.image = (
            flat.get("image")
            or flat.get("target_service_image")
            or "nginx:alpine"
        )
        self._image_pulled = False

        self.container_prefix = flat.get("container_prefix", "proximascale-worker")
        self.min_containers = int(flat.get("min_containers", 1))
        self.max_containers = int(flat.get("max_containers", 10))

        # cgroup limits -- locked in the plan
        self.cpu_quota = int(flat.get("cpu_quota", 50_000))   # 0.5 CPU
        self.mem_limit = flat.get("mem_limit", "256m")
        self.cpu_period = int(flat.get("cpu_period", 100_000))

        # M6: a worker that dies must not silently vanish from the fleet while
        # the dashboard still counts it as capacity.
        self.restart_policy = flat.get("restart_policy", "unless-stopped")
        self.health_check = bool(flat.get("worker_health_check", True))
        self.readiness_timeout = float(flat.get("readiness_timeout_seconds", 20))
        self.worker_host_port_base = flat.get("worker_host_port_base")

        self.scale_down_strategy = (
            flat.get("scale_down_strategy")
            or os.getenv("SCALE_DOWN_STRATEGY")
            or "newest"
        )
        self.last_error = None
        self.last_ready = None

    @property
    def client(self):
        """Lazily create the Docker client on first use."""
        if self._client is None:
            self._client = docker.from_env()
        return self._client

    # ----------------------------------------------------------------- helpers
    def _labels(self):
        return {
            MANAGED_LABEL: "true",
            ROLE_LABEL: "worker",
            PREFIX_LABEL: self.container_prefix,
        }

    def _ensure_image_pulled(self) -> None:
        """Pull the worker image once. A failure is only a warning: the image is
        normally already local (offline demo) and Docker will use it."""
        if self._image_pulled:
            return
        try:
            self.client.images.pull(self.image)
        except Exception as e:
            logger.debug("image pull skipped for %s (%s)", self.image, e)
        self._image_pulled = True

    def _workers(self):
        """Running worker containers managed by this scaler, oldest -> newest."""
        containers = self.client.containers.list(
            filters={"label": f"{PREFIX_LABEL}={self.container_prefix}"}
        )
        # Legacy containers (created before labelling existed) still count, so
        # a demo on an older build can scale down cleanly. The name match is
        # prefix-scoped and cannot adopt unrelated user containers.
        try:
            by_name = self.client.containers.list(
                filters={"name": f"{self.container_prefix}-"}
            )
        except docker.errors.DockerException:
            by_name = []
        seen = {c.id for c in containers}
        for c in by_name:
            if c.id not in seen:
                containers.append(c)
        return sorted(containers, key=lambda c: c.attrs.get("Created", ""))

    def replica_count(self) -> int:
        """Number of running worker containers."""
        return len(self._workers())

    def worker_status(self) -> list:
        """One record per running worker, with health -- never fabricated.

        `healthy` is None when Docker reports no health status (no
        healthcheck), True/"healthy" when it does. `traffic_serving` is always
        False in pool mode: nothing routes application requests here.
        """
        records = []
        for c in self._workers():
            try:
                c.reload()
                state = c.attrs.get("State", {}) or {}
                health = (state.get("Health") or {}).get("Status")
            except docker.errors.DockerException as e:
                state, health = {}, None
                logger.warning("could not inspect worker %s: %s",
                               getattr(c, "name", c.id), e)
            records.append({
                "name": c.name,
                "id": getattr(c, "short_id", c.id),
                "status": state.get("Status"),
                "healthy": health,
                "traffic_serving": False,
            })
        return records

    def healthy_replica_count(self) -> int:
        """Running workers Docker reports as healthy, or all running workers
        when no health status is available at all."""
        records = self.worker_status()
        healthy = [r for r in records if r["healthy"] in (True, "healthy")]
        if records and all(r["healthy"] is None for r in records):
            return len(records)
        return len(healthy)

    def describe_mode(self) -> dict:
        return {
            "mode": "pool",
            "has_load_balancer": False,
            "publishes_ports": self.worker_host_port_base is not None,
            "traffic_serving_verified": False,
            "restart_policy": self.restart_policy,
            "health_check": self.health_check,
        }

    # ------------------------------------------------------------------ actions
    def scale_up(self) -> bool:
        self._ensure_image_pulled()
        current = self._workers()
        if len(current) >= self.max_containers:
            return False

        n = len(current) + 1
        # Unique per attempt: rapid create/remove/create must never collide on
        # a name Docker still holds.
        new_name = f"{self.container_prefix}-{n}-{int(time.time() * 1000)}"

        kwargs = dict(
            name=new_name,
            detach=True,
            cpu_quota=self.cpu_quota,
            cpu_period=self.cpu_period,
            mem_limit=self.mem_limit,
            labels=self._labels(),
            remove=False,
        )
        if self.restart_policy and self.restart_policy != "no":
            kwargs["restart_policy"] = {"Name": self.restart_policy}
        if self.health_check:
            kwargs["healthcheck"] = {
                "test": _HEALTHCHECK,
                "interval": 10 * 1_000_000_000,
                "timeout": 3 * 1_000_000_000,
                "retries": 3,
                "start_period": 5 * 1_000_000_000,
            }
        if self.worker_host_port_base is not None:
            kwargs["ports"] = {"5000/tcp": int(self.worker_host_port_base) + n - 1}

        try:
            container = self.client.containers.run(self.image, **kwargs)
        except (docker.errors.APIError, docker.errors.DockerException) as e:
            self.last_error = str(e)
            logger.error("scale_up failed creating %s: %s", new_name, e)
            return False

        self.last_error = None
        self.last_ready = self._wait_ready(container)
        if self.last_ready is False:
            # Created but not ready in time: it exists and counts as a replica
            # (Docker is the source of truth), so this is NOT max_reached.
            logger.warning("worker %s created but not healthy within %.0fs",
                           new_name, self.readiness_timeout)
        return True

    def _wait_ready(self, container):
        """Best-effort readiness wait. True/False, or None when unknown."""
        if self.readiness_timeout <= 0:
            return None
        deadline = time.time() + self.readiness_timeout
        try:
            while time.time() < deadline:
                container.reload()
                state = container.attrs.get("State", {}) or {}
                health = (state.get("Health") or {}).get("Status")
                if not isinstance(health, str):
                    # No healthcheck / unreadable payload: "running" is the
                    # best signal available. (Also keeps this bounded when the
                    # container object is a stub.)
                    return (state.get("Status") == "running"
                            if isinstance(state.get("Status"), str) else None)
                if health == "healthy":
                    return True
                if health == "unhealthy":
                    return False
                if health is None:
                    return state.get("Status") == "running"
                time.sleep(0.5)
            return False
        except Exception:
            return None

    def scale_down(self) -> bool:
        current = self._workers()          # oldest -> newest
        if len(current) <= self.min_containers:
            return False

        # "newest" (default, matches c.md) -- configurable via
        # scale_down_strategy/SCALE_DOWN_STRATEGY without a code change.
        pick = self._pick_down_target(current)
        try:
            pick.stop(timeout=5)
            pick.remove()
        except docker.errors.NotFound:
            # Already gone: the fleet is smaller than the engine thought, which
            # is a successful scale-down from the caller's perspective.
            logger.warning("scale_down target %s disappeared before removal",
                           getattr(pick, "name", "?"))
            return True
        except docker.errors.DockerException as e:
            self.last_error = str(e)
            logger.error("scale_down failed removing %s: %s",
                         getattr(pick, "name", "?"), e)
            return False
        self.last_error = None
        return True

    def _pick_down_target(self, ordered):
        """`ordered` is oldest -> newest (from _workers()).
        Strategy precedence: config.yaml > env var > default 'newest'."""
        if self.scale_down_strategy == "oldest":
            return ordered[0]
        return ordered[-1]      # 'newest' (default)

    def hold(self) -> bool:
        """No-op for Docker -- nothing to do for a hold decision."""
        return True


DockerScaler = DockerActuator
