# actuator/docker_scaler.py  (edited)
import docker
import time


class DockerActuator:
    def __init__(self, config: dict):
        # Flatten nested scaling_rules → same trick as engine
        flat = dict(config)
        flat.update(config.get("scaling_rules", {}))

        self.client = docker.from_env()

        # Image: accept either 'image' (flat) or 'target_service_image' (nested)
        self.image = (
            flat.get("image")
            or flat.get("target_service_image")
            or "nginx:alpine"
        )

        # Optional: pre-pull (no-op if already cached; mocked in tests)
        try:
            self.client.images.pull(self.image)
        except Exception:
            pass

        self.container_prefix = flat.get("container_prefix", "proximascale-worker")
        self.min_containers = flat.get("min_containers", 1)
        self.max_containers = flat.get("max_containers", 10)

        # cgroup limits — locked in the plan
        self.cpu_quota = int(flat.get("cpu_quota", 50_000))   # 0.5 CPU
        self.mem_limit = flat.get("mem_limit", "256m")
        self.cpu_period = int(flat.get("cpu_period", 100_000))

        import os
        self.scale_down_strategy = (
            flat.get("scale_down_strategy")
            or os.getenv("SCALE_DOWN_STRATEGY")
            or "newest"
        )

    # ------------------------------------------------------------------ helpers
    def _workers(self):
        """All running containers managed by this scaler, oldest → newest."""
        containers = self.client.containers.list(
            filters={"name": self.container_prefix}
        )
        # sort by creation time — Docker list order is not guaranteed
        return sorted(containers, key=lambda c: c.attrs["Created"])

    # ------------------------------------------------------------------ actions
    def scale_up(self) -> bool:
        current = self._workers()
        if len(current) >= self.max_containers:
            return False

        n = len(current) + 1
        new_name = f"{self.container_prefix}-{n}-{int(time.time())}"

        self.client.containers.run(
            self.image,
            name=new_name,
            detach=True,
            cpu_quota=self.cpu_quota,
            cpu_period=self.cpu_period,
            mem_limit=self.mem_limit,
            # DO NOT auto-remove — we need it to survive for the demo
            remove=False,
        )
        return True

    def scale_down(self) -> bool:
        current = self._workers()          # oldest → newest
        if len(current) <= self.min_containers:
            return False

        # Spec resolution: c.md says "most recently added" (containers[-1]);
        # master plan says "oldest extra". Per audit, DEFAULT to newest
        # (matches c.md, which is the stricter/authoritative doc), but keep
        # it configurable so the team can flip without a code change.
        pick = self._pick_down_target(current)
        pick.stop(timeout=5)
        pick.remove()
        return True

    def _pick_down_target(self, ordered):
        """Select a container to stop.
        `ordered` is oldest → newest (from _workers()).
        Strategy precedence: config.yaml > env var > default 'newest'.
        """
        if self.scale_down_strategy == "oldest":
            return ordered[0]
        return ordered[-1]      # 'newest' (default)


DockerScaler = DockerActuator