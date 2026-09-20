"""cleanup.py must target the same container names the actuator creates."""
from unittest.mock import MagicMock, patch

import cleanup
from actuator.docker_scaler import DockerActuator


def test_prefix_matches_actuator_naming():
    with patch("docker.from_env") as m:
        m.return_value.images.pull.return_value = None
        m.return_value.containers.list.return_value = []
        import yaml
        cfg = yaml.safe_load(open(cleanup.CONFIG_PATH))
        actuator = DockerActuator(cfg)
    assert cleanup.worker_prefix() == actuator.container_prefix


def test_cleanup_removes_worker_containers_only():
    worker = MagicMock()
    worker.name = "proximascale-worker-2-1700000000"
    client = MagicMock()
    client.containers.list.return_value = [worker]
    with patch("cleanup.docker.from_env", return_value=client):
        cleanup.cleanup_environment()
    client.containers.list.assert_called_once_with(
        all=True, filters={"name": cleanup.worker_prefix()})
    worker.remove.assert_called_once_with(force=True)
