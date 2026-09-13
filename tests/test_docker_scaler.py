from unittest.mock import MagicMock, patch
from actuator.docker_scaler import DockerActuator


def _make(monkeypatch, running):
    with patch("actuator.docker_scaler.docker.from_env") as m:
        m.return_value.images.pull.return_value = MagicMock()
        a = DockerActuator({"image": "nginx:alpine",
                            "max_containers": 3,
                            "min_containers": 1})
        a._workers = lambda: [MagicMock() for _ in range(running)]
        return a


def test_scale_up_at_max_returns_false(monkeypatch):
    a = _make(monkeypatch, running=3)
    assert a.scale_up() is False


def test_scale_up_below_max_returns_true(monkeypatch):
    a = _make(monkeypatch, running=2)
    assert a.scale_up() is True


def test_scale_down_at_min_returns_false(monkeypatch):
    a = _make(monkeypatch, running=1)
    assert a.scale_down() is False


def test_scale_down_above_min_returns_true(monkeypatch):
    a = _make(monkeypatch, running=2)
    assert a.scale_down() is True