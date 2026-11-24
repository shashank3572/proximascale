"""
conftest.py -- pytest configuration for ProximaScale tests.

The docker SDK is always stubbed here, on purpose: the suite must never create,
stop or remove containers on the machine running the tests, whatever Docker SDK
happens to be installed. (Tests that want the real actuator inject a fake
actuator; the Docker integration checks live in docs/DEMO_TROUBLESHOOTING.md and
are run by hand against a live daemon.)

The stub's exception classes are real Python types, not MagicMocks: production
code does `except docker.errors.NotFound`, and catching a MagicMock raises
`TypeError: catching classes that do not inherit from BaseException`.
"""
import sys
from unittest.mock import MagicMock

if "docker" not in sys.modules:
    errors = MagicMock(name="docker.errors")
    docker_exception = type("DockerException", (Exception,), {})
    api_error = type("APIError", (docker_exception,), {})
    errors.DockerException = docker_exception
    errors.APIError = api_error
    errors.NotFound = type("NotFound", (api_error,), {})
    errors.ImageNotFound = type("ImageNotFound", (api_error,), {})
    errors.ContainerError = type("ContainerError", (docker_exception,), {})

    fake_docker = MagicMock(name="docker")
    fake_docker.from_env.return_value = MagicMock()
    fake_docker.errors = errors
    sys.modules["docker"] = fake_docker
    sys.modules["docker.errors"] = errors
