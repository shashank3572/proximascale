"""
conftest.py — pytest configuration for ProximaScale tests.
Injects a fake 'docker' module into sys.modules before any test runs,
so tests work on machines where the docker SDK is not installed.
Machines that have docker installed are unaffected.
"""
import sys
from unittest.mock import MagicMock

if 'docker' not in sys.modules:
    fake_docker = MagicMock()
    fake_docker.from_env.return_value = MagicMock()
    sys.modules['docker'] = fake_docker
    sys.modules['docker.errors'] = MagicMock()