"""Container metrics collector.

One process, one job: sample the workload container every POLL_INTERVAL
seconds and append the reading to the LIVE metrics file the control loop
reads (monitoring/storage.csv_path() -> metrics_live.csv).

Resilience contract (audit finding C3)
--------------------------------------
A stopped, restarting, paused or missing container used to crash the whole
collector with `KeyError: 'system_cpu_usage'` -- the loop died permanently and
the control loop then silently ran on stale data. The collector now:

  * checks the container's existence and State.Status before sampling,
  * reads every Docker stats field through .get() and validates numbers,
  * distinguishes four different things that all used to look like "0%":
        zero CPU usage             -> cpu_percent = 0.0
        missing CPU measurement    -> no row written, logged as such
        container not running      -> no row written, logged with its state
        Docker/monitoring failure  -> no row written, logged with a traceback
  * wraps each poll in its own try/except, so one bad cycle cannot end the
    process,
  * backs off (bounded) while the container is unavailable instead of
    hammering the Docker API,
  * never invents a CPU number, and never appends a row it could not measure.

Why no row on a failed cycle instead of a sentinel value: the CSV is the
model's input window. A -1/None sentinel would be fed straight into the
feature scaler and then the LSTM. A missing row keeps every sample a real
measurement; the gap is visible as an older newest-timestamp, which is exactly
what main.py's staleness check already looks for.

Request-rate metric definition
------------------------------
`request_rate` in the CSV is REQUESTS PER POLLING WINDOW (the number of
workload requests the app counted since the collector's previous reset), not
requests per second and not a cumulative total. The app reports the window it
measured (`window_seconds`); the collector logs the derived requests/second
alongside it. See docs/METRICS.md.
"""
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime

import docker
import requests
import yaml

from decision.scaling_log import read_scaling_events
from monitoring.metrics import reset_request_count
from monitoring.schema import MetricRecord
from monitoring.storage import append_row, csv_path

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


POLL_INTERVAL = _env_float("PROXIMASCALE_POLL_INTERVAL", 30)
# seconds between each sample -- matches the model's trained sampling rate
# (model/evaluate.py SAMPLING_INTERVAL_SEC, model/prophet_model.py
# DEFAULT_FREQ). Do not drift these apart.

CPU_SAMPLE_GAP = _env_float("PROXIMASCALE_CPU_SAMPLE_GAP", 2)
# seconds between the two samples used for one CPU reading

CONTAINER_NAME = os.getenv("PROXIMASCALE_CONTAINER_NAME", "proximascale-app")

# Where the Flask app exposes the request counter. The app runs inside a
# container, so its SQLite file (data/collected/metrics.db) is NOT shared with
# the host -- reading that file here always returned 0. The HTTP endpoint is
# the correct cross-boundary channel; the file is only a fallback for a
# host-run app.py (Option A in docs/setup_guide.md).
APP_HOST = os.getenv("PROXIMASCALE_APP_HOST", "localhost")
APP_PORT = int(_env_float("PROXIMASCALE_APP_PORT", 5000))

# The old 3 s timeout failed exactly when it mattered: under CPU saturation the
# single-threaded Flask dev server queued the reset behind /work/heavy, the
# reset timed out, and every window read 0 requests (audit C4). The app now
# runs threaded and this timeout is configurable and generous.
REQUEST_RESET_TIMEOUT = _env_float("PROXIMASCALE_REQUEST_RESET_TIMEOUT", 10)
REQUEST_RESET_RETRIES = int(_env_float("PROXIMASCALE_REQUEST_RESET_RETRIES", 2))

# Bounded backoff while the container is unavailable: 2s, 4s, 8s ... capped at
# one poll interval, so a stopped container does not spin the Docker socket.
BACKOFF_BASE_SECONDS = _env_float("PROXIMASCALE_COLLECTOR_BACKOFF", 2.0)

_docker_client = None
_stop_requested = False
_last_request_window = None       # seconds covered by the last request count


class MeasurementUnavailable(Exception):
    """Raised when a metric cannot be measured right now (not a defect).

    The collector treats this as a recoverable condition: log it, write no
    row, back off and try again next cycle.
    """


def _handle_signal(signum, frame):
    global _stop_requested
    _stop_requested = True


def _get_docker_client():
    """Create the Docker client on first use so importing this module
    does not require a running Docker daemon."""
    global _docker_client
    if _docker_client is None:
        _docker_client = docker.from_env()
    return _docker_client


def _reset_docker_client():
    """Drop a cached client after a daemon/transport error so the next cycle
    reconnects instead of reusing a dead connection pool."""
    global _docker_client
    _docker_client = None


def _get_container(require_running: bool = True):
    """The monitored container, validated as present and running.

    Raises MeasurementUnavailable (recoverable) rather than propagating a
    KeyError/NotFound into the poll loop.
    """
    try:
        client = _get_docker_client()
        container = client.containers.get(CONTAINER_NAME)
    except docker.errors.NotFound:
        raise MeasurementUnavailable(f"container '{CONTAINER_NAME}' does not exist")
    except docker.errors.DockerException as e:
        _reset_docker_client()
        raise MeasurementUnavailable(f"Docker daemon unavailable: {e}")

    try:
        container.reload()
    except docker.errors.DockerException as e:
        _reset_docker_client()
        raise MeasurementUnavailable(f"could not refresh container state: {e}")

    state = container.attrs.get("State") or {}
    status = state.get("Status", "unknown")
    if require_running and status != "running":
        detail = state.get("Error") or f"exit={state.get('ExitCode')}"
        raise MeasurementUnavailable(
            f"container '{CONTAINER_NAME}' is not running (state={status}, {detail})"
        )
    return container


def _stats(container):
    try:
        return container.stats(stream=False)
    except docker.errors.DockerException as e:
        _reset_docker_client()
        raise MeasurementUnavailable(f"could not read container stats: {e}")


def get_container_cpu_percent():
    """Container CPU% over CPU_SAMPLE_GAP seconds, from two consecutive stats.

    Why two samples: a single stats(stream=False) call compares total_usage
    against precpu_stats -- Docker's cached snapshot of the PREVIOUS call.
    When the previous call happened long ago (or is the first-ever call),
    precpu is stale/zero and the formula degenerates to 0% or 100%, which is
    exactly the flapping this collector used to log. Sampling twice and
    diffing the two fresh snapshots always yields a real window.

    Returns a float, or raises MeasurementUnavailable.
    """
    container = _get_container()
    stats_a = _stats(container)
    time.sleep(CPU_SAMPLE_GAP)
    if _stop_requested:
        raise MeasurementUnavailable("shutdown requested mid-sample")
    stats_b = _stats(container)

    value = _cpu_percent_from_stats(stats_a, stats_b)
    if value is None:
        raise MeasurementUnavailable(
            "CPU fields missing from the Docker stats payload "
            "(cpu_stats.cpu_usage.total_usage / cpu_stats.system_cpu_usage)"
        )
    return value


def _cpu_percent_from_stats(stats_a, stats_b):
    """CPU% between two stat snapshots. Separated from the I/O above for tests.

    Returns None when the payload does not carry the fields needed for a real
    measurement -- that is NOT the same as 0% usage.
    """
    try:
        cpu_a = stats_a.get("cpu_stats", {}).get("cpu_usage", {}).get("total_usage")
        sys_a = stats_a.get("cpu_stats", {}).get("system_cpu_usage")
        cpu_b = stats_b.get("cpu_stats", {}).get("cpu_usage", {}).get("total_usage")
        sys_b = stats_b.get("cpu_stats", {}).get("system_cpu_usage")
    except AttributeError:
        return None

    if None in (cpu_a, sys_a, cpu_b, sys_b):
        return None
    try:
        cpu_delta = float(cpu_b) - float(cpu_a)
        system_delta = float(sys_b) - float(sys_a)
    except (TypeError, ValueError):
        return None

    if system_delta <= 0 or cpu_delta < 0:
        return 0.0

    cpu_stats = stats_b.get("cpu_stats", {}) or {}
    num_cpus = cpu_stats.get("online_cpus")
    if not num_cpus:
        num_cpus = len(cpu_stats.get("cpu_usage", {}).get("percpu_usage") or []) or 1
    try:
        num_cpus = max(1, int(num_cpus))
    except (TypeError, ValueError):
        num_cpus = 1

    return round(min((cpu_delta / system_delta) * num_cpus * 100, 100.0), 2)


def get_container_memory_percent():
    """Container memory% (usage / limit). Raises MeasurementUnavailable."""
    container = _get_container()
    stats = _stats(container)
    mem = stats.get("memory_stats") or {}
    usage, limit = mem.get("usage"), mem.get("limit")
    if usage is None or not limit:
        raise MeasurementUnavailable("memory_stats.usage/limit missing from stats payload")
    try:
        return round((float(usage) / float(limit)) * 100, 2)
    except (TypeError, ValueError, ZeroDivisionError):
        raise MeasurementUnavailable("memory_stats.usage/limit not numeric")


def collect_request_count():
    """Requests that arrived during the CURRENT poll window, read-and-reset.

    Tries the Flask app's POST /request-rate/reset first (works across the
    container boundary); falls back to the local SQLite counter, which only
    matches the app when app.py runs on the host rather than in a container.

    Returns -1 when neither channel is reachable, so the CSV row visibly
    records 'requests unavailable' instead of a fake zero. Also records the
    measured window length in `_last_request_window` (seconds), so the caller
    can report requests per second without assuming the poll interval held.
    """
    global _last_request_window
    _last_request_window = None
    url = f"http://{APP_HOST}:{APP_PORT}/request-rate/reset"
    last_error = None

    for attempt in range(max(1, REQUEST_RESET_RETRIES)):
        try:
            resp = requests.post(url, timeout=REQUEST_RESET_TIMEOUT)
            if resp.ok:
                payload = resp.json()
                count = int(payload["request_count"])
                try:
                    window = float(payload.get("window_seconds") or 0) or None
                except (TypeError, ValueError):
                    window = None
                _last_request_window = window
                return count
            # 5xx under load is worth one retry; 4xx will not fix itself.
            last_error = f"HTTP {resp.status_code}"
            if resp.status_code < 500:
                break
        except Exception as e:                       # requests/timeout/JSON
            last_error = e
        if attempt + 1 < REQUEST_RESET_RETRIES:
            time.sleep(min(1.0 * (attempt + 1), 3.0))

    logger.warning("reset endpoint unreachable after %d attempt(s) (%s); "
                   "falling back to SQLite", REQUEST_RESET_RETRIES, last_error)

    try:
        return int(reset_request_count())
    except Exception as e:
        logger.warning("SQLite request counter unavailable: %s", e)
        return -1


_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"
)


def _resolve_tick_config_path() -> str:
    """Same config the engine was actually run with."""
    return os.environ.get("PROXIMASCALE_CONFIG_PATH", _CONFIG_PATH)


def _tick_seconds(default: float = 3.0, path: str = None) -> float:
    """Engine tick length; expires_after_steps is counted in these."""
    path = path or _resolve_tick_config_path()
    try:
        with open(path, "r") as f:
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
        window = last["expires_after_steps"] * _tick_seconds(_resolve_tick_config_path())
        return (time.time() - last["timestamp"]) < window
    except (KeyError, TypeError):
        logger.warning("scaling_events.json has malformed entries; "
                       "treating as not post-scaling")
        return False


def _next_sleep(success: bool, elapsed: float, failures: int) -> float:
    """Poll pacing: keep the effective interval on POLL_INTERVAL after a good
    cycle; back off (bounded) while the container is unavailable."""
    if success:
        return max(0.0, POLL_INTERVAL - elapsed)
    return min(POLL_INTERVAL, BACKOFF_BASE_SECONDS * (2 ** min(failures, 6)))


def collect_metrics(max_cycles: int = None):
    """Poll forever (or max_cycles times, for tests), appending one row per
    successful measurement."""
    failures = 0
    cycles = 0

    while not _stop_requested:
        cycles += 1
        cycle_start = time.time()
        wrote_row = False

        try:
            cpu = get_container_cpu_percent()
            memory = get_container_memory_percent()
            # collect_request_count() reads-and-clears the app's counter over
            # the HTTP boundary (SQLite fallback for a host-run app), so each
            # CSV row is the request volume of exactly one poll interval.
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
            # Live rows go to metrics_live.csv. NEVER append to metrics.csv:
            # that file is the frozen 90k synthetic training dataset -- the
            # saved Prophet/LSTM/scaler artifacts were fit on it, and appending
            # real idle rows after the synthetic history makes every window the
            # model sees cross a synthetic-to-live discontinuity.
            append_row(record, path=csv_path())
            wrote_row = True
            failures = 0

            rps = ""
            if request_rate >= 0 and _last_request_window:
                rps = f" ({request_rate / _last_request_window:.2f}/s)"

            print(
                f"{timestamp} | "
                f"Container CPU: {cpu}% | "
                f"Container Memory: {memory}% | "
                f"Requests: {request_rate}{rps} | "
                f"Post-scaling: {post_scaling}"
            )

        except MeasurementUnavailable as e:
            # Expected, recoverable: the workload container is down/absent.
            failures += 1
            logger.warning("collection skipped: %s", e)
            print(f"{datetime.now().isoformat()} | collection skipped: {e}")
        except (StopIteration, GeneratorExit):
            raise
        except Exception:
            # Unexpected: log the traceback (never swallow it) and keep going.
            failures += 1
            logger.exception("unexpected error in collection cycle %d", cycles)
        finally:
            if max_cycles is not None and cycles >= max_cycles:
                break
            time.sleep(_next_sleep(wrote_row, time.time() - cycle_start, failures))

    logger.info("collector stopped after %d cycle(s)", cycles)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    try:
        collect_metrics()
    except KeyboardInterrupt:
        pass
    print("\nCollector stopped gracefully.")
    sys.exit(0)
