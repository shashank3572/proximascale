# ProximaScale — Decision Engine + Actuator — Progress Log

**Branch:** `feature/decision-actuator`  
**Owner:** Person C (Decision Engine + Actuator)  
**Updated by:** Person D (Integration) on handover

---

## Week 9 — Sem 1 (Handover + Completion)

### What was done
- `decision/engine.py` — `evaluate(predicted_cpu, anomaly_flag)` fully implemented.
  Handles: anomaly bypass, cooldown timer, upper/lower threshold logic, min/max replica guards.
  Returns one of: `scale_up`, `scale_down`, `hold`, `hold_cooldown`, `hold_max_reached`, `hold_min_reached`.
- `actuator/docker_scaler.py` — `DockerActuator` implemented with Docker SDK.
  Uses `containers.run()` / `stop()` / `remove()`. Min/max replica bounds enforced.
- `actuator/scaler_interface.py` — Abstract base class. `scale_up`, `scale_down`, `hold`.
- `config.yaml` — Static thresholds (cpu_upper=75, cpu_lower=30), replica limits (min=1, max=5), image name.
- `decision/hysteresis.py` — `Hysteresis` class extracted from engine inline code.
  Encapsulates `last_action_time`, `cooldown_seconds`, `is_cooling_down()`, `record_action()`.
- `main.py` — Wired by Person D. Includes absolute config path, Docker guard on startup,
  signal normaliser (`hold_*` → `hold`), and orchestration loop with dummy predictions.
- `tests/test_decision.py` — 5 pytest test cases. Docker fully mocked. Tests: scale_up, scale_down,
  hold, anomaly bypass, cooldown enforcement.
- `tests/test_integration.py` — Smoke tests: all 4 modules import cleanly, Hysteresis logic verified,
  signal normaliser parametrised across all 6 return values.
- `requirements.txt` — Replaced 103-line pip freeze dump with 10-package minimal file.

### Known limitations (flagged for Sem 1 report)
- Actuator uses `containers.run()` (Docker Desktop, single machine) instead of
  `services.get().scale()` (Docker Swarm, multi-host). This is intentional for Sem 1 demo scope.
  Will be swapped to Swarm API in Sem 2 if moving to multi-host setup.

### What's next (Sem 2)
- `decision/adaptive_threshold.py` — Dynamic threshold using rolling mean + std of recent CPU.
- `decision/cost_model.py` — Cost-aware scaling (conditional, only if ahead of schedule).
- `actuator/k8s_scaler.py` — Kubernetes scaler (optional).
- Integrate Person B's real `predict.py` in `main.py` (replace dummy predictions).

### Blockers
- None for Sem 1. Person B's model not yet integrated — using dummy predictions.

---

## Week 5 — Sem 1

### What was done
- Interface designed: `evaluate(predicted_cpu, anomaly_flag)` agreed with Person D.
- Static threshold logic implemented in `engine.py`.
- Hysteresis (cooldown timer) added inline in `engine.py`.
- `config.yaml` created with threshold values and replica limits.

### Blockers
- None. Docker SDK (`pip install docker`) straightforward to use.
