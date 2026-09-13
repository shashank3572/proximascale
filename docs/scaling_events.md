## `data/scaling_events.json` — shared interface (Person A ↔ Person C)

**Format change from the master plan:** the file is a **JSON array**, not a
single object. Each successful `scale_up` / `scale_down` decision appends one
entry. `hold` / cooldown / failed-actuator decisions write nothing.

Reason for the change: preserves history so the counterfactual pipeline can
reconstruct "which decisions were active at step N" without losing prior
state. Atomic writes (temp-file + `.replace()`) and a thread lock make
concurrent appends safe.

**Entry schema:**

| key | type | notes |
|-----|------|-------|
| `action` | `"scale_up"` \| `"scale_down"` | never `hold` |
| `timestamp` | `float` | unix epoch, seconds |
| `predicted_cpu` | `float` | the mean prediction at decision time |
| `upper_bound` | `float \| null` | MC-Dropout upper bound, or `null` if B hasn't shipped |
| `reason` | `"anomaly"` \| `"upper_bound_risk"` \| `"cpu_high"` \| `"cpu_low"` | trigger |
| `expires_after_steps` | `int` | cycles the cooldown stays armed (`cooldown_seconds // tick_seconds`) |
| `upper_thresh` | `float` | present on scale-up entries |
| `lower_thresh` | `float` | present on scale-down entries |

**Cleared** at the start of every `main.py` run — the file describes decisions
made in **this** experiment.

**Person A:** build `is_post_scaling()` against the array. Ping Person C if
you need the object-per-line (NDJSON) variant instead, or if you want the
file truncated on each run — both are one-line changes.