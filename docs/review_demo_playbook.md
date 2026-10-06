# Review Demo Playbook — "both directions fire, at any hour"

Written 2026-09-22 after a live run proved scale_down could not fire between
10:00 and 17:00 with the previous config. Follow this and you get **both a
scale_up and a scale_down on screen** whether the panel shows up at 10 AM,
11:30, 2 PM, or 3 PM.

---

## 0. The two bugs we found and killed (background you must know)

### Bug A — the "~50% prediction" (not a bug: seasonality, plus one real bug)

1. **Seasonality (expected).** The synthetic training data is diurnal — Prophet
   fits an hourly curve peaking around 15:00. So the Prophet base for 2–3 PM
   is ~55%, and the hybrid mean idles there. Yesterday you ran in the evening
   (base ~30–35%); today at 1:20 PM the base is ~55%. **The prediction moving
   from ~30 to ~50 is the model faithfully following time-of-day.** Never
   present this as drift.
2. **Real bug (fixed).** The residual scaler printed "loaded" but wasn't
   actually applied in the live path — residuals stayed un-denormalised, so
   the LSTM's correction was nearly invisible and the live mean sat ~4–5 points
   above Prophet's own baseline (live probes: 13:50 → 47%, 14:30 → 49%; the
   dashboard simultaneously showed real idle containers at 0–5% CPU).

### Bug B — the scale_down dead zone (the real "it isn't scaling down")

The engine's scale_down test is `predicted_mean < adaptive_lower`, but the
adaptive lower is double-clamped (adaptive_threshold.py line 48):
`lower = max(static_lower, min(ceil_lower, mu ± k·σ))`, and the engine reads
`static_lower` from `cpu_lower_threshold`.

| Config | static_lower | ceil_lower | Dead zone at 14:00–15:00 |
|---|---|---|---|
| Old `config_demo.yaml` | 40 | 45 (default) | idle mean 56 ≥ lower 45 → **scale_down impossible** |
| Real `config.yaml` | 30 | 45 (default) | same: idle 56 ≥ 45 → impossible all afternoon |
| **New `config_demo.yaml`** | **78** | **78** | idle 56 < 78 → **fires on the first eligible poll** |

Offline probe of the exact demo wiring (8-sample window, k=1.5, min_band=10):

| Window content | upper | lower | mean | verdict |
|---|---|---|---|---|
| idle 56.3 ×8 | 61.3 | 78.0 | 56.3 | **scale_down** |
| idle 63 ×8 (worst-case 15:00: base 55 + weekly +8) | 68.0 | 78.0 | 63.0 | **scale_down** |
| idle 45 ×8 (11:00) | 60.0 | 78.0 | 45.0 | **scale_down** |
| mixed post-spike: 100×3 then 56×5 | 90.0 | 78.0 | 72.5 | in_band (good: waits out the spike) |
| full spike: 100 ×8 | 90.0 | 78.0 | 100.0 | scale_up (unaffected) |

`config.yaml` (numbers used for the report) is intentionally untouched.

---

## 1. The fix (already applied)

`config_demo.yaml`:

```yaml
cpu_upper_threshold: 75.0
cpu_lower_threshold: 78.0      # was 40 — see the dead-zone table above
adaptive_floor_lower: 60.0     # defaults (15/45) kept for every non-demo run
adaptive_ceil_lower: 78.0      # adapts the lower band up to 78, not just 45
```

Side effects (all verified, all demo-safe):

- `cpu_low` scale_down fires on the **first poll after cooldown** once idle.
- Mixed post-spike windows (100×3 + 56×5 → mean 72.5) sit **in_band**, so the
  sweep starts only after the spike has actually left the window — clean demo.
- Anomaly branch unaffected (z-score path).
- **One behavioural quirk to rehearse:** with a very wide adaptive band, MC
  Dropout's `upper_bound` can exceed the upper threshold during the spike →
  `upper_bound_risk` scale_up. A legit branch of the engine; explain it as
  "uncertainty-aware preemption" if the panel notices. (A sharper guard is a
  possible follow-up, NOT before the review.)
- A poll *during* the spike can read `hold (min_reached)` — that's the
  min-guard doing its job at 0 workers. Say exactly that if asked.

---

## 2. Pre-flight checklist (5 minutes before the panel)

```powershell
cd D:\proximascale
.\.venv\Scripts\Activate.ps1
docker ps                                   # daemon up; note existing workers
git log --oneline -3                        # you're on dev, clean-ish tree
```

- Close heavy apps (browser tabs with videos, IDE indexers). They inflate
  *real* container metrics during the spike.
- Have this playbook + `docs/viva_prep.md` open on the second screen.
- Know the hour: whatever it is, the same script below works.

## 3. The demo script (10–15 min)

1. **Start clean (t+0).** Remove leftover workers: `docker ps` → if any
   `proximascale-worker-*` exist, `docker rm -f` them. Then:
   `python main.py --config config_demo.yaml`
2. **Explain the idle line (t+0…3 min).** Predictions will read ~45–56 at
   midday (~30–35 in the evening) while `actual` is near 0 — say, verbatim:
   *"Prophet's base follows the training data's diurnal curve — midday is its
   peak, ~55%. The residual head corrects toward the live regime, and the
   anomaly + uncertainty branches don't depend on the mean being right."*
   Expect an early `scale_down` here — that's Bug B's fix working; narrate it
   as "the fleet already consolidated to the minimum."
3. **Fire the spike (t+4).** `docker run --rm proximascale-loadgen:latest`
   (or the stress method you rehearsed). Target several worker-class
   containers near 100% CPU.
4. **Watch the catch (t+4–6).** Expect, in order:
   `anomaly=True → scale_up` (z-score path), then SHAP attribution lines
   (~5 s later: "CPU% dominant"), possibly `upper_bound_risk` scale-ups —
   narrate as uncertainty-aware preemption (Section 1).
5. **Stop the load (t+7).** Kill the load generator.
6. **Watch the sweep (t+7–12).** As the window flushes, reasons walk
   `cooldown` → `cpu_low` and workers go `N → N-1 → … → 0`. With the new
   config the first `cpu_low` appears within ~1–2 polls after cooldown
   expires. This is your scale_down exhibit — point at it.
7. **Leave evidence.** Keep the terminal open; `events.csv` / the scaling log
   rows for the run are the appendix screenshot.

### If something misbehaves on stage
- **No scale_down after 2 min idle:** check `docker ps` — if workers are
  already 0, it's `min_reached` (correct behaviour; say so).
- **Scale-up storm look:** demo cooldown is 60 s on purpose — say
  *"demo config shortens hysteresis so the audience sees actions; production
  config gates it to one action per spike"* (see viva_prep.md).
- **The oneDNN / joblib tracebacks:** cosmetic (TF on Windows; SHAP workers).
  Both are pre-answered in viva_prep.md's rapid-fire card.

## 4. Rehearsal at 14:00, once, before review day

Run Section 3 verbatim at 2 PM the day before. Confirm in the log:
one `scale_up (anomaly)` on spike, at least two `scale_down (cpu_low)`
after stopping, no crashes, SHAP lines appear. If yes — you're done; the
demo is now hour-independent (morning slots only make idle means *lower*,
which the same thresholds still catch: idle 45 → scale_down verified).

## 5. Q&A answers this fix arms you with

- **"Why does the prediction sit at 50% when containers are idle?"**
  Time-of-day seasonality in the training data; Prophet's midday base is ~55%
  (hourly means: 12:00→51, 14:00→55, 15:00→55, 21:00→34). Idle evenings read
  ~30. The decision layer never trusts the mean alone: anomaly (z-score) and
  MC-Dropout uncertainty catch real spikes even when the mean is wrong — the
  live run proved exactly that.
- **"Why doesn't it scale down below one container in the real config?"**
  `min_containers: 1` in production (capacity floor); the demo config sets 0
  so the panel can watch a full collapse. Both are one-line config values,
  same code path.
- **"Why is the lower threshold 78 in the demo but 30 in the report config?"**
  The demo config trades threshold range for stage visibility — with idle
  means near 56, a 30–45 band can never trigger a scale-down in the 10–15 min
  a review lasts. The real config is what produced all report numbers.
- **"Aren't 75/78 thresholds basically 'always act'?"** No: actions are
  gated by cooldown (60 s), the min/max guards, and — between them — the
  adaptive band widens with variance so jittery windows sit in_band. The
  mixed-window row in Section 0's table is the proof: mean 72.5 → in_band.
