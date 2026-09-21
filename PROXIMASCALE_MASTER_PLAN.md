# PROXIMASCALE — TEAM LEADER MASTER PLAN
### Person D | Shashank B (1AM23CI138) | Repo Owner & Integration Lead

> **Project Title:** ProximaScale: AI-Powered Hybrid LSTM Framework for Proactive Auto-Scaling
> **College:** AMC Engineering College, Bengaluru | CSE-AIML | VTU
> **Guide:** Prof. Vinaya S Kavalgi
> **Status:** Integration + debugging. 2-day review crunch. Documentation starting now.

---

## VIVA ONE-LINER (everyone memorises — no exceptions)

> *"ProximaScale is an AI-driven proactive auto-scaling system that uses a multivariate residual-stacking hybrid LSTM model with multi-step prediction, MC Dropout uncertainty quantification, adaptive thresholding, anomaly detection, and counterfactual-corrected inference to forecast workloads and scale Docker containers before demand spikes occur — with real-time SHAP-based explainability of every scaling decision."*

---

## TEAM

| Person | Name | USN | Role |
|--------|------|-----|------|
| A | Shravani V | 1AM23CI140 | App Layer + Monitoring + Data |
| B | Shreyas R | 1AM23CI141 | LSTM + Residual Stacking Model + Novelty Core |
| C | Soubhagya Ranjan Behera | 1AM23CI148 | Decision Engine + Docker Actuator |
| D | Shashank B (me) | 1AM23CI138 | Integration + Dashboard + SHAP + Repo Owner |

---

## LOCKED ARCHITECTURE (updated — residual stacking, not ensemble)

```
Locust (synthetic traffic — 90k rows, normal/ramp/spike patterns)
        |
Flask App in Docker (--cpus=0.5 --memory=256m cgroup limits)
        |  [Container CPU%, Memory%, Req/min via Docker SDK]
Monitoring Module
        |  [10-step sliding window + post_scaling tag]
Preprocessing (MinMaxScaler — fit on train only)
        |
   LSTM Model
   (multivariate: CPU + memory + req rate, 3-step output)
        |
   Initial Prediction
        |
   Residual Computation
   (actual - predicted on training data)
        |
   Residual Correction Model
   (learns to correct LSTM's systematic errors)
        |
   Final Corrected Prediction
        |
   + MC Dropout uncertainty (30-pass, mean + 2sigma = upper_bound)
   + Anomaly Detection (Z-score flag)
   + Counterfactual Correction (post_scaling window adjustment)
        |
   predict_load() returns (predicted_cpu, upper_bound, anomaly_flag)
        |
   Decision Engine
   (adaptive threshold + hysteresis + risk-aware upper_bound logic)
        |
   scale_up / hold / scale_down
        |
   Docker Actuator (SDK — real container start/stop)
   1-10 replicas, each constrained to 0.5 vCPU / 256MB
        |
   <- Proactive feedback loop (scaling_events.json) <-
        |
   SHAP Explainability Dashboard (Streamlit)
```

---

## WHAT "HYBRID LSTM" MEANS NOW (residual stacking)

**Old (abandoned):** weighted average of LSTM + Prophet outputs.
Reason dropped: ensemble gave worse results than standalone multivariate LSTM.

**Current (locked):** Residual stacking.
1. Prophet generates the initial CPU prediction (base model).
2. Residual = actual CPU - Prophet prediction (computed on training data).
3. The LSTM is trained on those residuals.
4. Final prediction = Prophet output + LSTM residual correction.

The LSTM specifically targets WHERE the Prophet base model fails, not an average of two independent models.
Paper-backed from our references list.

**Viva answer:**
> "Our hybrid architecture uses residual stacking. The Prophet model generates the initial workload prediction — the base signal. The residual — the difference between that Prophet forecast and actual historical values — is then passed to the LSTM, which learns to correct Prophet's systematic errors. The final prediction is Prophet forecast plus the LSTM residual prediction. This outperforms both standalone Prophet and weighted ensemble averaging because the LSTM specifically targets Prophet's failure modes rather than blending two independent predictions."

## NOVELTY STACK (updated)

| # | Novelty | Owner | Backing | Priority |
|---|---------|-------|---------|----------|
| 1 | Residual stacking hybrid LSTM | Person B | Paper in references | CORE |
| 2 | Multivariate input (CPU + memory + req rate) | Person B | Paper [1] — 1.84x RMSE | CORE |
| 3 | Multi-step prediction (3 steps = 90 seconds ahead) | Person B | Paper [3] | CORE |
| 4 | MC Dropout uncertainty quantification | Person B | Not in cited papers — original | PRIMARY |
| 5 | Counterfactual-aware inference pipeline | A + B + C | Not in cited papers — original | PRIMARY |
| 6 | Anomaly detection Z-score flag | Person B | IEEE HPCC 2025 direction | CORE |
| 7 | Adaptive threshold (rolling window) | Person C | Documented research gap | SUPPORTING |
| 8 | SHAP explainability dashboard | Person D | Not in autoscaling papers | SUPPORTING |

**Dropped:** Prophet ensemble — produced worse results, replaced by residual stacking.

---

## SYNTHETIC DATA — HOW TO EXPLAIN IT

~90k rows generated. Real container data not collected due to hardware environment constraints.

**Viva framing (use this exactly):**
> "We generated a realistic synthetic workload dataset of approximately 90,000 timesteps using parameterised traffic simulation — covering normal steady load, gradual ramps, and sudden spike events — to ensure sufficient training data for temporal pattern learning. Synthetic workload generation is standard practice in autoscaling research where live cluster traces are unavailable, and is used in multiple papers in our literature survey including the Google Cluster and Alibaba trace benchmarks which are themselves replay datasets."

Do not apologise for synthetic data. It's fine. Move on.

---

## LOCKED FEATURE SET

### CORE — must be complete
- [ ] Flask app + Docker with --cpus=0.5 --memory=256m
- [ ] Container-level monitoring via Docker SDK (not host psutil)
- [ ] Locust traffic scenarios (normal, spike, ramp)
- [ ] post_scaling tag in metric records
- [ ] Multivariate LSTM trained and saved (.h5 + scaler.pkl)
- [ ] Residual correction model trained and saved
- [ ] predict_load() returns (predicted_cpu, upper_bound, anomaly_flag)
- [ ] MC Dropout 30-pass inference producing upper_bound
- [ ] Anomaly detection Z-score flag
- [ ] Counterfactual correction on post-scaling windows
- [ ] Decision engine (adaptive threshold + hysteresis)
- [ ] scaling_events.json written after every scale action
- [ ] Docker SDK actuator (real container start/stop, not just logs)
- [ ] main.py end-to-end loop running without errors
- [ ] Reactive baseline for comparison
- [ ] 4-way comparison results table with real RMSE/MAE numbers

### SUPPORTING
- [ ] Streamlit dashboard (actual vs predicted + confidence band + event markers)
- [ ] SHAP explainability panel
- [ ] Scaling lead time measurement
- [ ] Final report

### REMOVED PERMANENTLY
- [x] Prophet ensemble — dropped, gave worse results
- [x] stress.py — replaced by Docker cgroup limits
- [x] Kubernetes — too risky for demo
- [x] RL decision engine — out of scope
- [x] Multi-service scaling — out of scope

---

## MODULE INTERFACES (locked — do not change without telling Person D)

### Person B → Everyone
```python
predict_load(window: list[dict]) -> tuple[float, float, bool]
# (predicted_cpu, upper_bound, anomaly_flag)
# upper_bound = mean + 2*std from MC Dropout
```

### Person C → Person D
```python
evaluate(
    predicted_cpu: float,
    upper_bound: float,
    anomaly_flag: bool
) -> str  # "scale_up" | "scale_down" | "hold"
```

### Person C → Person A (filesystem)
```json
data/scaling_events.json
{"action": "scale_up", "timestamp": 1234567890.0, "expires_after_steps": 5}
```

---

## DATA SCHEMA (shared — never change without team agreement)

```json
{
  "timestamp": "2024-01-15T14:32:00",
  "cpu_percent": 67.4,
  "memory_percent": 52.1,
  "request_rate": 143,
  "post_scaling": false
}
```

- Stored: data/collected/metrics.csv (~90k rows synthetic)
- Poll: 60s | Window: 10 steps = 10 min | Prediction: 3 steps = 3 min ahead
- Container: proximascale-app at --cpus=0.5 --memory=256m

---

## PERSON A — SHRAVANI V | Checklist

**Branch:** feature/app-monitoring | **Commit:** [monitoring] description

### Files
```
app/app.py, app/Dockerfile
monitoring/collector.py   <- Docker SDK container stats
monitoring/storage.py
monitoring/schema.py      <- post_scaling field
data/locust_scenarios/normal_load.py
data/locust_scenarios/spike_load.py
data/locust_scenarios/gradual_ramp.py
```

### Checklist
- [ ] Flask endpoints work: /work/light, /work/heavy, /health
- [ ] Docker runs with --cpus=0.5 --memory=256m applied
- [ ] collector.py reads CONTAINER CPU via Docker SDK (not host psutil)
- [ ] collector.py reads CONTAINER memory via Docker SDK
- [ ] Metrics CSV saves correctly with all columns including post_scaling
- [ ] is_post_scaling() reads scaling_events.json correctly
- [ ] post_scaling=True stamps for 5 cycles after scaling event
- [ ] CPU spikes to 70%+ under spike_load.py (verified, not estimated)
- [ ] Locust master/worker tested on local network

### Viva questions owned
1. "What metrics do you monitor?" — container CPU/memory/req rate via Docker SDK
2. "Why Docker cgroup limits?" — simulates AWS t2.micro, real throttling
3. "How did you generate training data?" — 90k rows synthetic, parameterised patterns
4. "What is the post-scaling tag?" — feeds counterfactual correction in Person B

---

## PERSON B — SHREYAS R | Checklist

**Branch:** feature/lstm-model | **Commit:** [model] description
**Hardest role. 4-way comparison table is the most important deliverable.**

### Files
```
model/lstm_model.py, model/train.py
model/predict.py            <- STABLE interface
model/preprocessing.py
model/anomaly.py
model/evaluate.py           <- 4-way comparison table
model/residual_model.py     <- residual correction model
model/uncertainty.py        <- MC Dropout inference
model/counterfactual.py     <- post-scaling correction
model/saved/                <- .h5 + scaler.pkl (both committed)
```

### Checklist
- [ ] LSTM architecture: LSTM(64)->Dropout->LSTM(32)->Dropout->Dense(3)
- [ ] Scaler fitted on TRAIN data only — NOT re-fitted at inference
- [ ] LSTM trained on 90k row synthetic CSV — loss converges
- [ ] Residual model trained on (actual - LSTM prediction) pairs
- [ ] predict_load() returns (predicted_cpu, upper_bound, anomaly_flag)
- [ ] MC Dropout: 30 inference passes with training=True, returns mean+2σ
- [ ] Counterfactual correction applied on post_scaling=True windows
- [ ] Anomaly detection Z-score threshold=2.5 returns bool
- [ ] model.h5 AND scaler.pkl both committed to model/saved/
- [ ] evaluate.py produces 4-way table with real numbers

### 4-Way Comparison Table (fill with real numbers)
| Model | RMSE | MAE |
|-------|------|-----|
| Reactive baseline | N/A | N/A |
| Univariate LSTM | | |
| Multivariate LSTM | | |
| Residual Stacking Hybrid | | |

### Viva questions owned
1. "What is hybrid about your LSTM?" — residual stacking explanation above
2. "What does the model predict?" — CPU%, memory%, req rate for next 3 min
3. "What is MC Dropout?" — 30-pass inference, uncertainty distribution, scale on upper bound
4. "What is counterfactual correction?" — prevents artificial CPU drop from misleading model
5. "How do you know it works?" — 4-way comparison table with RMSE/MAE
6. "What if prediction is wrong?" — MC Dropout upper bound + Z-score anomaly fallback

---

## PERSON C — SOUBHAGYA RANJAN BEHERA | Checklist

**Branch:** feature/decision-actuator | **Commit:** [decision]/[actuator] description
**CRITICAL: Main repo only. Never separate repo. Never push to main directly.**

### Files
```
decision/engine.py
decision/adaptive_threshold.py
decision/hysteresis.py
actuator/docker_scaler.py   <- real SDK + scaling_events.json
actuator/scaler_interface.py
```

### Checklist
- [ ] evaluate() accepts (predicted_cpu, upper_bound, anomaly_flag) — 3 params
- [ ] Decision order correct:
  - [ ] 1. anomaly_flag=True → immediate scale_up
  - [ ] 2. upper_bound > threshold → scale_up (risk-aware)
  - [ ] 3. cooldown active → hold
  - [ ] 4. predicted_cpu > adaptive_upper → scale_up
  - [ ] 5. predicted_cpu < adaptive_lower → scale_down
  - [ ] 6. else → hold
- [ ] Hysteresis: real time-based 180s cooldown (not a flag)
- [ ] Docker actuator calls SDK — actually starts/stops containers
- [ ] New containers start with --cpus=0.5 --memory=256m
- [ ] Min=1, Max=10 replicas enforced
- [ ] scaling_events.json written after EVERY scale_up and scale_down
- [ ] Adaptive threshold uses rolling window (not static 70%)
- [ ] All code in main repo feature/decision-actuator branch

### Viva questions owned
1. "How does prediction cause actual scaling?" — full flow predict_load → evaluate → Docker SDK
2. "What is adaptive threshold?" — rolling window, dynamic, documented research gap
3. "What prevents oscillation?" — hysteresis 3-min cooldown + counterfactual correction
4. "Where is proactive?" — scaling fires on predicted CPU, not current CPU
5. "What is the scaling event log?" — feeds post_scaling tag → counterfactual pipeline

---

## PERSON D — SHASHANK B (ME) | Checklist

**Branch:** feature/integration-dashboard | **Commit:** [integration]/[dashboard] description

### Files
```
main.py
baseline/reactive_scaler.py
dashboard/live_plot.py
dashboard/shap_explain.py
dashboard/results_charts.py
tests/test_integration.py
docs/data_schema.md
docs/setup_guide.md
PROGRESS.md, requirements.txt, README.md
.github/workflows/ci.yml
.github/pull_request_template.md
```

### Repo owner duties
- [ ] All branches: feature/* only, nobody on main
- [ ] CI running (pytest + flake8 on every PR)
- [ ] feature/app-monitoring merged into dev last (most diverged)
- [ ] PROGRESS.md current

### Integration checklist
- [ ] main.py updated for 3-return predict_load: (predicted_cpu, upper_bound, anomaly_flag)
- [ ] main.py updated for 3-param evaluate: (predicted_cpu, upper_bound, anomaly_flag)
- [ ] End-to-end pipeline runs without errors (even with dummy data)
- [ ] Integration tests pass: pytest tests/ -v
- [ ] reactive_scaler.py baseline built (static threshold, no ML)

### Dashboard checklist
- [ ] Actual CPU vs predicted CPU line chart
- [ ] Confidence band (shaded region between mean and upper_bound)
- [ ] Scaling event markers (▲ scale_up, ▼ scale_down)
- [ ] Anomaly flag markers (red dots)
- [ ] Current replica count metric card
- [ ] SHAP feature attribution bar chart (updates on scale_up)
- [ ] Scaling lead time counter

### Experiment tracker

**Experiment 1 — Prediction accuracy** (from Person B's evaluate.py)
| Model | RMSE | MAE |
|-------|------|-----|
| Reactive | N/A | N/A |
| Univariate LSTM | | |
| Multivariate LSTM | | |
| Residual Stacking Hybrid | | |

**Experiment 2 — Scaling lead time** (run 5 times, record seconds before threshold breach)
| Run | Lead Time (s) |
|-----|---------------|
| 1 | |
| 2 | |
| 3 | |
| 4 | |
| 5 | |
| Mean | Target: ≥60s |

**Experiment 3 — Response latency under spike**
| Condition | Avg Latency (ms) | P95 (ms) |
|-----------|-----------------|----------|
| No scaling | | |
| Reactive | | |
| ProximaScale | | |

**Experiment 4 — Oscillation count (proves counterfactual)**
| Config | scale_up within 5min of scale_down |
|--------|-------------------------------------|
| Without counterfactual | |
| With counterfactual | |

---

## GIT WORKFLOW

```
main     <- demo-ready, never push directly
dev      <- all PRs merge here
feature/app-monitoring          (Person A — merge last)
feature/lstm-model              (Person B — merged)
feature/decision-actuator       (Person C — merged)
feature/integration-dashboard   (Person D — merged)
```

| Rule | |
|------|-|
| Commits | [module] short description |
| PRs | into dev only, Person D reviews |
| main | Person D merges at milestones only |
| Bugs | GitHub Issues, not group chat |
| Fridays | everyone updates PROGRESS.md |

---

## 2-DAY CRUNCH PLAN

### Today (Day 1) — Integration first, docs second
**Morning:**
- [ ] Verify predict_load() return types match main.py expectations
- [ ] Verify evaluate() accepts 3 params everywhere it's called
- [ ] Merge feature/app-monitoring into dev
- [ ] Run full pipeline once end-to-end — any output is fine

**Afternoon:**
- [ ] Run evaluate.py → get 4-way RMSE/MAE numbers
- [ ] Run Experiment 3 (latency) — Locust reports these automatically
- [ ] Start documentation: abstract + architecture section

**Evening:**
- [ ] Dashboard: confirm actual vs predicted chart renders
- [ ] SHAP: get it working even if not perfectly styled
- [ ] Documentation: problem statement + dataset section

### Tomorrow (Day 2) — Polish + review prep
**Morning:**
- [ ] Run Experiment 2 (lead time) — 5 spike runs
- [ ] Fill all experiment tables with real numbers
- [ ] Documentation: methodology + results section

**Afternoon:**
- [ ] Results charts for report
- [ ] Demo rehearsal — full 5-7 minute sequence once
- [ ] Everyone practices their viva Q&A

**Evening:**
- [ ] Buffer — fix whatever broke
- [ ] Pre-record backup demo video
- [ ] NO new features after this point

---

## DOCUMENTATION STRUCTURE (write in this order)

1. **Abstract** (150 words) — problem, approach, results, contribution
2. **Introduction** — reactive vs proactive, why AI, scope
3. **Literature Survey** — 6 papers, what each contributes, gap your project fills
4. **System Architecture** — pipeline diagram + component descriptions
5. **Methodology**
   - 5.1 Data generation (synthetic 90k rows, patterns covered)
   - 5.2 Preprocessing (MinMaxScaler, sliding window, train/test split)
   - 5.3 LSTM model (architecture, training)
   - 5.4 Residual stacking (what residual is, correction model, why it works)
   - 5.5 MC Dropout uncertainty quantification
   - 5.6 Anomaly detection
   - 5.7 Counterfactual correction
   - 5.8 Decision engine (adaptive threshold, hysteresis)
   - 5.9 Docker actuator
   - 5.10 SHAP explainability
6. **Experiments and Results** — 4 experiments, tables, charts
7. **Discussion** — what the results mean, why residual stacking won
8. **Limitations** — synthetic data, local Docker, training data requirements
9. **Conclusion and Future Work**
10. **References** — all 6 papers with DOIs

**Abstract template (fill in your numbers):**
> Traditional cloud auto-scaling systems react to resource exhaustion after it occurs, causing latency spikes and SLA violations. ProximaScale addresses this through proactive scaling driven by a residual-stacking hybrid LSTM model that forecasts CPU utilisation, memory usage, and request rate 1–3 minutes ahead of actual demand. The system integrates MC Dropout uncertainty quantification, Z-score anomaly detection, counterfactual-corrected inference, and adaptive thresholding within an automated Docker scaling pipeline. Evaluated on a synthetic workload of 90,000 timesteps, ProximaScale's residual-stacking hybrid achieves an RMSE of [X] and MAE of [Y], outperforming standalone multivariate LSTM by [Z]%. Under simulated traffic spikes, ProximaScale triggers scaling [N] seconds before threshold breach, reducing average response latency from [A]ms to [B]ms compared to reactive threshold-based scaling. Real-time SHAP explainability makes scaling decisions transparent and auditable.

---

## DEMO SEQUENCE (5-7 minutes)

| Time | Action | What to say |
|------|--------|-------------|
| 0:00 | Open Streamlit dashboard | "This is our live monitoring — actual CPU in blue, predicted in orange, confidence band shaded" |
| 0:30 | Trigger Locust spike | "We're simulating a traffic surge — 30 concurrent users on the heavy endpoint" |
| 1:30 | CPU climbs | "Container CPU is rising — monitored via Docker SDK, not host metrics" |
| 2:30 | scale_up fires (▲ appears) | "Scale-up just fired — notice CPU hasn't hit 70% yet. That's proactive scaling." |
| 3:00 | SHAP bar chart updates | "SHAP tells us why — request rate drove 60% of this decision" |
| 3:30 | Replica: 1 → 2 | "Second container is running. Load distributes. CPU stabilises." |
| 4:00 | Show lead time number | "We scaled [X] seconds before threshold breach. Reactive systems wait until after." |
| 4:30 | Show results table | "Across experiments — RMSE of [X] on our hybrid vs [Y] for standalone LSTM" |
| 5:00 | Q&A | Each person owns their module |

**Always have pre-recorded backup video. Demo on local network only.**

---

## PANEL Q&A — LOCKED ANSWERS

**"What is your project?"**
ProximaScale predicts future server load using AI and scales Docker containers before traffic spikes arrive — not after performance has degraded.

**"Why LSTM?"**
Workload is a time-series. The last 10 minutes directly influences the next 3. LSTM captures these non-linear temporal dependencies. A static threshold cannot.

**"What is hybrid about your LSTM?"**
Residual stacking. LSTM generates an initial prediction. A second correction model learns the residual — where the LSTM was systematically wrong. Final prediction is LSTM output plus the learned correction. Outperforms both standalone LSTM and weighted ensemble.

**"What does the model predict?"**
CPU%, memory%, and request rate for the next 3 minutes, from the last 10 minutes of container metrics.

**"Why synthetic data?"**
Synthetic workload generation is standard in autoscaling research — Google Cluster and Alibaba traces are themselves replay datasets. We generated 90,000 timesteps covering normal load, ramps, and spikes to ensure sufficient temporal pattern diversity.

**"How does prediction cause actual scaling?"**
Decision engine receives predicted CPU and upper confidence bound. If predicted CPU exceeds adaptive threshold — or if upper bound is risky even when mean is safe — the Docker SDK starts a new constrained container. Containers are ready before traffic peaks.

**"What happens if prediction is wrong?"**
Two safety nets: MC Dropout catches uncertain high-load scenarios by scaling on the upper confidence bound. Z-score anomaly detection catches unexpected spikes by bypassing prediction entirely.

**"What is counterfactual correction?"**
When we scale up, CPU drops artificially. Without correction, LSTM sees the drop and predicts low future load — causing premature scale-down. We tag post-scaling windows and blend those CPU readings toward pre-scaling levels before inference. Not in any cited paper.

**"What is MC Dropout?"**
30-pass inference with dropout active. Gives a distribution over predictions. We scale when the upper bound (mean + 2σ) is risky — even if the mean seems safe. Risk-aware, not point-estimate-dependent.

**"What is SHAP?"**
SHapley Additive exPlanations. Attributes each scaling decision to input features — what percentage was driven by request rate, CPU, or memory. Makes the AI decision auditable.

**"How is this better?"**
Reactive triggered [X]s after threshold breach. ProximaScale triggered [Y]s before. Response latency dropped [Z]%. Measured results, not claims.

**"What is your novelty?"**
Four: residual stacking hybrid (paper-backed), multivariate input (paper-backed, 1.84x RMSE), MC Dropout for autoscaling (original), counterfactual correction (original). Last two not in any cited paper.

**"What are your limitations?"**
Synthetic workload. Prophet ensemble abandoned — gave worse results. Local Docker not real cloud cluster. Generalisation to production-scale patterns requires real trace data.

---

## REAL-WORLD APPLICATIONS

| Domain | Application |
|--------|-------------|
| E-commerce (Flipkart, Amazon) | Predict checkout surge before flash sale hits |
| Food delivery (Swiggy, Zomato) | Capture meal-time peaks + rain/event irregular spikes |
| Live streaming (Hotstar, JioTV) | Scale before event start, not during buffering |
| Banking/UPI (NPCI, Paytm) | Salary-date peaks + crisis transaction surges |
| Healthcare (Practo, hospitals) | Never-seen emergency surges caught by anomaly detection |
| Gaming (multiplayer servers) | Scheduled tournament + irregular join curve |
| SaaS platforms | Weekly patterns + batch job spikes |
| Cloud cost optimisation | Counterfactual prevents wasteful over-provisioning |

---

## RISK REGISTER

| Risk | Mitigation |
|------|-----------|
| LSTM loss doesn't converge | Reduce LR, increase epochs, verify input shape (1,10,3) |
| Residual model performs worse | Report honestly; still novel architecture; note in limitations |
| Container CPU never spikes | Verify Docker SDK reading; reduce --cpus to 0.3 |
| predict_load() signature mismatch | First check on Day 1 — fix before anything else |
| End-to-end crashes during demo | Pre-recorded backup video mandatory |
| scaling_events.json not written | Test explicitly after every scale action |
| SHAP incompatible with LSTM | Fall back to gradient-based importance |
| College WiFi fails | Single-laptop fallback always prepared and tested |
| Panel asks about Prophet | "We evaluated ensemble — it performed worse than standalone LSTM. Residual stacking gave better results and is theoretically sounder." |

---

## WEEKLY LOG TEMPLATE (copy to PROGRESS.md every Friday)

```markdown
## Week [N] — [Date]

Person A (Shravani):
Done:
Next:
Blocked:

Person B (Shreyas):
Done:
Next:
Blocked:

Person C (Soubhagya):
Done:
Next:
Blocked:

Person D (Shashank):
Done:
Next:
Blocked:

PRs merged:
PRs open:
Blockers for team:
```

---

## FINAL SUBMISSION CHECKLIST

### Title alignment
- [ ] AI-Powered: LSTM + residual model + MC Dropout + SHAP all running
- [ ] Hybrid LSTM: residual stacking implemented and evaluated
- [ ] Framework: full 6-stage pipeline in main.py
- [ ] Proactive: scale fires before CPU breach (lead time experiment proves it)
- [ ] Auto-Scaling: Docker SDK adjusts replicas automatically

### Technical
- [ ] 4-way comparison table: real RMSE/MAE numbers
- [ ] Lead time experiment: mean ≥60s
- [ ] Latency experiment: ProximaScale < Reactive
- [ ] Oscillation experiment: counterfactual reduces count
- [ ] pytest passes
- [ ] End-to-end tested under spike scenario

### Demo
- [ ] Runs 3x without errors
- [ ] Backup video recorded
- [ ] Single-laptop fallback tested

### Report
- [ ] All 10 sections written
- [ ] Tables have real numbers
- [ ] Architecture diagram matches implementation
- [ ] Limitations honest
- [ ] References with DOIs

### Viva
- [ ] Everyone memorised the one-liner
- [ ] Each person owns their Q&A
- [ ] Everyone can answer "what is novelty?"
- [ ] Demo rehearsed together twice

---

*Owner: Shashank B (1AM23CI138) — Person D*
*Single source of truth. Architecture, interfaces, feature set are locked.*
*Last updated: 2-day review crunch — residual stacking architecture.*
