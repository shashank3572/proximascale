"""Manuscript content, part 1: front matter through Section IV.

Every number quoted in prose below is sourced from a file named in the
repository validation report (results/). Tables are generated from the CSVs at
build time by build_paper.py, not typed by hand.
"""

TITLE = ("ProximaScale: An Integrated Uncertainty-Aware and Feedback-Driven "
         "Framework for Proactive Auto-Scaling of Cloud Applications")

AUTHORS = [
    "Prof. Vinaya S Kavalgi, Assistant Professor, Department of Computer Science (AI & ML), "
    "AMC Engineering College, Bengaluru, India — vinaya.kavalgi@amceducation.in",
    "Shreyas R, Department of Computer Science (AI & ML), AMC Engineering College, Bengaluru, India — 1am23ci141@amceducation.in",
    "Shashank B, Department of Computer Science (AI & ML), AMC Engineering College, Bengaluru, India — 1am23ci138@amceducation.in",
    "Shravani V, Department of Computer Science (AI & ML), AMC Engineering College, Bengaluru, India — 1am23ci140@amceducation.in",
    "Soubhagya Ranjan Behera, Department of Computer Science (AI & ML), AMC Engineering College, Bengaluru, India — 1am23ci148@amceducation.in",
]

ABSTRACT = (
    "Cloud applications experience dynamic and unpredictable workloads, which makes reactive "
    "auto-scaling prone to acting only after resource demand has already risen. This paper presents "
    "ProximaScale, a proactive auto-scaling framework for containerized applications that combines "
    "statistical and neural workload forecasting with an explicit, auditable scaling policy. The "
    "served forecaster is a residual-stacking model: Facebook Prophet models trend and seasonality of "
    "container CPU utilization, and a two-layer Long Short-Term Memory (LSTM) network, fed with a "
    "30-reading window of corrected CPU, memory, request count and two Prophet channels, predicts "
    "Prophet's residual for the next three 30-second steps. Three independently seeded LSTMs are "
    "pooled into a Monte Carlo dropout ensemble of 30 stochastic passes per decision. The point "
    "forecast is re-anchored on the measured CPU level, and a split-conformal, sigma-binned upper "
    "bound provides a one-sided risk estimate. A decision engine combines an adaptive threshold band, "
    "a cooldown gate and scale-down guards to issue scale-up, hold and scale-down actions, which a "
    "Docker actuator executes on a worker pool. On a 90,000-row, 31-day synthetic workload with 36 "
    "generated spikes, the served model attains an RMSE of 4.35 and an MAE of 2.68 CPU percentage "
    "points at a 90-second horizon, against 10.15/4.68 for Prophet alone, 5.13/3.03 for a univariate "
    "LSTM and 6.64/3.92 for a reactive persistence signal; the improvement is significant under a "
    "Diebold-Mariano test (p < 1e-9). Mean lead time over spike onsets is 58.2 s against a feasible "
    "ceiling of 58.3 s, and 33.8 of 36 spikes are anticipated 90 s ahead under the forecast-mean "
    "policy with no false scale-up episodes per day, while the shipped configuration, which enables a "
    "reactive floor, anticipates 24 of 36. We also report negative results that qualify the framework's claims: raw "
    "Monte Carlo dropout variance is strongly miscalibrated (mean + 2 sigma covers 55-63% at nominal "
    "84-99.9%), the configured counterfactual blend weight is not optimal on held-out post-scaling "
    "rows, and two internal evaluation harnesses disagree on decision-level outcomes. The study "
    "therefore establishes forecast-quality and lead-time benefits on synthetic data, while leaving "
    "SLO-level and real-traffic benefits unestablished."
)

KEYWORDS = ("proactive auto-scaling, cloud applications, workload forecasting, residual stacking, "
            "LSTM, Prophet, uncertainty calibration, feedback control, Docker")

BLOCKS_1 = [
    ("h1", "I. Introduction"),
    ("p",
     "Cloud applications operate under continuously changing workloads, and variations in user demand "
     "translate into fluctuating computational resource requirements. Auto-scaling addresses this by "
     "adjusting capacity according to workload conditions. Conventional reactive mechanisms, including "
     "threshold rules and the Kubernetes Horizontal Pod Autoscaler, act on current or recently observed "
     "utilization. A scaling action therefore begins only after a workload increase has already "
     "produced elevated resource consumption, and container start-up and readiness latency is spent "
     "inside the overload interval rather than before it."),
    ("p",
     "Predictive auto-scaling attempts to remove that delay by forecasting future workload and acting "
     "before the predicted demand arrives. Statistical models such as Prophet capture trend and "
     "seasonality; recurrent networks capture short-range non-linear dynamics. Forecasts are, however, "
     "uncertain, and the uncertainty is not uniform: a model that is accurate under regular diurnal "
     "load may be least reliable precisely at abrupt transitions. A scaling policy that treats every "
     "forecast as equally trustworthy can therefore provision too early, too late, or repeatedly."),
    ("p",
     "Prediction alone is also insufficient as a control strategy. A deployed autoscaler must observe "
     "what happened after an action, because the action itself changes the measured signal: adding "
     "capacity depresses CPU utilization, so post-scaling observations are a mixture of workload and "
     "intervention. If such observations are used naively for training or for the next decision, the "
     "controller learns to under-react to the spikes it exists to absorb. This motivates a closed "
     "monitoring-decision-actuation loop with explicit treatment of post-scaling data."),
    ("p",
     "This paper reports the current state of ProximaScale, an implementation-led study rather than a "
     "proposal. The repository contains a working pipeline: a Flask workload container measured through "
     "the Docker API, a collector that separates \"no CPU used\" from \"could not measure\", a "
     "residual-stacking forecaster served from a versioned model bundle, a decision engine with "
     "adaptive thresholds and hysteresis, a Docker actuator, a decision log, and a Streamlit dashboard. "
     "The paper is written against that implementation and against results that the repository can "
     "regenerate; where earlier drafts of this work reported figures that the current code does not "
     "reproduce, the reproduced figures are used and the discrepancy is stated."),
    ("p", "The contributions of this version are:"),
    ("bullets", [
        "A residual-stacking forecaster in which Prophet's value at each historical reading and 90 s "
        "ahead are supplied to the LSTM as input channels, so the network models the residual "
        "conditional on the statistical forecast rather than forecasting CPU in isolation.",
        "A served uncertainty path with two distinct mechanisms, reported separately: a pooled "
        "Monte Carlo dropout ensemble over three seeded LSTMs, and a split-conformal, sigma-binned "
        "one-sided bound calibrated on a validation period.",
        "A decision layer whose components are ablated on a 17,988-decision replay: adaptive band, "
        "cooldown, reactive floor, anomaly bypass, upper-bound branch, and scale-down guard.",
        "An empirical evaluation on a frozen 90,000-row synthetic workload with a generator-derived "
        "ground truth, including lead-time analysis against a computed feasibility ceiling, "
        "Diebold-Mariano tests with moving-block bootstrap intervals, and out-of-distribution tests on "
        "three unseen one-week workloads plus the only real container data in the repository.",
        "A candid account of negative and inconclusive findings: miscalibrated raw dropout variance, "
        "a suboptimal counterfactual weight, disagreement between two evaluation harnesses, and the "
        "absence of any measured end-to-end latency benefit because the actuator operates in pool "
        "mode without a load balancer.",
    ]),
    ("p",
     "Section II reviews related work. Section III describes the deployed architecture, Section IV the "
     "forecasting model and its calibration, Section V the experimental protocol, Section VI the "
     "results, and Sections VII-IX the discussion, threats to validity, and conclusions."),

    ("h1", "II. Related Work"),
    ("p",
     "Research on cloud auto-scaling has moved from reactive rules toward predictive and learned "
     "policies. The studies most relevant to ProximaScale fall into four groups."),
    ("h2", "A. Deep Learning-Based Auto-Scaling"),
    ("p",
     "Dang-Quang and Yoo proposed a multivariate auto-scaling framework built on a bidirectional LSTM "
     "within a monitor-analyze-plan-execute loop [1]. Using several workload variables rather than a "
     "single request-rate series, their framework reported lower prediction error than univariate and "
     "alternative recurrent baselines. Suleiman et al. extended single-step prediction to multi-step "
     "LSTM forecasting for web-application workloads, motivated by the observation that provisioning "
     "decisions require a horizon rather than a point [2]. Golshani and Ashtiani treated proactive "
     "scaling as a sequence-modelling problem with temporal convolutional networks and coupled "
     "prediction to a multi-criteria decision mechanism, evaluating prediction error, service-level "
     "violations and under-utilization together [3]."),
    ("p",
     "These studies establish that recurrent and convolutional models capture short-range workload "
     "dynamics. They also show that prediction accuracy is not by itself a scaling policy: the mapping "
     "from forecast to action, and the treatment of forecast error, remain design decisions."),
    ("h2", "B. Hybrid Statistical-Neural Forecasting"),
    ("p",
     "Guruge and Priyadarshana combined Facebook Prophet and LSTM to forecast HTTP request workloads "
     "for Kubernetes pod scaling, reporting improvements over single-model forecasters and over the "
     "default horizontal autoscaler [4]. Their combination is a hybrid predictor in which the two "
     "models contribute complementary estimates. ProximaScale differs in the role assigned to each "
     "model: Prophet supplies a baseline and is also an input channel, while the LSTM is trained only "
     "on Prophet's residual, so the network's target is the part of the signal the statistical model "
     "does not explain. An earlier weighted-average blend was implemented in this project and "
     "abandoned because it could not outperform the LSTM alone; that module remains in the repository "
     "as a deprecated record."),
    ("h2", "C. Alternative Sequence Architectures"),
    ("p",
     "Arbat et al. proposed a Wasserstein adversarial Transformer for cloud workload prediction, "
     "reporting faster inference and improved accuracy relative to an LSTM baseline on real traces, "
     "and demonstrating predictive VM auto-scaling [5]. Such results show that architectural choice "
     "matters for both accuracy and inference cost. They do not remove the question addressed here: "
     "how much confidence a controller should place in a given forecast at a given moment."),
    ("h2", "D. Learned Scaling Policies"),
    ("p",
     "Agarwal, Rodriguez and Buyya applied deep recurrent reinforcement learning to serverless function "
     "auto-scaling, modelling the partially observable environment and comparing learned policies with "
     "threshold-based autoscaling [6]. Learned policies can adapt without hand-tuned thresholds, at "
     "the cost of training infrastructure, reward design and reduced auditability. ProximaScale "
     "retains an explicit, inspectable decision rule and uses learning only for forecasting and "
     "uncertainty estimation, which suits deployments in which each scaling action must be explainable "
     "after the fact."),
    ("h2", "E. Uncertainty Quantification and Calibration"),
    ("p",
     "Monte Carlo dropout interprets dropout at inference time as approximate Bayesian inference, "
     "yielding a predictive mean and variance from repeated stochastic passes [7]. The technique is "
     "attractive because it requires no separate probabilistic model, but its variance is only "
     "meaningful if it is calibrated against observed errors. Split conformal prediction provides "
     "distribution-free coverage guarantees under exchangeability by calibrating a residual quantile "
     "on held-out data [8], and is used here for the served upper bound. Forecast comparisons are "
     "tested with the Diebold-Mariano statistic using a heteroskedasticity- and autocorrelation-"
     "consistent variance estimate [9]."),
    ("h2", "F. Research Gap"),
    ("p",
     "The reviewed work establishes accurate forecasters and, separately, scaling policies. Three "
     "gaps remain. First, forecast uncertainty is often asserted rather than calibrated and reported: "
     "a variance estimate that is not tested for coverage cannot justify risk-aware scaling. Second, "
     "post-scaling observations are rarely corrected for the controller's own effect, even though "
     "they bias subsequent learning. Third, evaluations frequently report aggregate error without a "
     "feasibility ceiling for lead time, which makes a 60-second lead time look equally achievable on "
     "fast and slow workload ramps. This paper addresses all three explicitly, and reports where the "
     "implementation still falls short."),

    ("h1", "III. Proposed System"),
    ("h2", "A. System Overview"),
    ("p",
     "ProximaScale is organized as a closed control loop (Fig. 1). A workload generator drives a Flask "
     "application container. A separate collector process samples that container's resource usage "
     "through the Docker API at a fixed interval, obtains the workload request count from an HTTP "
     "endpoint exposed by the application, and appends one row per poll to a monitoring CSV. The "
     "control loop reads the most recent 30 rows, constructs the model input, obtains a point forecast "
     "and an upper bound, evaluates a decision, and asks the actuator to add or remove worker "
     "containers. Every decision is written to an event log consumed by a dashboard."),
    ("fig", "fig1_architecture.png",
     "Fig. 1. Deployed ProximaScale architecture. Solid arrows are data flow within one decision cycle; the "
     "orange path is the capacity change that closes the loop. Workers are a resource pool; no load "
     "balancer distributes requests to them (Section III-F)."),
    ("h2", "B. Monitoring and Measurement"),
    ("p",
     "The collector polls once every 30 s, matching the sampling interval on which every model was "
     "trained. Container CPU is computed from the difference between two Docker statistics snapshots "
     "taken two seconds apart, scaled by the number of online CPUs and capped at 100%. The two-snapshot "
     "difference is necessary because a single snapshot's fields are relative to Docker's cached "
     "previous call, which is stale on the first call and after idle periods; using one snapshot "
     "produced readings alternating between 0.01% and 100% in earlier versions."),
    ("p",
     "Request activity is counted inside the application and read-and-reset atomically by the "
     "collector through POST /request-rate/reset, which also reports the wall-clock length of the "
     "window it measured. The CSV field is therefore a count per polling window, not a rate; the "
     "collector prints the derived rate alongside it. Monitoring endpoints are excluded from the "
     "count so that the measurement does not observe itself."),
    ("p",
     "The collector distinguishes four situations that a naive implementation conflates: zero CPU "
     "usage, a missing CPU field, a container that is not running, and a Docker or network failure. "
     "Only the first is written as 0.0; the others produce no row and a log entry. A missing "
     "measurement must not enter the model as a value, and the control loop's staleness check detects "
     "the resulting gap: if the newest row is older than three poll intervals, the cycle is skipped "
     "with a warning instead of scaling on stale data."),
    ("h2", "C. Residual-Stacking Forecasting"),
    ("p",
     "The forecasting component (Fig. 2) combines Prophet and an LSTM in a residual-stacking "
     "arrangement. Prophet models trend and daily/weekly seasonality of CPU utilization and provides "
     "a fitted value for every timestamp in the window plus a forecast for the next three steps. The "
     "LSTM receives the 30-reading window of corrected CPU, memory utilization, request count, "
     "Prophet's fitted value at each reading's timestamp, and Prophet's forecast 90 s after each "
     "timestamp. Its target is the residual between the corrected CPU and the Prophet fitted value "
     "for the next three readings. The final forecast is the sum of Prophet's forecast and the "
     "predicted residual."),
    ("p",
     "Supplying Prophet's values as input channels, rather than only adding its output at the end, "
     "was selected by validation-period model search over more than twenty variants. It lets the "
     "network condition its correction on where the statistical model believes the series is and "
     "where it believes it is going, which matters at spike onsets where Prophet lags."),
    ("fig", "fig3_forecasting_stack.png",
     "Fig. 2. Served forecasting stack. Prophet's fitted value at each reading and its forecast "
     "90 s ahead enter the LSTM as input channels; the network predicts the residual, which is "
     "added back to Prophet's forecast. Thirty stochastic passes are pooled across three seeded "
     "members."),
    ("h2", "D. Uncertainty and Anomaly Awareness"),
    ("p",
     "Two mechanisms produce the risk signal. Dropout layers are kept active at inference and 30 "
     "stochastic forward passes are pooled across three independently seeded models, giving a "
     "predictive mean and standard deviation for each horizon step. Separately, a split-conformal "
     "procedure calibrates a one-sided upper bound on a held-out validation period: residuals are "
     "normalized by the predicted standard deviation, binned into sigma-quantile bins, and a per-bin "
     "margin is chosen so that the resulting bound covers the requested fraction of validation "
     "targets. The served bundle uses the calibrated 95% bound; the raw mean-plus-two-sigma rule "
     "remains available and is the default in the legacy single-model path."),
    ("p",
     "Anomaly awareness is a rolling z-score of the latest corrected CPU reading relative to the "
     "preceding nine readings, with an absolute floor of 5% CPU so that idle jitter cannot be flagged "
     "as a spike whatever its z-score. The flag is available to the decision engine but, as Section "
     "VI-H shows, its precision on this workload is low and it is switched off in the served "
     "configuration."),
    ("h2", "E. Adaptive Decision Engine"),
    ("p",
     "The decision engine converts the forecast into one of three actions: scale_up, scale_down or "
     "hold. Thresholds are adaptive: the upper and lower triggers are the mean of the last 30 "
     "predictions plus or minus 1.5 standard deviations, clamped into operator-controlled windows "
     "(60-90% for the upper trigger, 15-45% for the lower) and separated by at least 10 percentage "
     "points so that the band cannot invert. Until ten predictions exist, the static thresholds from "
     "configuration (75% and 30%) apply. The decision rule is"),
    ("eq", "a_t = scale_up   if  (u_t > U_t  and  not cooling down)  or  (anomaly bypass enabled and flagged)"),
    ("eq", "a_t = scale_down if  u_t < L_t  and  c_t <= 75  and  not cooling down"),
    ("eq", "a_t = hold       otherwise"),
    ("p",
     "where u_t is the forecast at the 90 s horizon, U_t and L_t are the effective upper and lower "
     "thresholds, and c_t is the current measured CPU. Cooldown is a 180 s hysteresis window armed "
     "only after a successful actuation; an unavailable actuator or a rejected action at the replica "
     "limit does not arm it. Two optional branches exist and are disabled in the served configuration: "
     "an anomaly bypass that scales up regardless of the forecast, and an upper-bound branch that "
     "scales up when the calibrated bound alone exceeds U_t. A reactive floor raises capacity when "
     "the current reading already exceeds U_t; it is enabled in the shipped configuration file, "
     "although the tuning documentation states it was rejected, a discrepancy analysed in Section "
     "VI-E."),
    ("h2", "F. Container-Based Actuation"),
    ("p",
     "The actuator uses the Docker SDK to create or remove worker containers from the same image as "
     "the application, each limited to 0.5 CPU and 256 MB, with a health check and a bounded readiness "
     "wait. Replica counts are read back from Docker rather than tracked internally, so a failed "
     "creation cannot desynchronize the fleet. Scale-down removes the newest worker by default; the "
     "strategy is configurable. The minimum is one worker and the maximum is ten."),
    ("p",
     "An important architectural limitation must be stated plainly: in the default pool mode the "
     "workers do not receive application traffic. No load balancer or service proxy is installed, and "
     "the actuator reports traffic_serving separately, as false. Adding workers therefore increases "
     "available capacity but does not reduce the latency of requests that continue to reach the "
     "original container. This is why this paper reports forecast, lead-time and decision-level "
     "metrics, and does not report an end-to-end latency improvement."),
    ("h2", "G. Feedback Mechanism"),
    ("p",
     "Feedback enters the system at two points, and the paper distinguishes them because earlier "
     "descriptions of this work conflated them. The online feedback is the monitoring loop itself plus "
     "delta calibration: the served forecast is re-anchored on the measured current CPU, so the "
     "measured state continuously corrects the model's absolute level. The offline feedback is the "
     "counterfactual correction applied to post-scaling rows before training and inference. There is "
     "no online learning: model weights are not updated at runtime."),
    ("p",
     "Delta calibration exists because the committed Prophet artifact was fitted on a synthetic "
     "series whose baseline is near 50% CPU, whereas a real container idles near 0-1%. The model's "
     "learned change over 90 s transfers; its absolute level does not. The served prediction is "
     "therefore current CPU plus the model's forecast delta, with a 1 percentage-point deadband to "
     "suppress jitter. The mechanism is documented in the code and is observable in the event log's "
     "forecast_mode column."),
    ("h2", "H. End-to-End Operation and Guardrails"),
    ("p",
     "Each cycle reads the last 30 rows, builds the input, predicts, decides, actuates and logs. "
     "Several guardrails make failure modes explicit rather than silent. Forecasts and bounds are "
     "clamped to the physical 0-100% CPU range while preserving their order. If any component of the "
     "prediction pipeline raises, the function returns a conservative overshoot of the measured CPU "
     "(1.15x for the mean, 1.3x for the bound) with the anomaly flag false, and the event log records "
     "forecast_mode as fallback; a broken model can never masquerade as a detected spike. SHAP "
     "attribution is computed on scale-ups using the same scaled window that produced the prediction, "
     "and a failure there records zeros rather than terminating the loop."),

    ("h1", "IV. Residual-Stacking Prediction Model"),
    ("h2", "A. Input Window and Forecast Horizon"),
    ("p",
     "The served model consumes 30 consecutive readings at 30 s spacing, i.e. 15 minutes of history "
     "(Fig. 3). Each reading contributes five channels: counterfactually corrected CPU utilization, "
     "memory utilization, request count for the polling window, Prophet's fitted value at that "
     "timestamp, and Prophet's forecast 90 s after that timestamp. The prediction target is CPU "
     "utilization three steps ahead:"),
    ("eq", "x_t = [ c_tau, m_tau, r_tau, p_tau, p_tau+3 ]  for tau = t-29, ..., t   (1)"),
    ("eq", "y_hat(t+k) = f(x_t),   k = 1, 2, 3                                (2)"),
    ("p",
     "The decision uses the third step, i.e. the forecast 90 s ahead. Three steps is the widest "
     "horizon at which the committed artifacts were trained and validated; longer horizons were not "
     "evaluated and are not claimed."),
    ("fig", "fig2_window_horizon.png",
     "Fig. 3. Input window and forecast horizon. The model sees 15 minutes of five-channel history "
     "and emits three residual steps; the controller acts on the third."),
    ("h2", "B. Prophet Baseline"),
    ("p",
     "Prophet decomposes the series into trend, seasonality and residual components and is fitted "
     "once on the training period [10]. For a window ending at time t it supplies fitted values at "
     "the window's own timestamps and forecasts at t+1, t+2 and t+3. Anchoring the forecast to the "
     "window's last timestamp, rather than to the end of Prophet's training data, was a correctness "
     "fix: the earlier implementation forecast forward from the training boundary, producing "
     "meaningless timestamps for live windows."),
    ("eq", "p(t+k) = Prophet(t+k),   k = 0, 1, 2, 3                            (3)"),
    ("h2", "C. Residual Construction"),
    ("p",
     "The residual is the difference between the corrected CPU and Prophet's fitted value at the same "
     "timestamp:"),
    ("eq", "e(t) = c_corrected(t) - p(t)                                      (4)"),
    ("p",
     "The LSTM is trained to predict e(t+1), e(t+2), e(t+3) from the window. Its task is therefore "
     "not to reproduce the seasonal structure, which Prophet already models, but to anticipate the "
     "part of the series that the statistical model misses, which on this workload is dominated by "
     "spike onsets."),
    ("h2", "D. LSTM Architecture and Training"),
    ("p",
     "Each ensemble member is a two-layer LSTM with 64 and 32 units, dropout 0.1 after each layer, "
     "and a dense output of three residual values, totalling 30,435 trainable parameters. Training "
     "uses mean squared error, Adam with a learning rate of 1e-3, batch size 32, a maximum of 50 "
     "epochs and early stopping with patience 5, restoring the best weights. The training set is the "
     "first 80% of the frozen 90,000-row dataset (of which the final 10% serves as validation); the "
     "models served in the bundle were additionally trained on a variant dataset in which every spike "
     "has a randomized ramp, hold and decay shape, which improved robustness to spike shapes not "
     "present in the original generator."),
    ("p",
     "Three members are trained with seeds 0, 1 and 2, and their stochastic passes are pooled. A "
     "decision therefore costs 30 forward passes in total (ten per member), preserving the "
     "uncertainty budget of the original single-model 30-pass design while reducing the variance "
     "contribution of any one initialization."),
    ("h2", "E. Final Forecast"),
    ("p",
     "The point forecast adds the predicted residual to Prophet's forecast:"),
    ("eq", "y_hat(t+k) = p(t+k) + e_hat(t+k)                                  (5)"),
    ("h2", "F. Live Calibration"),
    ("p",
     "The served point forecast is re-anchored on the measured CPU. Let p(t) be Prophet's value at "
     "the window's last timestamp, e_hat(t+1) the model's first residual step as a proxy for the "
     "residual now, and c(t) the measured CPU. Then"),
    ("eq", "delta_t = c(t) - [ p(t) + e_hat(t+1) ]                             (6)"),
    ("eq", "y_calibrated(t+k) = y_hat(t+k) + delta_t,   k = 1, 2, 3            (7)"),
    ("p",
     "Equivalently, the prediction equals the measured CPU plus the model's forecast change over the "
     "horizon. The offset is held within a 1 percentage-point deadband to suppress jitter but is "
     "applied immediately to larger changes; smoothing a genuine 0-100% transition would be worse "
     "than a visible step. The mode is recorded per decision, and setting the environment variable "
     "PROXIMASCALE_LIVE_CALIBRATION=off restores the uncalibrated absolute output."),
    ("h2", "G. Calibrated Upper Bound"),
    ("p",
     "For the served bundle the upper bound is a split-conformal estimate. On the validation period, "
     "standardized residuals z = (y - mean)/sigma are formed for every window; sigma is divided into "
     "quintile bins by its own quantiles, and for each bin b the margin q_b is the 95th percentile of "
     "z within that bin. At inference,"),
    ("eq", "U(t+k) = y_hat(t+k) + q_bin(sigma(t+k))                            (8)"),
    ("p",
     "This construction targets 95% one-sided coverage overall. Section VI-J shows that it achieves "
     "94.9% overall but only about 87% on targets above 75% CPU, and that the underlying raw sigma is "
     "poorly informative; the calibrated bound should therefore be read as a marginal, not a "
     "conditional, guarantee."),
    ("h2", "H. Counterfactual Correction of Post-Scaling Rows"),
    ("p",
     "Rows collected immediately after a scaling event are marked post_scaling. For each contiguous "
     "run of such rows, the baseline is the last pre-scaling value, and the corrected value is a "
     "fixed blend:"),
    ("eq", "c_corrected = 0.6 * c_observed + 0.4 * c_baseline                    (9)"),
    ("p",
     "The blend is not a full reversion: scaling has a real effect, so the observed value retains the "
     "majority weight. Section VI-G evaluates this choice against the generator's noiseless true load "
     "and finds 0.2 to be a better weight on the available post-scaling rows, so 0.6 is a design "
     "choice that the evidence does not support as optimal."),
    ("table", "TABLE I. Parameters of the served implementation (bundle proximascale_v4)",
     ["Parameter", "Value", "Source"],
     [
         ["Sampling interval", "30 s", "monitoring/collector.py (POLL_INTERVAL)"],
         ["CPU measurement", "two Docker stats snapshots, 2 s apart", "monitoring/metrics.py"],
         ["Input window", "30 readings (15 min)", "model/saved/proximascale_v4/config.json"],
         ["Input channels", "5 (CPU, memory, requests, Prophet t, Prophet t+3)", "config.json"],
         ["Forecast horizon", "3 steps (90 s)", "config.json"],
         ["LSTM", "64 / 32 units, dropout 0.1, dense 3", "seed*/info.json"],
         ["Trainable parameters", "30,435 per member", "seed*/info.json"],
         ["Ensemble", "3 seeds x 10 MC passes = 30 passes", "config.json"],
         ["Upper bound", "split-conformal, sigma-binned, 95%", "config.json (upper_bound)"],
         ["Decision band", "rolling 30 forecasts, k = 1.5", "config.yaml"],
         ["Band clamps", "upper 60-90, lower 15-45, min width 10", "config.yaml"],
         ["Cooldown", "180 s", "config.yaml"],
         ["Replicas", "min 1, max 10; 0.5 CPU and 256 MB each", "config.yaml"],
         ["Reactive floor", "enabled (scale_up_on_current_cpu: true)", "config.yaml"],
         ["Anomaly bypass / upper-bound branch", "disabled / disabled", "config.yaml"],
         ["Scale-down guard", "no scale-down while current CPU > 75%", "config.yaml"],
     ]),
]
