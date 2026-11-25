"""Manuscript content, part 2: Sections V-IX and references.

Tables marked by a marker tuple are filled from result CSVs by build_paper.py.
"""

BLOCKS_2 = [
    ("h1", "V. Experimental Setup"),
    ("h2", "A. Datasets"),
    ("p",
     "The primary dataset is a frozen synthetic series of 90,000 rows at 30 s spacing (31 days, "
     "2026-09-19 to 2026-10-20) produced by the repository's generator with a fixed seed. It contains "
     "daily and weekly seasonality, 180 non-overlapping CPU spikes with a 3-row ramp, 14-row hold and "
     "3-row decay, memory correlated with CPU, request counts correlated with CPU and time of day, "
     "and 1,800 post-scaling rows marked after each spike. Utilization spans 5.0-93.6% CPU "
     "(mean 36.0, 95th percentile 62.5), 23.1-81.5% memory and 12-1,446 requests per 30 s window. "
     "The generator retains its latent noiseless series and per-row regime labels (normal, ramp, hold, "
     "decay, post-scaling), which are used only for analysis, never as model inputs."),
    ("p",
     "A second synthetic series of the same length and base process randomizes every spike's shape "
     "(ramp 30 s to 4 min, hold 2-8 min, decay 30 s to 4 min, peak 75-95%, gaps 75 min to 6.7 h). It "
     "is used for training the served bundle and, in its own held-out split, for checking that "
     "improvements are not specific to one spike shape."),
    ("p",
     "Three unseen one-week series were generated after the primary series ends, with generator "
     "parameters fixed before any result on them was inspected: fresh (same generator, different "
     "seed), shifted (3-minute ramps, doubled spikes) and shifted2 (2.5-minute ramps, 1.5x spikes). "
     "The only real data in the repository are 121 rows collected from the Docker application over "
     "two days in September 2026 at roughly 63 s spacing, during which the container was essentially "
     "idle. All synthetic results therefore demonstrate pipeline coherence, not real-traffic "
     "behaviour."),
    ("h2", "B. Splits and Protocol"),
    ("p",
     "The primary series is split chronologically: the first 80% is training data, of which rows "
     "64,800-71,999 form the validation period used for early stopping, model selection, bound "
     "calibration and engine tuning. Rows 72,000 onward are the test split: 17,988 windows, "
     "corresponding to 6.25 days, containing 36 spike onsets above the 75% threshold. Choices were "
     "fixed on validation; the test split was scored after each choice was finalized."),
    ("p",
     "The published 10-reading model of the earlier version of this work used the test split for "
     "early stopping. To allow a like-for-like comparison, a second family of 10-reading models was "
     "retrained with early stopping on validation (the fair protocol). Results from both are reported "
     "where they differ."),
    ("h2", "C. Baselines"),
    ("p",
     "Five forecasters are compared. Persistence uses the current CPU as the forecast, which is the "
     "signal a reactive threshold scaler acts on. Linear extrapolation extends the slope of the last "
     "three readings three steps. Prophet alone uses the statistical model without a residual "
     "correction. The univariate LSTM is a single-input recurrent model without Prophet channels. "
     "ProximaScale denotes the served residual-stacking model; the earlier 10-reading hybrid is "
     "reported separately as ProximaScale (10-step)."),
    ("h2", "D. Metrics"),
    ("p",
     "Forecast error is RMSE and MAE over all test windows at the 90 s horizon, in CPU percentage "
     "points. An event is a target row above 75%; at the window level, precision, recall and F1 treat "
     "a forecast above threshold as an alarm and a false alarm episode as a run of consecutive alarm "
     "windows that overlaps no above-threshold target, reported per day. Lead time for a spike onset "
     "is (onset - t) x 30 s for the earliest window in the three windows preceding the onset whose "
     "forecast exceeds the threshold, and zero if none does. Because no input changes before a "
     "spike's first ramp row, no forecast made earlier can be informed; the feasible ceiling is "
     "therefore min(onset - ramp start, 3) x 30 s, which is 58.3 s on average for this test split. "
     "Crossings count changes in the alarm state; they are split into spurious and missed crossings "
     "by one-to-one matching within three windows."),
    ("p",
     "Uncertainty is evaluated by one-sided coverage, the share of windows whose 90 s target lies "
     "below the upper bound, and by the Spearman correlation between predicted sigma and absolute "
     "error. Differences against each baseline are tested with the Diebold-Mariano statistic using "
     "Newey-West variance at lags 2 and 60, with moving-block bootstrap intervals (120-window blocks, "
     "2,000 replicates) for the error differences and a spike-level bootstrap (10,000 replicates) for "
     "lead-time differences."),
    ("p",
     "Decision quality is evaluated by replaying the decision engine over every test window with a "
     "replica counter that enforces the real minimum and maximum: spikes anticipated (a scale-up "
     "30-90 s before onset), spikes reacted to after onset, spikes missed below the replica ceiling, "
     "false scale-ups per day (no above-threshold reading in the following five minutes), reversals "
     "(opposite actions within five minutes), mean replicas and the share of decisions taken at the "
     "replica maximum."),
    ("h2", "E. Software and Hardware"),
    ("p",
     "Experiments ran on Windows 11 x64 with Python 3.12.10, TensorFlow 2.16.1 (Keras 3), Prophet "
     "1.3.0, scikit-learn 1.4.2, pandas 2.2.2 and NumPy 1.26.4. The repository contains 210 automated "
     "tests, all passing in 61 s on this machine. The evaluation harness writes per-window dumps to "
     "disk and derives every table from them, so tables cannot drift from the predictions: the script "
     "aborts if the re-derived Table II disagrees with the dump's own check or if the dataset hash "
     "has changed."),

    ("h1", "VI. Results and Analysis"),
    ("h2", "A. Point-Forecast Accuracy"),
    ("p",
     "Table II and Fig. 4 report the reproduced test-split results. The served model attains RMSE 4.35 and MAE "
     "2.68, against 10.15/4.68 for Prophet alone, 5.13/3.03 for the univariate LSTM, 6.64/3.92 for "
     "persistence and 10.25/7.71 for linear extrapolation. On this split it also improves on the "
     "earlier 10-reading hybrid (5.09/3.06)."),
    ("table_ii",),
    ("p",
     "Diebold-Mariano tests confirm that the differences are not sampling noise: the RMSE reduction "
     "relative to persistence is 2.30 points (95% bootstrap interval [-2.60, -2.00], p = 8e-32 at lag "
     "2), to Prophet alone 5.81 points ([-6.90, -4.68], p = 6e-38), and to the univariate LSTM 0.78 "
     "points ([-0.94, -0.63], p = 2e-14). The MAE reduction against the univariate LSTM is 0.35 "
     "points ([-0.42, -0.30], p = 5e-48)."),
    ("fig", "fig4_accuracy.png",
     "Fig. 4. Forecast error at the 90 s horizon on the 17,988-window test split. Lower is better; the served "
     "v4 model is not shown because it belongs to the 30-reading family and is reported in Table II "
     "separately from these 10-reading baselines."),
    ("p",
     "Two qualifications belong with this result. First, under the fair protocol the 10-reading "
     "hybrid scores 5.13/3.09 while the fair univariate LSTM scores 5.06/2.99: with identical early "
     "stopping, the 10-reading hybrid is not better than the univariate model. The advantage of the "
     "served model therefore comes from the 30-reading window and the Prophet input channels, not "
     "from residual stacking alone. Second, all errors are dominated by the spike regime: on normal "
     "rows the served model's RMSE is 3.07 against a generator noise floor of 3.00, i.e. essentially "
     "the noise floor, whereas on spike rows every method errs by an order of magnitude more."),
    ("h2", "B. Lead Time"),
    ("p",
     "The served model anticipates 33.8 of 36 spike onsets 90 s ahead (Fig. 5 shows the same quantity "
     "for the 10-reading model) and 2.2 more 30 s ahead, with "
     "no misses; the mean lead time is 58.2 s, which is 0.2 s below the computed feasibility ceiling "
     "of 58.3 s, and the ceiling itself allows 34 of 36 onsets at the full 90 s. Persistence detects "
     "nothing before onset by construction; the univariate LSTM anticipates 30 of 36 onsets with a "
     "38.3 s mean lead. The practical interpretation is that the model extracts almost all of the "
     "lead time that the input window makes available on this workload, and that a target of 60 s is "
     "reachable only because the generator's ramps are short relative to the horizon."),
    ("fig", "fig5_lead_time.png",
     "Fig. 5. Lead-time distribution per generated spike at the 75% threshold for the 10-reading hybrid and "
     "the reactive signal (seed 0). The served 30-reading model is reported in Table II."),
    ("h2", "C. Event-Level Detection"),
    ("p",
     "At the window level the served model reaches precision 0.998, recall 0.891 and F1 0.941, with "
     "no false alarm episodes per day and 9.6 net excess alarm-state crossings (10.2 spurious, 0.6 "
     "missed) over 6.25 days. The linear extrapolator, by contrast, produces 196 crossings and 9.3 "
     "false alarm episodes per day. These values are conditional on the synthetic spike shapes; the "
     "unseen-data results below show how they change when spike shapes shift."),
    ("h2", "D. Decision-Level Replay"),
    ("p",
     "Table III and Fig. 6 report the replay of ten policies on the served model's test dump. The "
     "original engine configuration, with every branch enabled and no scale-down guard, scales up "
     "149 times in 6.25 days, reverses 99 times within five minutes and runs 80% of the time at the "
     "replica maximum; it anticipates only 7 of 36 spikes because the cooldown and the anomaly branch "
     "compete. Removing the anomaly bypass and the upper-bound branch, and adding the 180 s cooldown "
     "and the 75% scale-down guard, reduces scale-ups to 55-57, reversals to 0-1 and time at maximum "
     "to 22-35%, while anticipating 24-29 spikes."),
    ("table_iii",),
    ("p",
     "The shipped configuration file enables the reactive floor. Replaying it as shipped yields 57 "
     "scale-ups, 24 of 36 spikes anticipated, 0.8 false scale-ups per day and 6.16 mean replicas; "
     "disabling that one key yields 55 scale-ups, 29 of 36 anticipated, zero false scale-ups and 5.73 "
     "mean replicas. The floor converts some forecast-driven decisions into reactions to the current "
     "reading, which is why anticipation falls. This contradicts the repository's own tuning notes, "
     "which state that the floor was rejected because it made every evaluated period worse, and it "
     "means that the shipped system is not the policy the tuning procedure selected."),
    ("p",
     "A further inconsistency must be recorded. The tuning harness reports 36 of 36 spikes "
     "anticipated, zero false scale-ups and 2.67 mean replicas for its selected setting on the same "
     "test period, i.e. substantially better decision outcomes than the replay harness produces for "
     "the corresponding policy. The two harnesses define the onset set and the replica accounting "
     "differently, and we could not reconcile them within this study. The numbers in Table III are "
     "those of the replay harness, because they are reproducible from the committed dump; the "
     "tuning-harness figures should not be cited until the discrepancy is resolved."),
    ("fig", "fig6_decision_replay.png",
     "Fig. 6. Reproduced decision replay on the v4 test dump: 17,988 decisions over 6.25 days with 36 "
     "generated spikes. The shipped configuration differs from the forecast-mean policy only in "
     "enabling the reactive floor."),
    ("h2", "E. Generalization to Unseen Workloads"),
    ("p",
     "Table IV reports the unseen-data tests. The served model keeps its accuracy on a fresh "
     "one-week series generated with the same process (RMSE 4.25 versus 5.00 for the original "
     "10-reading model), and, because it was trained on varied spike shapes, it anticipates every "
     "spike on the shifted series where the original model misses 49 of 60 and the univariate LSTM "
     "misses 61 of 79. On the 121 rows of real idle container data every model is far from useful: "
     "the served model's RMSE is 36.0 against 22.7 for simple persistence, because the training "
     "distribution (a synthetic CPU baseline near 50% and memory 23-82%) does not overlap an idle "
     "container (CPU near 0-1%, memory below 1%). The distribution shift is detected and logged at "
     "runtime by a feature-range check, but detection is not correction: retraining on collected "
     "traffic is required before any real deployment claim."),
    ("table_iv",),
    ("h2", "F. Effect of the Prophet Input Channels"),
    ("p",
     "The validation-period model search compared more than twenty variants, including a 30-reading "
     "window without Prophet channels. The selected variant (Prophet channels, 30 readings, dropout "
     "0.1) reduced validation RMSE from 4.81 to 4.02 relative to the 10-reading model trained on the "
     "same data, and the two-stage bundle selection rule required the varied-spike retrain to stay "
     "within 2% on the original validation period (4.058 versus 4.023) while improving on the varied "
     "validation period (3.984 versus 5.038). This is the strongest evidence in the study for the "
     "residual-stacking design, and it is validation-period evidence rather than test-period evidence "
     "because the selection was made there."),
    ("h2", "G. Counterfactual Correction"),
    ("p",
     "Table V sweeps the blend weight on the 360 test windows whose target row is post-scaling, "
     "scored against the generator's noiseless true load. The configured weight of 0.6 yields RMSE "
     "4.43 and a systematic underestimate of 3.67 points; weight 0.2 yields RMSE 3.27 and reduces the "
     "bias, while full reversion to the baseline (weight 0.0) yields 3.89. The correction therefore "
     "does what it was designed to do, but its weight is not optimal on this data, and the effect on "
     "the aggregate metric is negligible: over all windows the difference between correction on and "
     "off is 0.0009 RMSE. Post-scaling rows are only 2% of the test set, which bounds how much any "
     "correction can change the headline error."),
    ("table_v",),
    ("h2", "H. Anomaly Detector Behaviour"),
    ("p",
     "Table VI reports the rolling z-score detector against the generator's known spike intervals. At "
     "the configured threshold of 2.5 it flags 3.5% of decisions, finds every spike (recall 1.0) but "
     "with precision 0.175 and 84 false flags per day; at z = 3.5 precision rises to 0.40 at the cost "
     "of 24 false flags per day. On a workload whose spikes are forecastable, the detector is "
     "redundant and noisy, which is why the anomaly bypass is disabled. On real traffic with "
     "unforecastable bursts the trade-off would need to be re-evaluated."),
    ("table_vi",),
    ("h2", "I. Computational Cost"),
    ("p",
     "Table VII reports per-decision latency measured on the test machine. A single-pass forecast "
     "costs 68 ms, of which Prophet accounts for 24 ms; the served 30-pass ensemble costs 1.36 s, "
     "i.e. 4.5% of the 30 s sampling interval. SHAP attribution on a scale-up adds 0.52 s and runs "
     "only when an action is taken. The univariate LSTM costs 47 ms. The ensemble is therefore "
     "affordable at this sampling rate but would not be at sub-second control intervals without "
     "batching or a reduced pass count."),
    ("table_vii",),
    ("h2", "J. Uncertainty Calibration"),
    ("p",
     "This is the clearest negative result of the study. Raw Monte Carlo dropout sigma is strongly "
     "miscalibrated: a one-sided mean + 2 sigma bound covers 55% of test targets where the nominal "
     "coverage is 98%, and even mean + 3 sigma covers only 58%. The correlation between sigma and "
     "absolute error is near zero overall (Spearman 0.08) and negative inside spikes (-0.20), so the "
     "ensemble's variance does not identify the windows where the model is wrong. The calibrated "
     "conformal bound repairs the marginal coverage (94.9% at a nominal 95%) but only because it "
     "applies a near-constant margin of about 5 CPU points; on targets above 75% CPU its coverage is "
     "about 87%, and its width is dominated by the bin, not by the instance."),
    ("fig", "fig7_uncertainty.png",
     "Fig. 7. Reliability of the raw Monte Carlo dropout bound. The diagonal is perfect calibration; the "
     "measured curve shows that nominal coverage between 69% and 99.9% is delivered as 50-58%."),
    ("p",
     "The consequence for the framework's framing is direct: ProximaScale is uncertainty-aware in the "
     "sense that it computes and calibrates a bound and can act on it, but the bound is a marginal "
     "risk estimate, not an instance-specific confidence measure. The upper-bound decision branch was "
     "disabled after tuning because it added false scale-ups without improving anticipation."),

    ("h1", "VII. Discussion"),
    ("p",
     "Three findings are supported by the evidence. First, the 30-reading residual-stacking forecaster "
     "with Prophet channels is more accurate than every baseline tested on the synthetic test split, "
     "with statistically significant differences and almost all of the feasible lead time. Second, "
     "the decision layer matters as much as the forecaster: replaying the same predictions through "
     "different policies changes scale-ups by a factor of three to seven, reversals from 0 to 99, and "
     "time at the replica ceiling from 22% to 80%. Third, spike-shape diversity in training is what "
     "produces robustness to unseen spike shapes; the original model, trained on a single shape, "
     "misses almost every spike on the shifted series while the varied-shape model misses none."),
    ("p",
     "Three claims are not supported. No end-to-end latency or SLO improvement is demonstrated, "
     "because the actuator adds workers to a pool that receives no traffic; the earlier latency "
     "comparison in this project has no generating script and cannot be reproduced, and is withdrawn. "
     "No real-traffic forecasting accuracy is demonstrated: on the only real data available, which is "
     "idle, every model is worse than persistence. And instance-specific uncertainty is not "
     "demonstrated: the raw dropout variance is uninformative, and the calibrated bound is close to a "
     "constant margin."),
    ("p",
     "The study also illustrates a reproducibility problem that is common in student projects and "
     "worth naming: two harnesses in the same repository, sharing the same predictions, disagree on "
     "the decision-level outcome, and documentation describes a configuration that the configuration "
     "file does not implement. Neither problem affects the forecast metrics, which are pinned by "
     "checksums and cross-checks, but both affect the decision-level claims, and both are reported "
     "rather than smoothed over."),

    ("h1", "VIII. Limitations and Threats to Validity"),
    ("p",
     "Internal validity. The engine's hyperparameters were selected on a 7,200-window validation "
     "period and evaluated on a 17,988-window test period from the same synthetic process; the "
     "selection is therefore only weakly protected against overfitting, and the test period covers "
     "6.25 days rather than a full seasonal cycle. The two decision harnesses disagree, so "
     "decision-level numbers depend on which harness is used; the paper reports the reproducible one."),
    ("p",
     "External validity. All quantitative results except the 121-row real sample come from a "
     "generator whose spike shapes, noise level and seasonality were chosen by the authors. The "
     "distribution shift between synthetic training data and an idle real container is severe "
     "(synthetic CPU baseline near 50%, memory 23-82%; real CPU near 0-1%, memory below 1%), and the "
     "feature-range check reports it but cannot correct it. The actuator runs in pool mode, so no "
     "traffic reaches the added workers and no latency benefit can be measured. Only one application, "
     "one host and one container runtime were exercised."),
    ("p",
     "Construct validity. RMSE and MAE at a 90 s horizon are not service-level objectives. The lead "
     "time ceiling depends on the generator's ramp length, so the near-ceiling result should not be "
     "read as a general 58 s guarantee. The counterfactual ground truth is the generator's latent "
     "series, which assumes scaling suppresses CPU by a known factor; on a real system that "
     "counterfactual is unobservable."),
    ("p",
     "Implementation validity. Model artifacts were serialized with a newer scikit-learn than the "
     "pinned version and load with an InconsistentVersionWarning; they load and reproduce the "
     "reported numbers on this machine, but the pin should be reconciled. The evaluation script for "
     "the 10-reading comparison crashes under the Windows default console encoding when it prints an "
     "emoji, and must be run with UTF-8 output; the vectorized dump harness used for the reported "
     "tables does not have this problem. The reactive floor enabled in config.yaml contradicts the "
     "tuning notes, and the two decision harnesses disagree."),

    ("h1", "IX. Conclusion and Future Work"),
    ("p",
     "ProximaScale is a working proactive autoscaler for containerized applications, built around a "
     "residual-stacking forecaster that combines Prophet with a three-seed Monte Carlo dropout LSTM "
     "ensemble and a decision layer with adaptive thresholds, cooldown and guards. On a synthetic "
     "31-day workload it forecasts CPU 90 s ahead with RMSE 4.35 and MAE 2.68, anticipates 33.8 of 36 "
     "spikes at the full horizon with a mean lead time of 58.2 s against a 58.3 s feasibility "
     "ceiling, and produces no false scale-up episodes per day under the forecast-mean policy. These "
     "gains are significant against persistence, Prophet alone, linear extrapolation and a univariate "
     "LSTM."),
    ("p",
     "The same evidence limits the claims that can be made. Raw dropout variance is not calibrated, "
     "the counterfactual weight is not optimal, no SLO benefit is measured because workers do not "
     "serve traffic, and real-traffic accuracy is untested. The most valuable next steps follow from "
     "those limitations rather than from further model search: install a load balancer or service "
     "proxy so that added replicas serve requests and latency can be measured; collect several days "
     "of real traffic at realistic utilization and retrain on it; replace the fixed counterfactual "
     "weight with an estimated one, or model the intervention explicitly; calibrate uncertainty "
     "conditionally rather than marginally, for example with Mondrian conformal prediction stratified "
     "by predicted load; and unify the two decision harnesses so that one reproducible path produces "
     "every decision-level number."),
    ("p",
     "Code and data availability. The repository contains the implementation, the frozen synthetic "
     "datasets, the trained bundles, the evaluation harnesses, and the scripts that regenerate every "
     "figure and table in this paper. The validation run referenced throughout is stored under "
     "results/validation_2026-11-25/."),

    ("h1", "References"),
    ("refs", [
        "[1] N.-M. Dang-Quang and M. Yoo, \"An efficient multivariate autoscaling framework using "
        "Bi-LSTM for cloud computing,\" Applied Sciences, vol. 12, no. 7, art. 3523, 2022, "
        "doi: 10.3390/app12073523.",
        "[2] B. Suleiman, M. J. Alibasa, Y.-Y. Chang, and A. Anaissi, \"Predictive auto-scaling: "
        "LSTM-based multi-step cloud workload prediction,\" in Service-Oriented Computing - ICSOC "
        "2023 Workshops, Springer, 2024, pp. 5-16, doi: 10.1007/978-981-97-0989-2_1.",
        "[3] E. Golshani and M. Ashtiani, \"Proactive auto-scaling for cloud environments using "
        "temporal convolutional neural networks,\" Journal of Parallel and Distributed Computing, "
        "vol. 154, pp. 119-141, 2021, doi: 10.1016/j.jpdc.2021.04.006.",
        "[4] P. B. Guruge and Y. H. P. P. Priyadarshana, \"Time series forecasting-based Kubernetes "
        "autoscaling using Facebook Prophet and long short-term memory,\" Frontiers in Computer "
        "Science, vol. 7, art. 1509165, 2025, doi: 10.3389/fcomp.2025.1509165.",
        "[5] S. Arbat, V. K. Jayakumar, J. Lee, W. Wang, and I. K. Kim, \"Wasserstein adversarial "
        "transformer for cloud workload prediction,\" Proceedings of the AAAI Conference on "
        "Artificial Intelligence, vol. 36, no. 11, pp. 12433-12439, 2022, "
        "doi: 10.1609/aaai.v36i11.21509.",
        "[6] S. Agarwal, M. A. Rodriguez, and R. Buyya, \"A deep recurrent-reinforcement learning "
        "method for intelligent autoscaling of serverless functions,\" IEEE Transactions on Services "
        "Computing, vol. 17, no. 5, pp. 1899-1910, 2024, doi: 10.1109/TSC.2024.3387661.",
        "[7] Y. Gal and Z. Ghahramani, \"Dropout as a Bayesian approximation: Representing model "
        "uncertainty in deep learning,\" in Proceedings of the 33rd International Conference on "
        "Machine Learning, 2016, pp. 1050-1059.",
        "[8] A. N. Angelopoulos and S. Bates, \"A gentle introduction to conformal prediction and "
        "distribution-free uncertainty quantification,\" arXiv:2107.07511, 2021.",
        "[9] F. X. Diebold and R. S. Mariano, \"Comparing predictive accuracy,\" Journal of Business "
        "and Economic Statistics, vol. 13, no. 3, pp. 253-263, 1995.",
        "[10] S. J. Taylor and B. Letham, \"Forecasting at scale,\" The American Statistician, "
        "vol. 72, no. 1, pp. 37-45, 2018, doi: 10.1080/00031305.2017.1380080.",
        "[11] A. Verma, L. Pedrosa, M. Korupolu, D. Oppenheimer, E. Tune, and J. Wilkes, \"Large-scale "
        "cluster management at Google with Borg,\" in Proceedings of the 10th European Conference on "
        "Computer Systems (EuroSys), 2015, doi: 10.1145/2741948.2741964.",
        "[12] Kubernetes documentation, \"Horizontal Pod Autoscaling,\" Kubernetes project. "
        "[Online]. Available: https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/",
    ]),
]
