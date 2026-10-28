# Methods notes (decisions to report in the thesis)

## Cohort and preprocessing
* MC-MED adult ED visits to monitored beds; vitals from `numerics.csv` (`HR`, `RR`, `SpO2`, `SBP`,
  `Temp` in °F → °C, `LPM_O2` for the NEWS2 oxygen item). `Source` Monitor + Chart.
* Visits with < 105 monitored minutes (window + horizon) are excluded. Report a cohort flow
  diagram from each stage's `meta.json`.
* Implausible values are dropped (`clean.ranges`); HR/RR/SpO2 get a causal 3-minute median filter.
* 1-minute grid; **forward fill only**, with per-vital limits (5 min HR/RR/SpO2, 60 min SBP,
  360 min temperature). No backward fill or interpolation (both leak the future).
* Each vital also gets a measured-mask channel and a minutes-since-measured channel.

## Labels
* NEWS2-derived score without the consciousness item (not available). Classes follow the RCP
  response bands: low / low-medium (any single 3) / medium (5–6) / high (≥ 7).
* A minute is scorable only if HR, RR and SpO2 are available.
* **Current** label: worst class in the last 5 minutes. **Future** label: worst class in
  (t+30, t+60] minutes, requiring ≥ 50% scorable minutes in that span.
* The current label is a deterministic function of the inputs, so the current head is a sanity
  check, not a result (persistence scores 100% on it).

## Splits
* MC-MED's chronological split files by default (`split.scheme=chrono`), so results are comparable
  with other MC-MED work. A patient never appears in two splits; later visits of a patient already
  in an earlier split are dropped and counted in `split/meta.json`.
* Normalisation statistics are fitted on the training split only.
* Training windows: 45 min, stride 5 min. Evaluation windows: stride 1 min (as on the Pi).

## Baselines and models
`persistence` (future = current), `news2_rule` (escalate on a ≥ 2-point rise in 30 min), `xgb`
(window statistics), `gru`, `tcn` (causal dilated convolutions, receptive field 61 min). All share
the same inputs, splits and metrics.

## Metrics
* **Primary:** transition AUPRC. Among windows currently below medium, does the model predict
  reaching ≥ medium within the horizon? Report it with its prevalence (the AUPRC of a random model).
* Transition AUROC, future macro-F1, per-class recall, ordinal MAE, Brier score.
* Event level, on stride-1 test windows: an event is the first time a visit reaches ≥ medium after
  being below it. Report sensitivity, median lead time, false alert onsets per patient-hour, and the
  fraction of pre-event time spent in alert (`alert_time_frac`, which exposes always-on alarms).
* The alert threshold is chosen on **validation** for 80% sensitivity, then applied to test.
* 95% CIs by bootstrap over **patients**.

## Model selection rule (fix this before looking at test results)
1. Rank models by validation transition AUPRC, mean over 3 seeds (`seed=0,1,2`).
2. If confidence intervals overlap, prefer the model with lower Pi latency / fewer parameters.
3. Run the chosen model and all baselines on test once, and report everything.

## Known limitations to state
* No consciousness (ACVPU) item; SBP and temperature are intermittent in ED data.
* Labels come from the same vitals as the inputs. Add an outcome check (e.g. ICU admission from
  `visits.csv` `ED_dispo`) as secondary validation.
* Domain gap between Philips bedside monitors (training) and low-cost ESP32 sensors (deployment).
