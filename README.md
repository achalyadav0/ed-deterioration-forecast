# ED deterioration forecasting on the edge

Forecasting NEWS2-based clinical deterioration of emergency department patients from five
vital signs, using the [MC-MED](https://physionet.org/content/mc-med/) dataset, with models
deployed to a Raspberry Pi 5 for streaming inference and escalation.

```
MC-MED CSVs ─► ingest ─► featurize ─► label ─► index ─► train ─► evaluate ─► export ─► Raspberry Pi
                  └──────► split ──────────────┘          (tcn | gru | xgb | baselines)   (ONNX, replay, MQTT)
```

* **Modular**: every stage is a separate module behind a config section; models share one interface.
* **Cached**: each stage's output is stored under a hash of the config it reads. Change the
  horizon and only labelling onwards reruns; switch TCN → GRU and only training onwards reruns.
* **No train/serve skew**: feature building, normalisation and NEWS2 scoring live in
  `edews.core` (pure numpy) and are reused unchanged on the Pi. A replay test proves the Pi's
  predictions match the offline ones.
* **Honest evaluation**: patient-disjoint splits, persistence and NEWS2-rule baselines,
  transition-subset AUPRC, event-level lead time and alarm burden, patient-bootstrap CIs.

> **Data licence.** MC-MED is credentialed PhysioNet data. It is never committed here
> (see `.gitignore`); keep any Kaggle copy private and never print patient-level rows in shared
> outputs. Everything in this repo runs on synthetic data for development and CI.

## Quick start (laptop, synthetic data, CPU only)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[train,dev]"

edews synth --out data/raw_synth --n-visits 1000           # fake MC-MED-shaped data
for m in persistence news2_rule xgb gru tcn; do
  edews run --stages evaluate paths.raw_dir=data/raw_synth model=$m
done
edews compare paths.raw_dir=data/raw_synth                 # results table -> runs/comparison.csv
edews run --stages export paths.raw_dir=data/raw_synth model=tcn
python -m edews.edge.replay --bundle runs/tcn_*/export/*/  # on-device simulation + parity check
pytest                                                      # leakage, NEWS2, parity, end-to-end tests
```

## Real data

1. Download MC-MED from PhysioNet (needs credentialed access) into `data/raw/`
   (`visits.csv`, `numerics.csv`, `split_chrono_*.csv`).
2. `edews run --stages all` (or `data.max_visits=2000` first for a quick check).
3. Without a GPU: use the two Kaggle notebooks, see [docs/kaggle.md](docs/kaggle.md).

## Changing things

Everything is in [`configs/config.yaml`](configs/config.yaml) and `configs/model/*.yaml`, and
can be overridden on the command line:

```bash
edews run --stages evaluate model=gru model.hidden=128
edews run --stages evaluate labels.horizon_start=0 labels.horizon_end=30
edews run --stages evaluate windows.length_min=30 train.loss_weight_current=0
edews run --stages featurize --force featurize          # recompute a stage explicitly
```

If you change a stage's *code* so its output changes, bump that stage in `VERSIONS` in
`src/edews/pipeline.py`; otherwise the old cached output is reused.

## Repository layout

```
configs/                  config.yaml + model/{tcn,gru,xgb,persistence,news2_rule}.yaml
src/edews/
  core/                   pure numpy, shared with the Pi: news2, features, labels, channels
  data/                   ingest, featurize, label, split, windows (index + batch reader), synthetic
  models/                 base interface, torch (TCN, GRU), tabular (XGBoost), baselines
  evaluate/               metrics (transition, events, bootstrap) and the evaluation stage
  edge/                   runtime (per-patient monitor + escalation), replay benchmark, MQTT
  pipeline.py             stage graph, caching, CLI (edews run | compare | synth)
notebooks/                Kaggle: 01 preprocess (CPU), 02 train/evaluate/export (GPU)
tests/                    NEWS2, causality/leakage, online==offline parity, end-to-end
docs/                     kaggle.md, raspberry-pi.md, methods.md
```

## Outputs

* `data/work/<stage>/<hash>/` cached stage outputs with `meta.json` (counts, class balance, timings)
* `runs/<model>_<hash>/` trained model, `config.yaml`, `metrics.jsonl`, and
  * `evaluate/<hash>/metrics.json`, `test_predictions.parquet`
  * `export/<hash>/` **edge bundle**: `model.onnx`, `model.int8.onnx`, `normalizer.json`,
    `edge_config.json`, `stream_sample.jsonl.gz`

See [docs/methods.md](docs/methods.md) for the labelling, evaluation and model-selection
decisions, and [docs/raspberry-pi.md](docs/raspberry-pi.md) for deployment.
