# Raspberry Pi 5 deployment

The export stage produces a self-contained **bundle** folder:

| File | Purpose |
| --- | --- |
| `model.onnx`, `model.int8.onnx` | the model (FP32 and dynamically quantised INT8) |
| `normalizer.json` | training-set normalisation, applied on the Pi exactly as in training |
| `edge_config.json` | feature spec, window length, NEWS2 settings, alert threshold chosen on validation, escalation rules |
| `stream_sample.jsonl.gz` | a few test visits minute-by-minute, with the offline model's expected output |

The stream sample contains de-identified MC-MED test data. Keep it on your own devices only.

## Install

```bash
sudo apt install -y python3-venv mosquitto   # mosquitto only for the MQTT demo
git clone https://github.com/<you>/ed-deterioration-forecast.git && cd ed-deterioration-forecast
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[edge]"                      # numpy, onnxruntime, paho-mqtt; no torch needed
# copy the bundle folder from Kaggle/laptop, e.g. to ~/bundle
```

## 1. Replay benchmark (no network)

```bash
python -m edews.edge.replay --bundle ~/bundle            # FP32
python -m edews.edge.replay --bundle ~/bundle --int8     # INT8
python -m edews.edge.replay --bundle ~/bundle --threads 4
```

`replay_report.json` gives inference latency (p50/p95/p99), throughput, peak memory, the number
of escalations, and **parity vs offline**: the maximum difference between the Pi's predictions and
the offline model's for the same windows. It should be ~1e-6 for FP32. A large value means the
Pi's preprocessing differs from training, and results on the Pi cannot be trusted.

## 2. MQTT end-to-end (simulated ESP32 nodes)

```bash
python -m edews.edge.mqtt serve   --bundle ~/bundle             # terminal 1 (on the Pi)
python -m edews.edge.mqtt publish --bundle ~/bundle --speedup 60  # terminal 2 (Pi or laptop, --host <pi-ip>)
mosquitto_sub -t 'ed/alerts/#'                                   # terminal 3: watch alerts
```

The real ESP32 firmware only has to publish the same JSON once per minute to `ed/vitals/<bed>`:

```json
{"bed": "bed-07", "t": 123, "obs": {"hr": 92.0, "rr": 18.0, "spo2": 95.0, "sbp": null, "temp": null}, "on_o2": 0, "sent": 1761650000.0}
```

Use `null` for vitals not measured in that minute; the runtime applies the same forward-fill
limits, masks and time-since features as training.

## Escalation

`edews.edge.runtime.PatientMonitor._escalate` is a transparent rule layer:

* current NEWS2 class high → `CALL_MET`
* P(high within horizon) ≥ `met_threshold` → `CALL_MET`
* P(≥ medium within horizon) ≥ validation threshold → `NOTIFY_DOCTOR`
* a cooldown per level suppresses repeats

This is the place to plug in a learned (e.g. RL or contextual-bandit) policy later. Evaluate any
policy on alerts per patient-hour against missed deteriorations.
