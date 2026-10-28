"""MQTT version of the replay: a publisher plays the role of the ESP32 sensor nodes, the
service runs on the Pi and publishes escalation alerts.

On the Pi (with a broker, e.g. `sudo apt install mosquitto`):
    python -m edews.edge.mqtt serve   --bundle <bundle> [--int8]
From a laptop or the Pi itself (simulated ESP32s):
    python -m edews.edge.mqtt publish --bundle <bundle> --speedup 60

Message in  (topic ed/vitals/<bed>): {"bed": ..., "t": minute, "obs": {...}, "on_o2": 0|1, "sent": unix_ts}
Message out (topic ed/alerts/<bed>): {"bed", "t", "decision", "reason", "p_fut", "e2e_ms"}
Later, the real ESP32 firmware only has to publish the same JSON once per minute.
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np

from edews.edge.runtime import Bundle, PatientMonitor


def _client(host: str, port: int, name: str):
    import paho.mqtt.client as mqtt
    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=name)
    c.connect(host, port)
    return c


def serve(bundle_dir: Path, host: str, port: int, topic_in: str, topic_out: str, int8: bool) -> None:
    bundle = Bundle.load(bundle_dir, int8=int8)
    monitors: dict[str, PatientMonitor] = {}
    e2e = []
    c = _client(host, port, "edews-pi")

    def on_message(client, userdata, msg):
        rec = json.loads(msg.payload)
        mon = monitors.setdefault(rec["bed"], PatientMonitor(bundle))
        r = mon.step(rec["obs"], rec.get("on_o2", 0))
        ms = (time.time() - rec.get("sent", time.time())) * 1000
        e2e.append(ms)
        if r.decision != "NONE":
            client.publish(f"{topic_out}/{rec['bed']}", json.dumps(
                {"bed": rec["bed"], "t": r.t, "decision": r.decision, "reason": r.reason,
                 "p_fut": r.p_fut, "e2e_ms": round(ms, 2)}))
            print(f"[{rec['bed']} t={r.t}] {r.decision}: {r.reason}")
        if len(e2e) % 1000 == 0:
            a = np.asarray(e2e)
            print(f"{len(a)} msgs, end-to-end ms p50={np.percentile(a, 50):.2f} p99={np.percentile(a, 99):.2f}")

    c.on_message = on_message
    c.subscribe(f"{topic_in}/+")
    print(f"listening on {topic_in}/+ ({bundle.kind})")
    c.loop_forever()


def publish(bundle_dir: Path, host: str, port: int, topic_in: str, speedup: float, stream: Path | None) -> None:
    stream = stream or bundle_dir / "stream_sample.jsonl.gz"
    by_minute: dict[int, list[dict]] = {}
    with gzip.open(stream, "rt") as f:
        for line in f:
            rec = json.loads(line)
            rec.pop("expected_p_fut", None)
            by_minute.setdefault(rec["t"], []).append(rec)
    c = _client(host, port, "edews-sim-esp32")
    c.loop_start()
    for t in sorted(by_minute):
        for rec in by_minute[t]:
            rec["sent"] = time.time()
            c.publish(f"{topic_in}/{rec['bed']}", json.dumps(rec))
        time.sleep(60.0 / speedup)
    c.loop_stop()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["serve", "publish"])
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("--stream", type=Path, default=None)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--topic-in", default="ed/vitals")
    ap.add_argument("--topic-out", default="ed/alerts")
    ap.add_argument("--speedup", type=float, default=60.0)
    ap.add_argument("--int8", action="store_true")
    a = ap.parse_args()
    if a.mode == "serve":
        serve(a.bundle, a.host, a.port, a.topic_in, a.topic_out, a.int8)
    else:
        publish(a.bundle, a.host, a.port, a.topic_in, a.speedup, a.stream)


if __name__ == "__main__":
    main()
