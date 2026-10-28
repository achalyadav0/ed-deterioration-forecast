"""Replay the bundle's test stream through the on-device runtime and benchmark it.

Run this ON the Raspberry Pi (or locally first):

    python -m edews.edge.replay --bundle runs/<run>/export/<key> [--int8] [--threads 1]

Reports: inference latency (p50/p95/p99), peak memory, agreement with the offline model's
predictions on the same windows (should be ~0 difference; that proves preprocessing parity),
and the escalation decisions made. Writes ``replay_report.json`` and ``decisions.csv``.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

from edews.edge.runtime import Bundle, PatientMonitor


def replay(bundle_dir: Path, stream: Path | None = None, int8: bool = False, threads: int = 1,
           out_dir: Path | None = None) -> dict:
    bundle = Bundle.load(bundle_dir, int8=int8, threads=threads)
    stream = stream or bundle_dir / "stream_sample.jsonl.gz"
    out_dir = out_dir or bundle_dir
    monitors: dict[str, PatientMonitor] = {}
    lat, diffs, decisions = [], [], []
    t_start = time.perf_counter()
    n = 0
    with gzip.open(stream, "rt") as f:
        for line in f:
            rec = json.loads(line)
            mon = monitors.setdefault(rec["bed"], PatientMonitor(bundle))
            r = mon.step(rec["obs"], rec.get("on_o2", 0))
            n += 1
            if r.ready:
                lat.append(r.infer_ms)
                if "expected_p_fut" in rec:
                    diffs.append(np.abs(np.asarray(r.p_fut) - np.asarray(rec["expected_p_fut"])).max())
            if r.decision != "NONE":
                decisions.append({"bed": rec["bed"], "t": r.t, "decision": r.decision, "reason": r.reason,
                                  "news2": r.news2, "current_class": r.current_class})
    wall = time.perf_counter() - t_start
    lat = np.asarray(lat)
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 if sys.platform != "darwin" else 1024 ** 2)
    report = {
        "model_file": bundle.kind, "threads": threads, "patients": len(monitors), "minutes_streamed": n,
        "inferences": len(lat),
        "latency_ms": {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95)),
                       "p99": float(np.percentile(lat, 99)), "mean": float(lat.mean())} if len(lat) else None,
        "throughput_minutes_per_sec": n / wall if wall else None,
        "peak_rss_mb": round(rss_mb, 1),
        "parity_vs_offline": {"compared_windows": len(diffs),
                              "max_abs_diff": float(np.max(diffs)) if diffs else None,
                              "mean_abs_diff": float(np.mean(diffs)) if diffs else None},
        "decisions": {k: sum(d["decision"] == k for d in decisions) for k in ("NOTIFY_DOCTOR", "CALL_MET")},
    }
    (out_dir / "replay_report.json").write_text(json.dumps(report, indent=2))
    with (out_dir / "decisions.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["bed", "t", "decision", "reason", "news2", "current_class"])
        w.writeheader()
        w.writerows(decisions)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("--stream", type=Path, default=None)
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    print(json.dumps(replay(a.bundle, a.stream, a.int8, a.threads), indent=2))


if __name__ == "__main__":
    main()
