"""Minimal experiment tracker: always writes metrics.jsonl; mirrors to Weights & Biases if enabled.

Only configs and aggregate metrics are logged -- never patient-level rows (PhysioNet DUA).
On Kaggle, put WANDB_API_KEY in Kaggle Secrets and set ``train.tracker=wandb``.
"""

from __future__ import annotations

import json
from pathlib import Path


class Tracker:
    def __init__(self, run_dir: Path, run_name: str, config: dict, backend: str = "none"):
        self.path = Path(run_dir) / "metrics.jsonl"
        self.wandb = None
        if backend == "wandb":
            try:
                import wandb

                self.wandb = wandb.init(project="ed-deterioration", name=run_name, config=config,
                                        dir=str(run_dir), resume="allow", id=run_name)
            except Exception as e:  # tracking must never break training
                print(f"[tracker] wandb disabled: {e}")

    def log(self, metrics: dict, step: int | None = None) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps({"step": step, **metrics}, default=float) + "\n")
        if self.wandb is not None:
            self.wandb.log(metrics, step=step)

    def summary(self, metrics: dict) -> None:
        if self.wandb is not None:
            for k, v in metrics.items():
                self.wandb.summary[k] = v

    def close(self) -> None:
        if self.wandb is not None:
            self.wandb.finish()
