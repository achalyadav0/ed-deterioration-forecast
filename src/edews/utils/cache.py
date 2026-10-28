"""Content-addressed stage cache.

Every pipeline stage writes its outputs to ``<work_dir>/<stage>/<key>/``. The key is a hash of
(stage name, stage code version, the config sections the stage reads, upstream stage keys).
If that directory already holds a ``_DONE`` marker the stage is skipped. Change a config value
and only the stages that read it (plus everything downstream) get a new key and rerun.

When you change a stage's *code* in a way that changes its output, bump its VERSION constant
in ``pipeline.py`` (or run with ``--force <stage>``).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def stage_key(name: str, version: int, params: dict[str, Any], upstream: list[str]) -> str:
    blob = json.dumps(
        {"stage": name, "version": version, "params": params, "upstream": sorted(upstream)},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


@dataclass
class StageDir:
    name: str
    key: str
    path: Path

    @property
    def done(self) -> bool:
        return (self.path / "_DONE").exists()

    def begin(self) -> None:
        if self.path.exists():
            shutil.rmtree(self.path)  # remove a half-finished previous attempt
        self.path.mkdir(parents=True)

    def finish(self, meta: dict[str, Any]) -> None:
        (self.path / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
        (self.path / "_DONE").write_text("ok")

    def meta(self) -> dict[str, Any]:
        return json.loads((self.path / "meta.json").read_text())


def stage_dir(work_dir: str | Path, name: str, version: int, params: dict, upstream: list[str]) -> StageDir:
    key = stage_key(name, version, params, upstream)
    return StageDir(name=name, key=key, path=Path(work_dir) / name / key)
