"""Config loading: one main YAML + one model YAML, plus command-line dotlist overrides."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_config(path: str | Path = "configs/config.yaml", overrides: Iterable[str] = ()) -> DictConfig:
    """Load the main config, resolve the selected model file, apply overrides.

    Overrides use OmegaConf dotlist syntax: ``labels.horizon_end=30``, ``model=gru``,
    ``model.hidden=128``. Model-level overrides are applied after the model file is loaded.
    """
    path = Path(path)
    cfg = OmegaConf.load(path)
    overrides = list(overrides)
    model_over = [o for o in overrides if o.startswith("model.")]
    other_over = [o for o in overrides if not o.startswith("model.")]
    if other_over:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(other_over))

    model_name = cfg.model if isinstance(cfg.model, str) else cfg.model.name
    model_file = path.parent / "model" / f"{model_name}.yaml"
    if not model_file.exists():
        raise FileNotFoundError(f"No model config {model_file}")
    cfg.model = OmegaConf.load(model_file)
    if model_over:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(model_over))
    return cfg


def section(cfg: DictConfig, *keys: str) -> dict:
    """Plain-dict snapshot of selected config sections (used for cache keys and run metadata)."""
    return {k: OmegaConf.to_container(cfg[k], resolve=True) if isinstance(cfg[k], DictConfig)
            else cfg[k] for k in keys}
