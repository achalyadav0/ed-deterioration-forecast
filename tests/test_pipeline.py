"""End-to-end on a tiny synthetic dataset: stages, caching, split integrity, export, edge parity."""

import json

import numpy as np
import polars as pl
import pytest

from edews.config import load_config
from edews.data.synthetic import generate
from edews.pipeline import Pipeline


@pytest.fixture(scope="module")
def cfg_factory(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2e")
    raw = generate(root / "raw", n_visits=120, seed=3)

    def make(*overrides):
        return load_config("configs/config.yaml", [
            f"paths.raw_dir={raw}", f"paths.work_dir={root / 'work'}", f"paths.runs_dir={root / 'runs'}",
            "train.max_epochs=2", "train.batch_size=256", "eval.bootstrap=5", "export.stream_visits=3", *overrides])
    return make


def test_split_is_patient_disjoint(cfg_factory):
    p = Pipeline(cfg_factory("model=persistence"))
    sd = p.ensure("index")
    idx = pl.read_parquet(sd.path / "index.parquet")
    per_patient = idx.group_by("MRN").agg(pl.col("split").n_unique().alias("n"))
    assert per_patient["n"].max() == 1
    assert set(idx["split"].unique().to_list()) <= {0, 1, 2}


def test_inputs_end_at_t_and_labels_follow(cfg_factory):
    p = Pipeline(cfg_factory("model=persistence"))
    p.ensure("index")
    d = p.data(0)
    b = d.get(np.arange(min(len(d), 50)))
    W = int(p.cfg.windows.length_min)
    assert b.x.shape[1:] == (W, 3 * len(p.cfg.resample.vitals))
    # the last input row is the window's own minute t (row) -- never later
    rows = d.rows[:50]
    np.testing.assert_array_equal(b.mask[:, -1], np.asarray(d.arr["mask"][rows]))


def test_caching_reruns_only_downstream(cfg_factory):
    p = Pipeline(cfg_factory("model=persistence"))
    keys = {s: p.ensure(s).key for s in ("ingest", "featurize", "label", "split", "index")}
    p2 = Pipeline(cfg_factory("model=persistence", "labels.horizon_end=45"))
    keys2 = {s: p2.dir(s).key for s in keys}
    assert keys2["ingest"] == keys["ingest"] and keys2["featurize"] == keys["featurize"]
    assert keys2["split"] == keys["split"]
    assert keys2["label"] != keys["label"] and keys2["index"] != keys["index"]


@pytest.mark.parametrize("model", ["persistence", "xgb", "tcn", "gru"])
def test_train_evaluate(cfg_factory, model):
    p = Pipeline(cfg_factory(f"model={model}"))
    sd = p.ensure("evaluate")
    m = json.loads((sd.path / "metrics.json").read_text())
    assert 0.0 <= m["test"]["future_acc"] <= 1.0
    if model == "persistence":
        assert m["test"]["current_acc"] == 1.0      # current state is computable from inputs


@pytest.mark.parametrize("model", ["tcn", "xgb"])
def test_export_and_edge_parity(cfg_factory, model):
    from edews.edge.replay import replay
    p = Pipeline(cfg_factory(f"model={model}"))
    sd = p.ensure("export")
    rep = replay(sd.path)
    assert rep["inferences"] > 0
    assert rep["parity_vs_offline"]["compared_windows"] > 0
    assert rep["parity_vs_offline"]["max_abs_diff"] < 1e-4
