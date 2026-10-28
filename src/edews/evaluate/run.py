"""Evaluate a trained run: threshold chosen on validation, everything reported on test."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

from edews.data.windows import WindowData
from edews.evaluate.metrics import bootstrap_ci, choose_threshold, event_metrics, window_metrics
from edews.utils.misc import get_logger

log = get_logger(__name__)


def predict_all(model, data: WindowData, batch_size: int):
    pc, pf = [], []
    for _, b in data.batches(batch_size):
        a, f = model.predict(b)
        pc.append(a)
        pf.append(f)
    return np.concatenate(pc), np.concatenate(pf)


def run(cfg, model, val: WindowData, test: WindowData, run_dir: Path) -> dict:
    ac = int(cfg.eval.alert_class)
    bs = int(cfg.train.batch_size) * 2

    pc_v, pf_v = predict_all(model, val, bs)
    det_v = pf_v[:, ac:].sum(1)
    tr_v = val.y_cur < ac
    thr = choose_threshold(det_v[tr_v], (val.y_fut[tr_v] >= ac).astype(int), float(cfg.eval.target_sensitivity))

    pc, pf = predict_all(model, test, bs)
    det = pf[:, ac:].sum(1)
    res = {"model": cfg.model.name, "threshold_from_val": thr,
           "val": window_metrics(pc_v, pf_v, val.y_cur, val.y_fut, ac),
           "test": window_metrics(pc, pf, test.y_cur, test.y_fut, ac)}

    idx = test.index
    csn, t, mrn = idx["CSN"].to_numpy(), idx["t"].to_numpy(), idx["MRN"].to_numpy()
    res["test_events"] = event_metrics(csn, t, test.y_cur, det, thr, ac, int(cfg.eval.event_lookback_min))

    tr = test.y_cur < ac
    y_bin = (test.y_fut >= ac).astype(int)

    def auprc(i):
        i = i[tr[i]]
        return average_precision_score(y_bin[i], det[i]) if len(np.unique(y_bin[i])) > 1 else np.nan

    def auroc(i):
        i = i[tr[i]]
        return roc_auc_score(y_bin[i], det[i]) if len(np.unique(y_bin[i])) > 1 else np.nan

    n_boot = int(cfg.eval.bootstrap)
    if n_boot:
        res["test_ci95"] = {"future_transition_auprc": bootstrap_ci(mrn, auprc, n_boot, cfg.seed),
                            "future_transition_auroc": bootstrap_ci(mrn, auroc, n_boot, cfg.seed)}

    preds = idx.select("CSN", "MRN", "t", "y_cur", "y_fut").with_columns(
        *[pl.Series(f"p_fut_{k}", pf[:, k]) for k in range(4)],
        *[pl.Series(f"p_cur_{k}", pc[:, k]) for k in range(4)])
    preds.write_parquet(run_dir / "test_predictions.parquet")
    (run_dir / "metrics.json").write_text(json.dumps(res, indent=2))
    t_ = res["test"]
    log.info("test: transition AUPRC %.3f (prev %.3f) AUROC %.3f | events %s sens %.2f lead %.0f min | "
             "false alerts/pt-h %.3f", t_["future_transition_auprc"], t_["transition_prevalence"],
             t_["future_transition_auroc"], res["test_events"]["events"], res["test_events"]["event_sensitivity"],
             res["test_events"]["lead_time_median_min"], res["test_events"]["false_alerts_per_patient_hour"])
    return res
