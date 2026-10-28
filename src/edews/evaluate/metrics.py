"""Metrics. The headline numbers are the *transition* ones: among windows where the patient is
currently below ``alert_class``, how well do we predict they will reach it within the horizon?
Persistence scores no better than chance there, so it is where a model has to earn its keep.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    recall_score,
    roc_auc_score,
)


def _safe(fn, y, s):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(fn(y, s))


def window_metrics(p_cur, p_fut, y_cur, y_fut, alert_class: int) -> dict:
    out: dict = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pc, pf = p_cur.argmax(1), p_fut.argmax(1)
        out["current_acc"] = float(accuracy_score(y_cur, pc))
        out["current_macro_f1"] = float(f1_score(y_cur, pc, average="macro", labels=[0, 1, 2, 3], zero_division=0))
        out["future_acc"] = float(accuracy_score(y_fut, pf))
        out["future_macro_f1"] = float(f1_score(y_fut, pf, average="macro", labels=[0, 1, 2, 3], zero_division=0))
        rec = recall_score(y_fut, pf, average=None, labels=[0, 1, 2, 3], zero_division=0)
        for k in range(4):
            out[f"future_recall_c{k}"] = float(rec[k])
        out["future_ordinal_mae"] = float(np.abs(pf - y_fut).mean())

        det_s = p_fut[:, alert_class:].sum(1)
        det_y = (y_fut >= alert_class).astype(int)
        out["future_det_auroc"] = _safe(roc_auc_score, det_y, det_s)
        out["future_det_auprc"] = _safe(average_precision_score, det_y, det_s)
        out["future_det_brier"] = float(brier_score_loss(det_y, np.clip(det_s, 0, 1))) if len(det_y) else float("nan")

        tr = y_cur < alert_class
        out["transition_n"] = int(tr.sum())
        out["transition_prevalence"] = float(det_y[tr].mean()) if tr.any() else float("nan")
        out["future_transition_auroc"] = _safe(roc_auc_score, det_y[tr], det_s[tr])
        out["future_transition_auprc"] = _safe(average_precision_score, det_y[tr], det_s[tr])
    return out


def choose_threshold(det_score: np.ndarray, det_y: np.ndarray, target_sensitivity: float) -> float:
    """Highest threshold that still flags ``target_sensitivity`` of the positives. Pick on validation only."""
    pos = np.sort(det_score[det_y == 1])
    if len(pos) == 0:
        return 0.5
    return float(np.quantile(pos, 1 - target_sensitivity, method="lower"))


def event_metrics(csn, t, y_cur, det_score, threshold: float, alert_class: int, lookback: int) -> dict:
    """Visit-level evaluation on stride-1 windows.

    Event = first minute a visit reaches ``alert_class`` after having been below it.
    Caught = an alert fired in the ``lookback`` minutes before the event. Lead time = event
    minute - first such alert. False alert episodes = alert onsets outside any pre-event span.
    """
    order = np.lexsort((t, csn))
    csn, t, y_cur, s = csn[order], t[order], y_cur[order], det_score[order]
    alert = s >= threshold
    leads, missed, false_onsets, alert_onsets, calm_minutes, calm_alert_minutes = [], 0, 0, 0, 0, 0
    bounds = np.flatnonzero(np.diff(csn)) + 1
    for a, b in zip(np.r_[0, bounds], np.r_[bounds, len(csn)]):
        tt, yy, al = t[a:b], y_cur[a:b], alert[a:b]
        below = yy < alert_class
        calm_minutes += int(below.sum())
        calm_alert_minutes += int((below & al).sum())
        onset = al & ~np.r_[False, al[:-1] & (np.diff(tt) == 1)]
        alert_onsets += int((onset & below).sum())
        hit = np.flatnonzero(~below)
        ev_t = None
        if len(hit) and below[:hit[0]].any():
            ev_t = tt[hit[0]]
            pre = (tt >= ev_t - lookback) & (tt < ev_t) & al
            if pre.any():
                leads.append(float(ev_t - tt[pre][0]))
            else:
                missed += 1
        in_pre = (tt >= ev_t - lookback) & (tt <= ev_t) if ev_t is not None else np.zeros_like(below)
        false_onsets += int((onset & below & ~in_pre).sum())
    n_events = len(leads) + missed
    hours = calm_minutes / 60.0
    return {
        "events": n_events,
        "event_sensitivity": len(leads) / n_events if n_events else float("nan"),
        "lead_time_median_min": float(np.median(leads)) if leads else float("nan"),
        "lead_time_iqr_min": [float(np.percentile(leads, 25)), float(np.percentile(leads, 75))] if leads else None,
        "false_alerts_per_patient_hour": false_onsets / hours if hours else float("nan"),
        "alerts_per_patient_hour": alert_onsets / hours if hours else float("nan"),
        # share of not-yet-deteriorated time spent in alert; exposes "always on" alarms
        "alert_time_frac": calm_alert_minutes / calm_minutes if calm_minutes else float("nan"),
        "threshold": threshold,
    }


def bootstrap_ci(groups: np.ndarray, fn, n: int, seed: int = 0) -> tuple[float, float]:
    """Percentile CI, resampling *patients* (groups), not windows."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv))[:-1])
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[i] for i in pick])
        v = fn(idx)
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
