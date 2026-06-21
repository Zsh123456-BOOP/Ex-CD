"""Selective-prediction / risk-coverage evaluation for CertCD.

Given a per-prediction confidence score (higher = keep) and per-prediction errors, compute
the risk-coverage curve and AURC. The headline CertCD claim is EXCESS AURC: the certificate
must achieve lower AURC (and higher selective AUC) than the count and learned-confidence
baselines — not merely "risk drops as coverage drops" (true of any non-random score).
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import roc_auc_score


def risk_coverage(scores: np.ndarray, errors: np.ndarray):
    """Return (coverages, risk_at_coverage, aurc). Lower AURC is better."""
    scores = np.asarray(scores)
    errors = np.asarray(errors, dtype=np.float64)
    order = np.argsort(-scores, kind="stable")  # most confident first
    e = errors[order]
    n = len(e)
    coverages = np.arange(1, n + 1) / n
    risk = np.cumsum(e) / np.arange(1, n + 1)
    aurc = float(np.mean(risk))
    return coverages, risk, aurc


def selective_auc(scores: np.ndarray, labels: np.ndarray, probs: np.ndarray, coverages=(0.5, 0.7, 0.9, 1.0)) -> Dict[str, float]:
    """AUC on the most-confident top-c fraction, for each coverage c."""
    order = np.argsort(-np.asarray(scores), kind="stable")
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    n = len(scores)
    out = {}
    for c in coverages:
        k = max(2, int(round(c * n)))
        idx = order[:k]
        try:
            out[f"{c:.2f}"] = float(roc_auc_score(labels[idx], probs[idx]))
        except Exception:
            out[f"{c:.2f}"] = float("nan")
    return out


def evaluate_methods(score_dict: Dict[str, np.ndarray], labels: np.ndarray, probs: np.ndarray) -> Dict:
    """Compute AURC + selective AUC for every abstention method; report excess AURC of the
    certificate over each baseline (positive = certificate better)."""
    preds = (np.asarray(probs) >= 0.5).astype(np.float64)
    errors = (preds != np.asarray(labels)).astype(np.float64)

    per_method = {}
    for name, sc in score_dict.items():
        _, _, aurc = risk_coverage(sc, errors)
        per_method[name] = {
            "aurc": aurc,
            "selective_auc": selective_auc(sc, labels, probs),
        }

    excess = {}
    if "certificate" in per_method:
        cert_aurc = per_method["certificate"]["aurc"]
        for name, m in per_method.items():
            if name == "certificate":
                continue
            excess[name] = float(m["aurc"] - cert_aurc)  # >0 => certificate has lower (better) AURC
    return {"per_method": per_method, "certificate_excess_aurc_vs": excess}
