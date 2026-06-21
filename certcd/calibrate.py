"""Calibration (temperature scaling) and ECE for CertCD.

Temperature is fit on the validation split; ECE is reported overall and split by the
certificate (identifiable vs abstained cells) to show that calibration error concentrates
where the certificate says mastery is unreliable.
"""
from __future__ import annotations

import numpy as np


def prob_to_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return np.log(p / (1 - p))


def _bce(logits: np.ndarray, labels: np.ndarray, T: float) -> float:
    p = 1.0 / (1.0 + np.exp(-logits / T))
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))


def fit_temperature(probs_valid: np.ndarray, labels_valid: np.ndarray) -> float:
    """Fit a single temperature T>0 minimising validation BCE (coarse grid + refine)."""
    logits = prob_to_logit(probs_valid)
    labels = np.asarray(labels_valid, dtype=np.float64)
    grid = np.linspace(0.5, 5.0, 46)
    best = min(grid, key=lambda T: _bce(logits, labels, T))
    fine = np.linspace(max(0.05, best - 0.1), best + 0.1, 21)
    best = min(fine, key=lambda T: _bce(logits, labels, T))
    return float(best)


def apply_temperature(probs: np.ndarray, T: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-prob_to_logit(probs) / T))


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error (equal-width bins)."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if len(probs) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, bins) - 1, 0, n_bins - 1)
    total = len(probs)
    err = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        conf = probs[m].mean()
        acc = labels[m].mean()
        err += (m.sum() / total) * abs(conf - acc)
    return float(err)
