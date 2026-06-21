"""Per-prediction abstention/confidence scores for selective prediction.

Convention: HIGHER score = more confident = keep first when computing risk-coverage.
These are the competitors the CertCD certificate must beat (the kill-switch):
  - count           : per-cell train coverage count (the key TRIVIAL baseline)
  - prob_margin     : |p - 0.5| from the model's own output (trivial model confidence)
  - mc_dropout      : negative MC-dropout variance of the predicted prob (LEARNED epistemic
                      uncertainty — stands in for ReliCD-style confidence)
  - ability         : the student's mean train accuracy, broadcast to each prediction
                      (the unidimensional-ability baseline that should LOSE)
  - random          : reference lower bound
  - certificate     : CertCD (computed in certificate.py)
"""
from __future__ import annotations

import numpy as np

from certcd.certificate import per_prediction


def count_scores(count, student, concept_lists):
    return per_prediction(count, student, concept_lists, agg="mean")


def prob_margin_scores(probs: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(probs) - 0.5)


def ability_scores(train_student, train_label, num_students, test_student):
    acc = np.full(num_students, 0.5, dtype=np.float32)
    s = np.zeros(num_students, dtype=np.float64)
    c = np.zeros(num_students, dtype=np.float64)
    np.add.at(s, train_student, train_label)
    np.add.at(c, train_student, 1.0)
    nz = c > 0
    acc[nz] = (s[nz] / c[nz]).astype(np.float32)
    # confidence proxy: distance of student ability from 0.5 (very high / very low ability
    # students are more predictable). This is intentionally weak — it is the ability baseline.
    return np.abs(acc[test_student] - 0.5)


def random_scores(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random(n).astype(np.float32)


def mc_dropout_scores(model, loader, device, passes: int = 20) -> np.ndarray:
    """Negative predictive variance under MC-dropout (higher = more confident).

    Keeps dropout active (model.train) and averages the variance of the predicted prob over
    several stochastic passes — a standard learned-uncertainty baseline (proxy for ReliCD-style
    confidence). torch is imported lazily so the rest of this module stays torch-free.
    """
    import torch

    was_training = model.training
    model.train()  # enable dropout
    probs = []
    with torch.no_grad():
        for _ in range(passes):
            chunk = []
            for b in loader:
                p, _ = model(b["student"].to(device), b["exercise"].to(device), b["concept_mask"].to(device))
                chunk.append(p.cpu().numpy())
            probs.append(np.concatenate(chunk))
    if not was_training:
        model.eval()
    probs = np.stack(probs, axis=0)  # [passes, N]
    var = probs.var(axis=0)
    return (-var).astype(np.float32)
