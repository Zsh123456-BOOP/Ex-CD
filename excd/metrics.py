"""Evaluation metrics for Ex-CD.

- ``binary_metrics``: AUC / ACC / RMSE / BCE.
- ``compute_doa``: per-concept Degree of Agreement (DOA) of the mastery artifact, the
  standard same-item pairwise test (a correct responder should have higher mastery on the
  concept than an incorrect responder on the same item). This is what makes Ex-CD a
  cognitive-diagnosis paper rather than KT.
- ``stratified_doa``: DOA bucketed by per-concept train-exposure decile + the head-minus-tail
  gap. This is the Exposure-Stratified DOA — the contribution's evaluation axis (Figure-1 cliff).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, roc_auc_score


def binary_metrics(labels, probs) -> Dict[str, float]:
    y = np.asarray(labels, dtype=float)
    p = np.asarray(probs, dtype=float)
    out: Dict[str, float] = {}
    try:
        out["auc"] = float(roc_auc_score(y, p))
    except Exception:
        out["auc"] = float("nan")
    out["acc"] = float(accuracy_score(y, (p >= 0.5).astype(int)))
    out["rmse"] = float(mean_squared_error(y, p) ** 0.5)
    try:
        out["bce"] = float(log_loss(y, p, labels=[0, 1]))
    except Exception:
        out["bce"] = float("nan")
    out["n"] = int(len(y))
    return out


def compute_doa(
    mastery: np.ndarray,
    students: np.ndarray,
    exercises: np.ndarray,
    concept_lists: List[List[int]],
    labels: np.ndarray,
    num_concepts: int,
    max_responders_per_item: int = 80,
    min_pairs: int = 10,
    seed: int = 42,
) -> Tuple[np.ndarray, float]:
    """Return (per_concept_doa[num_concepts], aggregate_doa).

    For each concept k and each item j tagged with k on the eval split, form all
    (correct, incorrect) responder pairs; the correct responder should have higher mastery
    on k. DOA(k) = agreeing_pairs / total_pairs. Responders per item are capped (seeded
    subsample) to bound compute. Concepts with < ``min_pairs`` usable pairs return NaN.
    """
    rng = np.random.default_rng(seed)
    by_concept_item: Dict[int, Dict[int, List[Tuple[int, float]]]] = defaultdict(lambda: defaultdict(list))
    for s, e, cl, y in zip(students, exercises, concept_lists, labels):
        if s < 0:
            continue
        for k in cl:
            by_concept_item[k][int(e)].append((int(s), float(y)))

    doa = np.full(num_concepts, np.nan, dtype=np.float64)
    for k, items in by_concept_item.items():
        num = 0.0
        den = 0.0
        for _e, resp in items.items():
            if len(resp) > max_responders_per_item:
                pick = rng.choice(len(resp), max_responders_per_item, replace=False)
                resp = [resp[i] for i in pick]
            correct = [s for s, y in resp if y >= 0.5]
            incorrect = [s for s, y in resp if y < 0.5]
            if not correct or not incorrect:
                continue
            mc = mastery[correct, k]
            mi = mastery[incorrect, k]
            # all correct x incorrect pairs on this item
            diff = mc[:, None] - mi[None, :]
            den += diff.size
            num += float((diff > 0).sum()) + 0.5 * float((diff == 0).sum())
        if den >= min_pairs:
            doa[k] = num / den
    valid = doa[~np.isnan(doa)]
    aggregate = float(valid.mean()) if valid.size else float("nan")
    return doa, aggregate


def _group_mean(doa: np.ndarray, exposure_decile: np.ndarray, deciles) -> float:
    sel = np.isin(exposure_decile, deciles) & ~np.isnan(doa)
    return float(doa[sel].mean()) if sel.any() else float("nan")


def stratified_doa(doa: np.ndarray, exposure_decile: np.ndarray) -> Dict:
    """DOA bucketed by per-concept exposure decile (0 = rarest/tail, 9 = head).

    Reports per-decile DOA, the single-decile head-minus-tail gap, AND a robust
    bottom-group (deciles 0-2) vs top-group (deciles 7-9) gap. The rarest concepts are
    often too data-poor to measure DOA per single decile (NaN), so the GROUP gap is the
    headline; ``n_measurable_deciles`` flags how trustworthy the stratification is.
    """
    per_decile: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for d in range(10):
        sel = (exposure_decile == d) & ~np.isnan(doa)
        counts[str(d)] = int(sel.sum())
        per_decile[str(d)] = float(doa[sel].mean()) if sel.any() else float("nan")

    head = per_decile["9"]
    tail = per_decile["0"]
    single_gap = float(head - tail) if not (np.isnan(head) or np.isnan(tail)) else float("nan")

    bottom_group = _group_mean(doa, exposure_decile, [0, 1, 2])
    top_group = _group_mean(doa, exposure_decile, [7, 8, 9])
    group_gap = float(top_group - bottom_group) if not (np.isnan(top_group) or np.isnan(bottom_group)) else float("nan")

    return {
        "per_decile": per_decile,
        "decile_counts": counts,
        "n_measurable_deciles": int(sum(1 for d in range(10) if not np.isnan(per_decile[str(d)]))),
        "head_decile_doa": head,
        "tail_decile_doa": tail,
        "head_minus_tail_gap": single_gap,
        "bottom_group_doa": bottom_group,
        "top_group_doa": top_group,
        "group_gap": group_gap,
    }
