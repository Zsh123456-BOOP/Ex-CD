"""Instance-level identifiability certificate for cognitive diagnosis.

For each (student, concept) cell we estimate whether that mastery coordinate is
statistically identifiable given THAT student's realized item coverage and the Q-matrix
structure (Gu-style discrete-CDM identifiability intuition, operationalised as a coverage
geometry score). This is CertCD's core contribution: instead of fighting the information
limit (rare-concept mastery is unrecoverable), we CERTIFY which cells are trustworthy and
abstain on the rest.

Ingredients per cell (s, k), all from the TRAIN split + the Q-matrix + learned item params:
  - count_{s,k}   : number of train items the student answered that tag concept k
  - purity_{s,k}  : sum over those items of 1/|concepts(item)|  (a pure k-item is most informative)
  - dwp_{s,k}     : discrimination-weighted purity (informative items with high discrimination
                    pin down mastery better) — uses the trained model's item discriminations
  - concept_ident_k : Q-structural gate — concept k must have a pure item and >= 2 items at all,
                    otherwise NO student's k-mastery is identifiable regardless of coverage
"""
from __future__ import annotations

from typing import List

import numpy as np


def q_structure(vocab):
    """Return (items_per_concept[K], has_pure[K] bool, pure_item[E] bool, concepts_per_item[E])."""
    Q = vocab.q_matrix  # [E, K], row 0 = UNK exercise (all zeros)
    items_per_concept = Q.sum(axis=0)
    concepts_per_item = Q.sum(axis=1)
    pure_item = concepts_per_item == 1.0
    if pure_item.any():
        has_pure = Q[pure_item].sum(axis=0) > 0
    else:
        has_pure = np.zeros(vocab.num_concepts, dtype=bool)
    return items_per_concept, has_pure, pure_item, concepts_per_item


def build_coverage(student, exercise, concept_lists, num_students, num_concepts, disc_by_exercise=None):
    """Per-(student,concept) coverage arrays from TRAIN interactions.

    Returns (count[S,K], purity[S,K], dwp[S,K] or None).
    """
    count = np.zeros((num_students, num_concepts), dtype=np.float32)
    purity = np.zeros((num_students, num_concepts), dtype=np.float32)
    dwp = np.zeros((num_students, num_concepts), dtype=np.float32) if disc_by_exercise is not None else None
    for i in range(len(student)):
        cl = concept_lists[i]
        if not cl:
            continue
        s = int(student[i])
        w = 1.0 / len(cl)
        d = float(disc_by_exercise[int(exercise[i])]) if disc_by_exercise is not None else 1.0
        for k in cl:
            count[s, k] += 1.0
            purity[s, k] += w
            if dwp is not None:
                dwp[s, k] += w * d
    return count, purity, dwp


def certificate_scores(count, purity, dwp, vocab, n_min: int = 2):
    """Return (score[S,K] continuous, cert[S,K] binary in {0,1}, concept_ident[K]).

    score: higher = more identifiable (continuous, for ranking / selective prediction).
    cert : 1 if the cell is certified identifiable (enough informative coverage AND the concept
           is Q-structurally identifiable), else 0.
    """
    items_per_concept, has_pure, _, _ = q_structure(vocab)
    concept_ident = (has_pure & (items_per_concept >= 2)).astype(np.float32)  # [K]
    base = dwp if dwp is not None else purity
    score = np.log1p(base) * concept_ident[None, :]
    cert = ((count >= n_min) & (concept_ident[None, :] > 0)).astype(np.float32)
    return score, cert, concept_ident


def per_prediction(cell_values: np.ndarray, student, concept_lists, agg: str = "mean") -> np.ndarray:
    """Map a per-(student,concept) array to a per-prediction array for the test set.

    For a test row (student s, concept set C), aggregate cell_values[s, k] over k in C.
    agg='mean' (default) or 'min' (conservative — abstain unless ALL concepts are covered).
    Empty concept set -> 0 (abstain).
    """
    out = np.zeros(len(student), dtype=np.float32)
    for i in range(len(student)):
        cl = concept_lists[i]
        if not cl:
            out[i] = 0.0
            continue
        vals = cell_values[int(student[i]), cl]
        out[i] = float(vals.mean() if agg == "mean" else vals.min())
    return out
