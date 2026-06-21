"""Synthetic DINA-style cognitive-diagnosis generator with KNOWN true mastery.

Used to validate the certificate: we control the Q-matrix completeness and per-student item
coverage, train the real CDM on the generated responses, then check whether the certificate
(computed only from coverage geometry + Q, never from the truth) predicts the cells where the
LEARNED mastery is actually correct. That is the non-circular novelty evidence.

Emits the same CSV schema as the real benchmarks (stu_id,exer_id,cpt_seq,label) so the whole
excd/certcd pipeline runs on it unchanged. Student/concept ids are 0..N-1 and all appear in
train, so build_vocab's mapping is the identity for them (M_true stays aligned).
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def generate(
    num_students: int = 2000,
    num_concepts: int = 20,
    num_items: int = 400,
    max_concepts_per_item: int = 3,
    pure_frac: float = 0.5,
    entangled_frac: float = 0.3,
    items_per_student: int = 60,
    guess: float = 0.2,
    slip: float = 0.1,
    seed: int = 42,
) -> Dict:
    rng = np.random.default_rng(seed)

    # A fraction of concepts are ENTANGLED: they never appear as a pure item, so they are
    # structurally weakly identifiable no matter how many times a student practises them. This is
    # exactly where the certificate (structural gate + purity) should beat a raw coverage count.
    n_ent = int(round(entangled_frac * num_concepts))
    entangled = set(rng.choice(num_concepts, size=n_ent, replace=False).tolist()) if n_ent else set()
    clean_concepts = [k for k in range(num_concepts) if k not in entangled]

    Q = np.zeros((num_items, num_concepts), dtype=np.float32)
    for j in range(num_items):
        if rng.random() < pure_frac and clean_concepts:
            Q[j, int(rng.choice(clean_concepts))] = 1.0  # pure item only on a clean concept
        else:
            m = int(rng.integers(2, max_concepts_per_item + 1))
            for k in rng.choice(num_concepts, size=min(m, num_concepts), replace=False):
                Q[j, k] = 1.0
    # guarantee every concept has >= 2 items; entangled concepts get MULTI-concept items only
    for k in range(num_concepts):
        while Q[:, k].sum() < 2:
            j = int(rng.integers(num_items))
            Q[j, k] = 1.0
            if k in entangled and Q[j].sum() == 1:  # keep entangled concepts non-pure
                others = [c for c in range(num_concepts) if c != k]
                Q[j, int(rng.choice(others))] = 1.0

    # --- true mastery: ability + concept-specific structure (mirrors the real ability+signal mix) ---
    ability = rng.normal(0, 1, size=num_students)
    concept_easiness = rng.normal(0, 1, size=num_concepts)
    p_master = 1.0 / (1.0 + np.exp(-(ability[:, None] + concept_easiness[None, :])))
    M_true = (rng.random((num_students, num_concepts)) < p_master).astype(np.int8)

    # --- item popularity (mild skew) -> per-cell coverage varies across (student, concept) ---
    pop = rng.random(num_items) ** 1.5
    pop = pop / pop.sum()

    rows = []
    item_concepts = [np.where(Q[j] > 0)[0].tolist() for j in range(num_items)]
    for s in range(num_students):
        items = rng.choice(num_items, size=min(items_per_student, num_items), replace=False, p=pop)
        for j in items:
            cs = item_concepts[j]
            eta = 1 if all(M_true[s, k] == 1 for k in cs) else 0
            p_correct = (1 - slip) if eta == 1 else guess
            label = 1.0 if rng.random() < p_correct else 0.0
            rows.append((s, int(j), ",".join(str(k) for k in cs), label))

    df = pd.DataFrame(rows, columns=["stu_id", "exer_id", "cpt_seq", "label"])
    # per-student random split (synthetic -> static random is fine); keep all students+concepts in train
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df)
    n_tr, n_va = int(0.7 * n), int(0.15 * n)
    train_df = df.iloc[:n_tr].reset_index(drop=True)
    valid_df = df.iloc[n_tr:n_tr + n_va].reset_index(drop=True)
    test_df = df.iloc[n_tr + n_va:].reset_index(drop=True)

    return {"train_df": train_df, "valid_df": valid_df, "test_df": test_df, "M_true": M_true, "Q_true": Q}
