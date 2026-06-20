"""Data loading for Ex-CD.

Consumes the prepared text-free CSV splits in ``data_retricd/<dataset>/{train,valid,test}.csv``
(format ``stu_id,exer_id,cpt_seq,label``). All id maps, the Q-matrix and the per-concept
exposure propensity are built from the TRAIN split only; cold valid/test exercises map to a
reserved UNK index and cold concepts are dropped. Nothing reads valid/test statistics.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import partial
from typing import Dict, List

import numpy as np
import pandas as pd

UNK_EXERCISE = 0  # reserved index for cold / unseen exercises


def parse_concepts(value) -> List[int]:
    if value is None:
        return []
    if isinstance(value, float):
        if np.isnan(value):
            return []
        return [int(value)]
    if isinstance(value, (int, np.integer)):
        return [int(value)]
    return [int(x) for x in str(value).split(",") if x.strip() != ""]


@dataclass
class Vocab:
    student_map: Dict[int, int]
    exercise_map: Dict[int, int]
    concept_map: Dict[int, int]
    num_students: int
    num_exercises: int            # includes the UNK exercise at index 0
    num_concepts: int
    q_matrix: np.ndarray          # [num_exercises, num_concepts] float32
    concept_exposure: np.ndarray  # [num_concepts] train interaction counts
    concept_propensity: np.ndarray  # [num_concepts] exposure proxy in (0, 1]
    exposure_decile: np.ndarray   # [num_concepts] int in 0..9 (0 = rarest/tail)


def build_vocab(train_df: pd.DataFrame, propensity_floor: float = 0.05) -> Vocab:
    students = sorted(int(x) for x in train_df["stu_id"].unique())
    student_map = {s: i for i, s in enumerate(students)}

    exercises = sorted(int(x) for x in train_df["exer_id"].unique())
    exercise_map = {e: i + 1 for i, e in enumerate(exercises)}  # 0 reserved for UNK

    train_concepts = [parse_concepts(v) for v in train_df["cpt_seq"].values]
    concept_set = set()
    for cs in train_concepts:
        concept_set.update(cs)
    concepts = sorted(concept_set)
    concept_map = {c: i for i, c in enumerate(concepts)}

    num_students = len(students)
    num_exercises = len(exercises) + 1
    num_concepts = len(concepts)

    q_matrix = np.zeros((num_exercises, num_concepts), dtype=np.float32)
    exposure = np.zeros(num_concepts, dtype=np.float64)
    for cs, e in zip(train_concepts, train_df["exer_id"].values):
        em = exercise_map[int(e)]
        for c in cs:
            k = concept_map[c]
            q_matrix[em, k] = 1.0
            exposure[k] += 1.0

    max_exp = exposure.max() if exposure.size and exposure.max() > 0 else 1.0
    propensity = np.clip(exposure / max_exp, propensity_floor, 1.0).astype(np.float32)

    # Exposure deciles by ascending rank: decile 0 = rarest concepts (tail), 9 = head.
    order = np.argsort(exposure, kind="stable")
    ranks = np.empty(num_concepts, dtype=np.int64)
    ranks[order] = np.arange(num_concepts)
    denom = max(1, num_concepts)
    decile = np.minimum((ranks * 10) // denom, 9).astype(np.int64)

    return Vocab(
        student_map=student_map,
        exercise_map=exercise_map,
        concept_map=concept_map,
        num_students=num_students,
        num_exercises=num_exercises,
        num_concepts=num_concepts,
        q_matrix=q_matrix,
        concept_exposure=exposure.astype(np.float32),
        concept_propensity=propensity,
        exposure_decile=decile,
    )


class CDDataset:
    """Static cognitive-diagnosis samples: (student, exercise, concept multi-hot, label, ips_weight).

    A plain map-style dataset (``__len__`` + ``__getitem__``); torch's DataLoader consumes it
    directly, so this module stays importable without torch.
    """

    def __init__(self, df: pd.DataFrame, vocab: Vocab, use_ips: bool = False, ips_weight_cap: float = 10.0):
        df = df[df["stu_id"].isin(vocab.student_map)].reset_index(drop=True)
        self.student = df["stu_id"].map(vocab.student_map).astype(np.int64).to_numpy()
        self.exercise = (
            df["exer_id"].map(lambda e: vocab.exercise_map.get(int(e), UNK_EXERCISE)).astype(np.int64).to_numpy()
        )
        self.label = df["label"].astype(np.float32).to_numpy()
        self.concept_lists: List[List[int]] = [
            [vocab.concept_map[c] for c in parse_concepts(v) if c in vocab.concept_map]
            for v in df["cpt_seq"].values
        ]
        self.num_concepts = vocab.num_concepts

        weight = np.ones(len(df), dtype=np.float32)
        if use_ips:
            prop = vocab.concept_propensity
            for i, cl in enumerate(self.concept_lists):
                p = float(np.mean([prop[k] for k in cl])) if cl else 1.0
                weight[i] = 1.0 / max(p, 1e-6)
            # cap extreme weights (rare concepts) to avoid early-training instability, then
            # normalise so the mean training weight is 1 (keeps the loss scale comparable).
            if ips_weight_cap and ips_weight_cap > 0:
                weight = np.minimum(weight, ips_weight_cap)
            if weight.sum() > 0:
                weight *= len(weight) / float(weight.sum())
        self.weight = weight

    def __len__(self) -> int:
        return len(self.student)

    def __getitem__(self, idx: int):
        return (
            int(self.student[idx]),
            int(self.exercise[idx]),
            self.concept_lists[idx],
            float(self.label[idx]),
            float(self.weight[idx]),
        )


def _collate(batch, num_concepts: int):
    import torch

    students = torch.tensor([b[0] for b in batch], dtype=torch.long)
    exercises = torch.tensor([b[1] for b in batch], dtype=torch.long)
    labels = torch.tensor([b[3] for b in batch], dtype=torch.float32)
    weights = torch.tensor([b[4] for b in batch], dtype=torch.float32)
    mask = torch.zeros(len(batch), num_concepts, dtype=torch.float32)
    for i, b in enumerate(batch):
        if b[2]:
            mask[i, b[2]] = 1.0
    return {"student": students, "exercise": exercises, "concept_mask": mask, "label": labels, "weight": weights}


def make_collate(num_concepts: int):
    return partial(_collate, num_concepts=num_concepts)


def load_splits(dataset: str, data_root: str):
    base = os.path.join(data_root, dataset)
    train = pd.read_csv(os.path.join(base, "train.csv"))
    valid = pd.read_csv(os.path.join(base, "valid.csv"))
    test = pd.read_csv(os.path.join(base, "test.csv"))
    return train, valid, test
