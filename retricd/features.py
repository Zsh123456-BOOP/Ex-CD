from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


PAD_ID = 0
UNK_ID = 1


def parse_concepts(value) -> List[int]:
    if pd.isna(value):
        return []
    if isinstance(value, str):
        return [int(x) for x in value.split(",") if x.strip()]
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [int(x) for x in value]
    return [int(value)]


def concept_text(concepts: Iterable[int]) -> str:
    return ",".join(str(int(x)) for x in concepts)


def _sorted_values(values: Iterable) -> List:
    return sorted(set(values), key=lambda x: (str(type(x)), str(x)))


@dataclass
class FeatureMaps:
    exercise_map: Dict[object, int]
    concept_map: Dict[int, int]
    difficulty_by_exer: Dict[int, float]
    global_difficulty: float


def build_train_only_features(train_df: pd.DataFrame) -> FeatureMaps:
    train_concepts = [parse_concepts(x) for x in train_df["cpt_seq"].values]
    concept_map = {cid: i + 2 for i, cid in enumerate(_sorted_values(c for row in train_concepts for c in row))}
    exercise_map = {eid: i + 2 for i, eid in enumerate(_sorted_values(train_df["exer_id"].values))}
    train_exer = train_df["exer_id"].map(lambda x: exercise_map.get(x, UNK_ID)).to_numpy(np.int64)
    sums: Dict[int, float] = {}
    counts: Dict[int, int] = {}
    for exer, label in zip(train_exer, train_df["label"].astype(float).values):
        sums[exer] = sums.get(exer, 0.0) + float(label)
        counts[exer] = counts.get(exer, 0) + 1
    global_correct = float(train_df["label"].astype(float).mean()) if len(train_df) else 0.5
    global_difficulty = 1.0 - global_correct
    difficulty_by_exer = {UNK_ID: global_difficulty, PAD_ID: 0.0}
    for exer, total in sums.items():
        difficulty_by_exer[exer] = 1.0 - total / max(1, counts[exer])
    return FeatureMaps(exercise_map, concept_map, difficulty_by_exer, global_difficulty)


def cold_stats(df: pd.DataFrame, train_df: pd.DataFrame) -> dict:
    train_exer = set(train_df["exer_id"].values)
    train_concepts = set()
    for value in train_df["cpt_seq"].values:
        train_concepts.update(parse_concepts(value))
    cold_exer = 0
    cold_concept = 0
    for _, row in df.iterrows():
        concepts = set(parse_concepts(row["cpt_seq"]))
        cold_exer += int(row["exer_id"] not in train_exer)
        cold_concept += int(bool(concepts.difference(train_concepts)))
    return {
        "rows": int(len(df)),
        "cold_exercise_rows": int(cold_exer),
        "cold_concept_rows": int(cold_concept),
        "cold_exercise_rate": float(cold_exer / max(1, len(df))),
        "cold_concept_rate": float(cold_concept / max(1, len(df))),
    }

