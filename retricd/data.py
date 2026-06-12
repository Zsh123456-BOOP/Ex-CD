from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


PAD_ID = 0
UNK_ID = 1
REGIME_NAMES = {
    0: "direct_seen",
    1: "partial_overlap",
    2: "target_concept_unseen",
}


def parse_concepts(value) -> List[int]:
    if pd.isna(value):
        return []
    if isinstance(value, str):
        return [int(x) for x in value.split(",") if x.strip()]
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [int(x) for x in value]
    return [int(value)]


def _sorted_values(values: Iterable) -> List:
    return sorted(set(values), key=lambda x: (str(type(x)), str(x)))


@dataclass
class EncodedData:
    exercise_ids: np.ndarray
    concept_ids: np.ndarray
    concept_mask: np.ndarray
    labels: np.ndarray
    difficulties: np.ndarray
    raw_students: np.ndarray
    raw_exercises: np.ndarray
    raw_concepts: List[List[int]]
    num_exercises: int
    num_concepts: int
    short_history_threshold: int


class PrefixDataset(Dataset):
    def __init__(
        self,
        encoded: EncodedData,
        target_indices: np.ndarray,
        history_indices: np.ndarray,
        history_lens: np.ndarray,
        regimes: np.ndarray,
    ):
        self.encoded = encoded
        self.target_indices = target_indices.astype(np.int64)
        self.history_indices = history_indices.astype(np.int64)
        self.history_lens = history_lens.astype(np.int64)
        self.regimes = regimes.astype(np.int64)

    def __len__(self) -> int:
        return int(self.target_indices.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        target = int(self.target_indices[idx])
        hist_idx = self.history_indices[idx]
        hist_mask = hist_idx >= 0
        safe_hist = np.where(hist_mask, hist_idx, 0)
        hist_len = int(self.history_lens[idx])
        distances = np.zeros_like(hist_idx, dtype=np.float32)
        if hist_mask.any():
            valid_count = int(hist_mask.sum())
            distances[hist_mask] = np.arange(valid_count, 0, -1, dtype=np.float32)
        recency = np.zeros_like(distances, dtype=np.float32)
        recency[hist_mask] = 1.0 / np.log2(distances[hist_mask] + 2.0)

        hist_exer = self.encoded.exercise_ids[safe_hist].copy()
        hist_concepts = self.encoded.concept_ids[safe_hist].copy()
        hist_cmask = self.encoded.concept_mask[safe_hist].copy()
        hist_diff = self.encoded.difficulties[safe_hist].copy()
        hist_correct = self.encoded.labels[safe_hist].copy()
        hist_exer[~hist_mask] = PAD_ID
        hist_concepts[~hist_mask] = PAD_ID
        hist_cmask[~hist_mask] = False
        hist_diff[~hist_mask] = 0.0
        hist_correct[~hist_mask] = 0.0

        return {
            "query_exercise_id": np.int64(self.encoded.exercise_ids[target]),
            "query_concept_ids": self.encoded.concept_ids[target],
            "query_concept_mask": self.encoded.concept_mask[target],
            "query_difficulty": np.float32(self.encoded.difficulties[target]),
            "label": np.float32(self.encoded.labels[target]),
            "hist_exercise_id": hist_exer,
            "hist_concept_ids": hist_concepts,
            "hist_concept_mask": hist_cmask,
            "hist_difficulty": hist_diff,
            "hist_correct": hist_correct,
            "hist_recency": recency,
            "hist_mask": hist_mask,
            "hist_len": np.int64(hist_len),
            "regime": np.int64(self.regimes[idx]),
            "is_short_history": np.bool_(hist_len <= self.encoded.short_history_threshold),
            "raw_query_exercise": np.int64(self.encoded.raw_exercises[target]),
        }


def _read_split(data_dir: Path, split: str) -> pd.DataFrame:
    path = data_dir / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing split file: {path}")
    df = pd.read_csv(path)
    expected = {"stu_id", "exer_id", "cpt_seq", "label"}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df[["stu_id", "exer_id", "cpt_seq", "label"]].copy()


def load_prefix_datasets(
    data_root: str,
    dataset: str,
    max_history_len: int,
    max_concepts: int,
) -> Tuple[Dict[str, PrefixDataset], EncodedData]:
    data_dir = Path(data_root) / dataset
    frames = {split: _read_split(data_dir, split) for split in ("train", "valid", "test")}

    train = frames["train"]
    train_concepts = [parse_concepts(x) for x in train["cpt_seq"].values]
    concept_map = {cid: i + 2 for i, cid in enumerate(_sorted_values(c for row in train_concepts for c in row))}
    exercise_map = {eid: i + 2 for i, eid in enumerate(_sorted_values(train["exer_id"].values))}

    train_exer = train["exer_id"].map(lambda x: exercise_map.get(x, UNK_ID)).to_numpy(np.int64)
    sums = {}
    counts = {}
    for exer, label in zip(train_exer, train["label"].astype(float).values):
        sums[exer] = sums.get(exer, 0.0) + float(label)
        counts[exer] = counts.get(exer, 0) + 1
    global_correct = float(train["label"].astype(float).mean()) if len(train) else 0.5
    difficulty_by_exer = {UNK_ID: 1.0 - global_correct, PAD_ID: 0.0}
    for exer, total in sums.items():
        difficulty_by_exer[exer] = 1.0 - total / max(1, counts[exer])

    merged = []
    for split_id, split in enumerate(("train", "valid", "test")):
        part = frames[split].copy()
        part["_split"] = split_id
        merged.append(part)
    all_df = pd.concat(merged, ignore_index=True)
    raw_concepts = [parse_concepts(x) for x in all_df["cpt_seq"].values]
    n = len(all_df)

    exercise_ids = all_df["exer_id"].map(lambda x: exercise_map.get(x, UNK_ID)).to_numpy(np.int64)
    concept_ids = np.zeros((n, max_concepts), dtype=np.int64)
    concept_mask = np.zeros((n, max_concepts), dtype=np.bool_)
    for i, concepts in enumerate(raw_concepts):
        mapped = [concept_map.get(cid, UNK_ID) for cid in concepts[:max_concepts]]
        if mapped:
            concept_ids[i, : len(mapped)] = mapped
            concept_mask[i, : len(mapped)] = True
    labels = all_df["label"].astype(np.float32).to_numpy()
    difficulties = np.array([difficulty_by_exer.get(exer, difficulty_by_exer[UNK_ID]) for exer in exercise_ids], dtype=np.float32)

    split_sizes = {split: int((all_df["_split"] == i).sum()) for i, split in enumerate(("train", "valid", "test"))}
    targets = {split: np.zeros(split_sizes[split], dtype=np.int64) for split in split_sizes}
    histories = {split: np.full((split_sizes[split], max_history_len), -1, dtype=np.int32) for split in split_sizes}
    hist_lens = {split: np.zeros(split_sizes[split], dtype=np.int64) for split in split_sizes}
    regimes = {split: np.full(split_sizes[split], 2, dtype=np.int64) for split in split_sizes}
    counters = {split: 0 for split in split_sizes}
    split_names = {0: "train", 1: "valid", 2: "test"}

    history_by_student: Dict[object, List[int]] = {}
    concept_union_by_student: Dict[object, set] = {}
    for idx, row in all_df.iterrows():
        split = split_names[int(row["_split"])]
        pos = counters[split]
        counters[split] += 1
        student = row["stu_id"]
        history = history_by_student.setdefault(student, [])
        concept_union = concept_union_by_student.setdefault(student, set())
        selected = history[-max_history_len:]
        if selected:
            histories[split][pos, -len(selected) :] = selected
        hist_lens[split][pos] = len(history)
        targets[split][pos] = idx

        query_concepts = set(cid for cid in raw_concepts[idx] if cid in concept_map)
        if query_concepts and query_concepts.issubset(concept_union):
            regimes[split][pos] = 0
        elif query_concepts and query_concepts.intersection(concept_union):
            regimes[split][pos] = 1
        else:
            regimes[split][pos] = 2

        history.append(idx)
        concept_union.update(query_concepts)

    train_threshold = int(np.quantile(hist_lens["train"], 0.25)) if len(hist_lens["train"]) else 0
    encoded = EncodedData(
        exercise_ids=exercise_ids,
        concept_ids=concept_ids,
        concept_mask=concept_mask,
        labels=labels,
        difficulties=difficulties,
        raw_students=all_df["stu_id"].to_numpy(),
        raw_exercises=all_df["exer_id"].to_numpy(),
        raw_concepts=raw_concepts,
        num_exercises=len(exercise_map) + 2,
        num_concepts=len(concept_map) + 2,
        short_history_threshold=train_threshold,
    )
    datasets = {
        split: PrefixDataset(encoded, targets[split], histories[split], hist_lens[split], regimes[split])
        for split in ("train", "valid", "test")
    }
    return datasets, encoded

