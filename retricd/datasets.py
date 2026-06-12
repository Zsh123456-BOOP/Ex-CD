from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from retricd.features import PAD_ID, UNK_ID, FeatureMaps, build_train_only_features, concept_text, parse_concepts
from retricd.preprocess import read_interactions


REGIME_NAMES = {
    0: "direct_seen",
    1: "partial_overlap",
    2: "target_concept_unseen",
}


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
    raw_concept_text: np.ndarray
    feature_maps: FeatureMaps
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
            "row_index": np.int64(target),
            "query_exercise_id": np.int64(self.encoded.exercise_ids[target]),
            "query_concept_ids": self.encoded.concept_ids[target],
            "query_concept_mask": self.encoded.concept_mask[target],
            "query_difficulty": np.float32(self.encoded.difficulties[target]),
            "label": np.float32(self.encoded.labels[target]),
            "hist_row_index": hist_idx.astype(np.int64),
            "hist_exercise_id": hist_exer,
            "hist_concept_ids": hist_concepts,
            "hist_concept_mask": hist_cmask,
            "hist_difficulty": hist_diff,
            "hist_correct": hist_correct,
            "hist_recency": recency,
            "hist_distance": distances,
            "hist_mask": hist_mask,
            "hist_len": np.int64(hist_len),
            "regime": np.int64(self.regimes[idx]),
            "is_short_history": np.bool_(hist_len <= self.encoded.short_history_threshold),
        }


def _encode_frames(frames: Dict[str, pd.DataFrame], max_concepts: int) -> Tuple[EncodedData, pd.DataFrame]:
    features = build_train_only_features(frames["train"])
    merged = []
    for split_id, split in enumerate(("train", "valid", "test")):
        part = frames[split].copy()
        part["_split"] = split_id
        merged.append(part)
    all_df = pd.concat(merged, ignore_index=True)
    raw_concepts = [parse_concepts(x) for x in all_df["cpt_seq"].values]
    n = len(all_df)

    exercise_ids = all_df["exer_id"].map(lambda x: features.exercise_map.get(x, UNK_ID)).to_numpy(np.int64)
    concept_ids = np.zeros((n, max_concepts), dtype=np.int64)
    concept_mask = np.zeros((n, max_concepts), dtype=np.bool_)
    for i, concepts in enumerate(raw_concepts):
        mapped = [features.concept_map.get(cid, UNK_ID) for cid in concepts[:max_concepts]]
        if mapped:
            concept_ids[i, : len(mapped)] = mapped
            concept_mask[i, : len(mapped)] = True
    labels = all_df["label"].astype(np.float32).to_numpy()
    difficulties = np.array([features.difficulty_by_exer.get(exer, features.global_difficulty) for exer in exercise_ids], dtype=np.float32)
    encoded = EncodedData(
        exercise_ids=exercise_ids,
        concept_ids=concept_ids,
        concept_mask=concept_mask,
        labels=labels,
        difficulties=difficulties,
        raw_students=all_df["stu_id"].to_numpy(),
        raw_exercises=all_df["exer_id"].to_numpy(),
        raw_concepts=raw_concepts,
        raw_concept_text=np.asarray([concept_text(x) for x in raw_concepts]),
        feature_maps=features,
        num_exercises=len(features.exercise_map) + 2,
        num_concepts=len(features.concept_map) + 2,
        short_history_threshold=0,
    )
    return encoded, all_df


def load_prefix_datasets(
    data_root: str,
    dataset: str,
    max_history_len: int,
    max_concepts: int,
) -> Tuple[Dict[str, PrefixDataset], EncodedData]:
    data_dir = Path(data_root) / dataset
    frames = {split: read_interactions(data_dir / f"{split}.csv") for split in ("train", "valid", "test")}
    encoded, all_df = _encode_frames(frames, max_concepts)

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

        query_concepts = set(cid for cid in encoded.raw_concepts[idx] if cid in encoded.feature_maps.concept_map)
        if query_concepts and query_concepts.issubset(concept_union):
            regimes[split][pos] = 0
        elif query_concepts and query_concepts.intersection(concept_union):
            regimes[split][pos] = 1
        else:
            regimes[split][pos] = 2
        history.append(idx)
        concept_union.update(query_concepts)

    encoded.short_history_threshold = int(np.quantile(hist_lens["train"], 0.25)) if len(hist_lens["train"]) else 0
    datasets = {
        split: PrefixDataset(encoded, targets[split], histories[split], hist_lens[split], regimes[split])
        for split in ("train", "valid", "test")
    }
    return datasets, encoded

