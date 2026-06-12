from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd


CANONICAL_COLUMNS = ["stu_id", "exer_id", "cpt_seq", "label"]


def read_interactions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(CANONICAL_COLUMNS).difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    out = df[CANONICAL_COLUMNS].copy()
    out["label"] = out["label"].astype(float)
    return out


def load_source_splits(source_root: str, dataset: str) -> Dict[str, pd.DataFrame]:
    root = Path(source_root) / dataset
    return {split: read_interactions(root / f"{split}.csv") for split in ("train", "valid", "test")}


def concatenate_splits(splits: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for split_order, split in enumerate(("train", "valid", "test")):
        part = splits[split].copy()
        part["source_split"] = split
        part["source_split_order"] = split_order
        part["source_order"] = range(len(part))
        frames.append(part)
    return pd.concat(frames, ignore_index=True)

