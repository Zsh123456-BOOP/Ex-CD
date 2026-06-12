from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd

from retricd.config import RAW_DATASETS
from retricd.features import cold_stats
from retricd.preprocess import CANONICAL_COLUMNS, concatenate_splits, load_source_splits


def _stable_hash(value, seed: int) -> int:
    payload = f"{seed}:{value}".encode("utf-8")
    return int(hashlib.md5(payload).hexdigest(), 16)


def downsample_students(df: pd.DataFrame, target_rows: int, seed: int) -> pd.DataFrame:
    if target_rows <= 0 or len(df) <= target_rows:
        return df.copy()
    sizes = df.groupby("stu_id").size().reset_index(name="rows")
    sizes["_key"] = sizes["stu_id"].map(lambda x: _stable_hash(x, seed))
    sizes = sizes.sort_values(["_key", "stu_id"])
    keep = []
    total = 0
    for _, row in sizes.iterrows():
        keep.append(row["stu_id"])
        total += int(row["rows"])
        if total >= target_rows:
            break
    return df[df["stu_id"].isin(set(keep))].copy()


def split_per_student_chrono(df: pd.DataFrame, ratio: Tuple[float, float, float]) -> Dict[str, pd.DataFrame]:
    buckets = {"train": [], "valid": [], "test": []}
    sort_cols = ["source_split_order", "source_order"] if "source_split_order" in df.columns else None
    for _, group in df.groupby("stu_id", sort=False):
        group = group.sort_values(sort_cols) if sort_cols else group
        n = len(group)
        if n == 1:
            cuts = (1, 1)
        elif n == 2:
            cuts = (1, 1)
        else:
            train_end = max(1, int(n * ratio[0]))
            valid_end = max(train_end + 1, train_end + int(n * ratio[1]))
            valid_end = min(valid_end, n - 1)
            cuts = (train_end, valid_end)
        buckets["train"].append(group.iloc[: cuts[0]])
        buckets["valid"].append(group.iloc[cuts[0] : cuts[1]])
        buckets["test"].append(group.iloc[cuts[1] :])
    return {k: pd.concat(v, ignore_index=True) if v else pd.DataFrame(columns=df.columns) for k, v in buckets.items()}


def split_per_student_static(df: pd.DataFrame, ratio: Tuple[float, float, float], seed: int) -> Dict[str, pd.DataFrame]:
    buckets = {"train": [], "valid": [], "test": []}
    for student, group in df.groupby("stu_id", sort=False):
        group = group.copy()
        group["_rand"] = [_stable_hash(f"{student}:{i}", seed) for i in range(len(group))]
        group = group.sort_values("_rand").drop(columns=["_rand"])
        n = len(group)
        train_end = max(1, int(n * ratio[0])) if n else 0
        valid_end = min(n, max(train_end, train_end + int(n * ratio[1])))
        buckets["train"].append(group.iloc[:train_end])
        buckets["valid"].append(group.iloc[train_end:valid_end])
        buckets["test"].append(group.iloc[valid_end:])
    return {k: pd.concat(v, ignore_index=True) if v else pd.DataFrame(columns=df.columns) for k, v in buckets.items()}


def build_dataset(
    source_root: str,
    output_root: str,
    dataset: str,
    *,
    output_name: str,
    split_mode: str,
    ratio: Tuple[float, float, float],
    seed: int,
    target_rows: int = 0,
) -> Path:
    source_splits = load_source_splits(source_root, dataset)
    df = concatenate_splits(source_splits)
    original_rows = len(df)
    if target_rows > 0:
        df = downsample_students(df, target_rows, seed)
    if split_mode == "chronological_per_student":
        splits = split_per_student_chrono(df, ratio)
    elif split_mode == "static_random":
        splits = split_per_student_static(df, ratio, seed)
    else:
        raise ValueError(f"unsupported split_mode: {split_mode}")

    out_dir = Path(output_root) / output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, part in splits.items():
        part[CANONICAL_COLUMNS].to_csv(out_dir / f"{split}.csv", index=False)

    manifest = {
        "source_dataset": dataset,
        "output_dataset": output_name,
        "source_root": source_root,
        "split_mode": split_mode,
        "split_ratio": list(ratio),
        "seed": seed,
        "original_rows": int(original_rows),
        "output_rows": int(sum(len(x) for x in splits.values())),
        "target_rows": int(target_rows),
        "rows": {split: int(len(part)) for split, part in splits.items()},
        "students": {split: int(part["stu_id"].nunique()) for split, part in splits.items()},
        "cold_handling": {
            "policy": "train-only maps with PAD/UNK ids; cold valid/test exercises or concepts are retained and mapped to UNK at load time",
            "valid": cold_stats(splits["valid"], splits["train"]),
            "test": cold_stats(splits["test"], splits["train"]),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return out_dir


def build_all(
    source_root: str,
    output_root: str,
    datasets: Iterable[str],
    split_mode: str,
    ratio: Tuple[float, float, float],
    seed: int,
    nips34_match_junyi: bool,
) -> None:
    target_rows = 0
    if nips34_match_junyi:
        junyi = concatenate_splits(load_source_splits(source_root, "junyi"))
        target_rows = len(junyi)
    for dataset in datasets:
        output_name = "nips34_retricd_small" if dataset == "nips34" and nips34_match_junyi else dataset
        rows = target_rows if dataset == "nips34" and nips34_match_junyi else 0
        path = build_dataset(
            source_root,
            output_root,
            dataset,
            output_name=output_name,
            split_mode=split_mode,
            ratio=ratio,
            seed=seed,
            target_rows=rows,
        )
        print(f"[OK] built {dataset} -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RetriCD chronological/static splits with cold handling manifests.")
    parser.add_argument("--source-root", default="data_source")
    parser.add_argument("--output-root", default="data_retricd")
    parser.add_argument("--datasets", nargs="+", default=list(RAW_DATASETS))
    parser.add_argument("--split-mode", choices=["chronological_per_student", "static_random"], default="chronological_per_student")
    parser.add_argument("--ratio", nargs=3, type=float, default=(0.7, 0.1, 0.2))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nips34-match-junyi", action="store_true")
    args = parser.parse_args()
    build_all(
        args.source_root,
        args.output_root,
        args.datasets,
        args.split_mode,
        tuple(args.ratio),
        args.seed,
        args.nips34_match_junyi,
    )


if __name__ == "__main__":
    main()
