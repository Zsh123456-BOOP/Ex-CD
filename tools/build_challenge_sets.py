#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from retricd.datasets import REGIME_NAMES, load_prefix_datasets


def main() -> None:
    parser = argparse.ArgumentParser(description="Export query ids for RetriCD challenge subsets.")
    parser.add_argument("--data-root", default="data_retricd")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--out-dir", default="outputs/challenge_sets")
    parser.add_argument("--max-history-len", type=int, default=128)
    parser.add_argument("--max-concepts", type=int, default=8)
    args = parser.parse_args()

    datasets, encoded = load_prefix_datasets(args.data_root, args.dataset, args.max_history_len, args.max_concepts)
    ds = datasets[args.split]
    rows = []
    for local_idx, row_index in enumerate(ds.target_indices):
        subset = REGIME_NAMES.get(int(ds.regimes[local_idx]), "unknown")
        rows.append(
            {
                "split": args.split,
                "local_index": int(local_idx),
                "encoded_row_index": int(row_index),
                "student_id": encoded.raw_students[row_index],
                "exercise_id": encoded.raw_exercises[row_index],
                "concepts": encoded.raw_concept_text[row_index],
                "hist_len": int(ds.history_lens[local_idx]),
                "subset": subset,
                "is_short_history": bool(ds.history_lens[local_idx] <= encoded.short_history_threshold),
            }
        )
    out_dir = Path(args.out_dir) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / f"{args.split}_challenge_sets.csv", index=False)
    print(f"[OK] wrote {out_dir / f'{args.split}_challenge_sets.csv'}")


if __name__ == "__main__":
    main()

