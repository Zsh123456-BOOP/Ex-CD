#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from retricd.datasets import REGIME_NAMES, load_prefix_datasets


def main() -> None:
    parser = argparse.ArgumentParser(description="Export RetriCD evidence-regime counts.")
    parser.add_argument("--data-root", default="data_retricd")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-dir", default="outputs/analysis")
    parser.add_argument("--max-history-len", type=int, default=128)
    parser.add_argument("--max-concepts", type=int, default=8)
    args = parser.parse_args()

    datasets, _ = load_prefix_datasets(args.data_root, args.dataset, args.max_history_len, args.max_concepts)
    rows = []
    for split, ds in datasets.items():
        total = len(ds)
        for code, name in REGIME_NAMES.items():
            rows.append({"split": split, "subset": name, "rows": int((ds.regimes == code).sum()), "total": total})
        rows.append({"split": split, "subset": "short_history", "rows": int((ds.history_lens <= ds.encoded.short_history_threshold).sum()), "total": total})
    out_dir = Path(args.out_dir) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "weak_evidence_counts.csv", index=False)
    print(f"[OK] wrote {out_dir / 'weak_evidence_counts.csv'}")


if __name__ == "__main__":
    main()

