"""Aggregate Ex-CD metrics.json files under an output dir into one CSV table.

Run after a batch of experiments so results can be pulled and compared at a glance:
    python -m excd.summarize --output-dir outputs --out outputs/summary.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

COLUMNS = [
    "dataset", "model", "variant", "seed",
    "auc", "acc", "rmse", "bce", "doa",
    "bottom_group_doa", "top_group_doa", "group_gap",
    "tail_decile_doa", "head_decile_doa", "head_minus_tail_gap", "n_measurable_deciles",
    "best_valid_auc", "best_epoch", "n",
]


def collect(output_dir: str):
    rows = []
    for path in sorted(glob.glob(os.path.join(output_dir, "**", "metrics.json"), recursive=True)):
        try:
            with open(path) as f:
                m = json.load(f)
        except Exception:
            continue
        test = m.get("test", {})
        strat = test.get("exposure_stratified_doa", {}) or {}
        rows.append(
            {
                "dataset": m.get("dataset"),
                "model": m.get("model"),
                "variant": m.get("variant"),
                "seed": m.get("seed"),
                "auc": test.get("auc"),
                "acc": test.get("acc"),
                "rmse": test.get("rmse"),
                "bce": test.get("bce"),
                "doa": test.get("doa"),
                "bottom_group_doa": strat.get("bottom_group_doa"),
                "top_group_doa": strat.get("top_group_doa"),
                "group_gap": strat.get("group_gap"),
                "tail_decile_doa": strat.get("tail_decile_doa"),
                "head_decile_doa": strat.get("head_decile_doa"),
                "head_minus_tail_gap": strat.get("head_minus_tail_gap"),
                "n_measurable_deciles": strat.get("n_measurable_deciles"),
                "best_valid_auc": m.get("best_valid_auc"),
                "best_epoch": m.get("best_epoch"),
                "n": test.get("n"),
            }
        )
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=str, default="outputs")
    p.add_argument("--out", type=str, default="outputs/summary.csv")
    args = p.parse_args()
    rows = collect(args.output_dir)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {len(rows)} rows -> {args.out}")
    for r in rows:
        print(
            f"  {r['dataset']:<22} {r['model']:<6} {r['variant']:<8} "
            f"auc={r['auc']} doa={r['doa']} group_gap={r['group_gap']}"
        )


if __name__ == "__main__":
    main()
