"""MNAR-stress harness (Phase 2 closed loop).

Synthetically increases exposure skew by thinning a fraction ``gamma`` of train responses that
touch the rarest (bottom-decile) concepts, retrains both ``vanilla`` and ``ips`` (and optionally
``dr``), and charts how the Exposure-Stratified DOA / head-minus-tail gap degrade with gamma.
The claim passes if the tail-decile DOA gap grows for ``vanilla`` but Ex-CD (``ips``/``dr``)
keeps it smaller as gamma rises.

Example:
    python -m excd.mnar_stress --dataset assist_09 --model ncdm \
        --gammas 0.0 0.3 0.6 --variants vanilla ips --data-root data_retricd --output-dir outputs
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd

from excd.data import build_vocab, load_splits, parse_concepts
from excd.train import RunConfig, run


def _thin_train(train_df: pd.DataFrame, vocab, gamma: float, bottom_deciles: int, seed: int) -> pd.DataFrame:
    """Drop a fraction ``gamma`` of train rows whose concepts fall in the bottom ``bottom_deciles`` deciles."""
    if gamma <= 0:
        return train_df.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    rare_concepts = {
        orig for orig, idx in vocab.concept_map.items() if vocab.exposure_decile[idx] < bottom_deciles
    }
    touches_rare = np.array(
        [bool(rare_concepts.intersection(parse_concepts(v))) for v in train_df["cpt_seq"].values]
    )
    drop_mask = touches_rare & (rng.random(len(train_df)) < gamma)
    return train_df[~drop_mask].reset_index(drop=True)


def run_stress(args) -> None:
    train_df, valid_df, test_df = load_splits(args.dataset, args.data_root)
    base_vocab = build_vocab(train_df)

    stress_root = os.path.join(args.data_root, "_stress")
    os.makedirs(stress_root, exist_ok=True)
    summary = []

    for gamma in args.gammas:
        thinned = _thin_train(train_df, base_vocab, gamma, args.bottom_deciles, args.seed)
        ds_name = f"{args.dataset}_g{gamma:.2f}".replace(".", "")
        ds_dir = os.path.join(stress_root, ds_name)
        os.makedirs(ds_dir, exist_ok=True)
        thinned.to_csv(os.path.join(ds_dir, "train.csv"), index=False)
        valid_df.to_csv(os.path.join(ds_dir, "valid.csv"), index=False)
        test_df.to_csv(os.path.join(ds_dir, "test.csv"), index=False)
        print(f"\n=== gamma={gamma} | train rows {len(train_df)} -> {len(thinned)} ({ds_name}) ===", flush=True)

        for variant in args.variants:
            cfg = RunConfig(
                dataset=ds_name,
                data_root=stress_root,
                output_dir=os.path.join(args.output_dir, "mnar_stress"),
                model=args.model,
                variant=variant,
                epochs=args.epochs,
                device=args.device,
                seed=args.seed,
                tag=f"gamma{gamma:.2f}".replace(".", ""),
            )
            res = run(cfg)
            strat = res["test"]["exposure_stratified_doa"]
            summary.append(
                {
                    "gamma": gamma,
                    "variant": variant,
                    "auc": res["test"]["auc"],
                    "doa": res["test"]["doa"],
                    "tail_decile_doa": strat["tail_decile_doa"],
                    "head_decile_doa": strat["head_decile_doa"],
                    "head_minus_tail_gap": strat["head_minus_tail_gap"],
                }
            )

    out_dir = os.path.join(args.output_dir, "mnar_stress")
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, f"{args.dataset}_{args.model}_stress_summary.csv")
    cols = ["gamma", "variant", "auc", "doa", "tail_decile_doa", "head_decile_doa", "head_minus_tail_gap"]
    with open(summary_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for row in summary:
            f.write(",".join(f"{row[c]:.6f}" if isinstance(row[c], float) else str(row[c]) for c in cols) + "\n")
    print(f"\nMNAR-stress summary -> {summary_path}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)

    if args.cleanup:
        shutil.rmtree(stress_root, ignore_errors=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Ex-CD MNAR-stress harness")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--data-root", type=str, default="data_retricd")
    p.add_argument("--output-dir", type=str, default="outputs")
    p.add_argument("--model", type=str, default="ncdm", choices=["ncdm", "kancd"])
    p.add_argument("--variants", type=str, nargs="+", default=["vanilla", "ips"])
    p.add_argument("--gammas", type=float, nargs="+", default=[0.0, 0.3, 0.6])
    p.add_argument("--bottom-deciles", type=int, default=3, help="thin concepts in deciles 0..(b-1)")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cleanup", action="store_true", help="delete the temp _stress datasets afterwards")
    run_stress(p.parse_args())


if __name__ == "__main__":
    main()
