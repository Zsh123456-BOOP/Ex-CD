"""CertCD synthetic validation — the primary novelty evidence.

Generate synthetic data with known true mastery -> train the CDM -> for every (student,concept)
cell define `estimate_correct` = (mastery_hat>0.5) matches true mastery -> test whether the
certificate (coverage geometry + Q, never the truth) predicts estimate_correct, and whether it
does so BETTER than the trivial coverage-count baseline.

Headline numbers (per config):
  - cert_auroc / count_auroc : AUROC predicting estimate_correct (cells with >=1 attempt)
  - acc_certified / acc_abstained : estimation accuracy on certified vs abstained cells
    (a large gap = the certificate genuinely separates trustworthy from untrustworthy mastery)

Example:
    python -m certcd.run_synthetic --output-dir outputs/synth --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

from certcd.backbone import discrimination_by_exercise, train_backbone
from certcd.certificate import build_coverage, certificate_scores
from certcd.run import resolve_device
from excd.data import CDDataset, build_vocab


def _auroc(y, s):
    y = np.asarray(y)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, s))


def evaluate_config(gen, device, *, model="ncdm", epochs=30, seed=42, n_min=2) -> dict:
    vocab = build_vocab(gen["train_df"])
    train_ds = CDDataset(gen["train_df"], vocab, use_ips=False)
    valid_ds = CDDataset(gen["valid_df"], vocab, use_ips=False)
    model_obj, best_auc, _ = train_backbone(
        vocab, train_ds, valid_ds, model_type=model, epochs=epochs, device=device, seed=seed,
    )
    mastery_hat = model_obj.mastery_matrix(device)  # [S, K] aligned with M_true (identity ids)
    M_true = gen["M_true"]
    S = min(mastery_hat.shape[0], M_true.shape[0])
    K = min(mastery_hat.shape[1], M_true.shape[1])
    mastery_hat = mastery_hat[:S, :K]
    M_true = M_true[:S, :K]

    estimate_correct = ((mastery_hat > 0.5).astype(np.int8) == M_true).astype(np.int8)

    disc = discrimination_by_exercise(model_obj, device)
    count, purity, dwp = build_coverage(
        train_ds.student, train_ds.exercise, train_ds.concept_lists,
        vocab.num_students, vocab.num_concepts, disc_by_exercise=disc,
    )
    count, dwp = count[:S, :K], (dwp[:S, :K] if dwp is not None else None)
    cert_score, cert_bin, _ = certificate_scores(count, purity[:S, :K], dwp, vocab, n_min=n_min)
    cert_score, cert_bin = cert_score[:S, :K], cert_bin[:S, :K]

    attempted = count.reshape(-1) >= 1  # cells the student actually attempted
    yc = estimate_correct.reshape(-1)
    cs = cert_score.reshape(-1)
    ct = count.reshape(-1)
    cb = cert_bin.reshape(-1)

    res = {
        "best_valid_auc": best_auc,
        "n_cells": int(yc.size),
        "attempted_fraction": float(attempted.mean()),
        "estimate_correct_overall": float(yc.mean()),
        "cert_auroc_attempted": _auroc(yc[attempted], cs[attempted]),
        "count_auroc_attempted": _auroc(yc[attempted], ct[attempted]),
        "cert_auroc_all": _auroc(yc, cs),
        "acc_certified": float(yc[cb >= 0.5].mean()) if (cb >= 0.5).any() else float("nan"),
        "acc_abstained": float(yc[cb < 0.5].mean()) if (cb < 0.5).any() else float("nan"),
        "certified_cell_fraction": float((cb >= 0.5).mean()),
    }
    res["cert_beats_count"] = bool(
        not np.isnan(res["cert_auroc_attempted"]) and not np.isnan(res["count_auroc_attempted"])
        and res["cert_auroc_attempted"] > res["count_auroc_attempted"]
    )
    return res


def main():
    p = argparse.ArgumentParser(description="CertCD synthetic validation")
    p.add_argument("--output-dir", type=str, default="outputs/synth")
    p.add_argument("--model", type=str, default="ncdm")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-min", type=int, default=2)
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()
    device = resolve_device(args.device)

    # sweep Q-completeness (pure_frac) x coverage (items_per_student): the certificate should
    # track estimation correctness across regimes; both baselines should be weaker.
    configs = [
        {"pure_frac": 0.6, "entangled_frac": 0.3, "items_per_student": 60},
        {"pure_frac": 0.3, "entangled_frac": 0.4, "items_per_student": 60},
        {"pure_frac": 0.6, "entangled_frac": 0.3, "items_per_student": 30},
        {"pure_frac": 0.3, "entangled_frac": 0.4, "items_per_student": 30},
    ]
    from certcd.synthetic import generate

    out = []
    for i, c in enumerate(configs):
        print(f"\n=== synthetic config {i+1}/{len(configs)}: {c} ===", flush=True)
        gen = generate(seed=args.seed + i, **c)
        r = evaluate_config(gen, device, model=args.model, epochs=args.epochs, seed=args.seed, n_min=args.n_min)
        r["config"] = c
        out.append(r)
        print(
            f"  cert_auroc {r['cert_auroc_attempted']:.3f} vs count_auroc {r['count_auroc_attempted']:.3f} "
            f"| acc certified {r['acc_certified']:.3f} / abstained {r['acc_abstained']:.3f} "
            f"| beats_count={r['cert_beats_count']}",
            flush=True,
        )

    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, f"synthetic_{args.model}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {path}", flush=True)


if __name__ == "__main__":
    main()
