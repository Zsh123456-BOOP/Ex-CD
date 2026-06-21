"""CertCD real-data pipeline.

Train a CDM backbone -> calibrate -> build the per-cell identifiability certificate ->
evaluate selective prediction (risk-coverage / AURC / selective-AUC) for the certificate vs
the count, learned-confidence (MC-dropout), margin, ability and random baselines -> report
ECE split by certified vs abstained cells -> also report the DOA mastery artifact (CD identity).

The pre-registered KILL-SWITCH lives in the output `certificate_excess_aurc_vs`: if the
certificate does NOT achieve positive excess AURC over BOTH `count` and `mc_dropout`, the
identifiability machinery is not pulling its weight and the idea should be reconsidered.

Example:
    python -m certcd.run --dataset assist_09 --model ncdm --data-root data_retricd --output-dir outputs
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from certcd import calibrate as C
from certcd import scores as S
from certcd.backbone import discrimination_by_exercise, predict, train_backbone
from certcd.certificate import build_coverage, certificate_scores, per_prediction
from certcd.selective import evaluate_methods
from excd.data import CDDataset, build_vocab, load_splits, make_collate
from excd.metrics import binary_metrics, compute_doa, stratified_doa


@dataclass
class RunConfig:
    dataset: str = "assist_09"
    data_root: str = "data_retricd"
    output_dir: str = "outputs"
    model: str = "ncdm"            # ncdm | kancd
    epochs: int = 30
    lr: float = 1e-3
    batch_size: int = 256
    dropout: float = 0.2
    latent_dim: int = 32
    grad_clip: float = 5.0
    patience: int = 8
    n_min: int = 2                 # certificate: min informative items for a cell to be certified
    mc_passes: int = 20
    doa_max_responders: int = 80
    num_workers: int = 2
    seed: int = 42
    device: str = "auto"


def resolve_device(name: str):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def run(cfg: RunConfig) -> dict:
    device = resolve_device(cfg.device)
    train_df, valid_df, test_df = load_splits(cfg.dataset, cfg.data_root)
    vocab = build_vocab(train_df)
    collate = make_collate(vocab.num_concepts)

    train_ds = CDDataset(train_df, vocab, use_ips=False)
    valid_ds = CDDataset(valid_df, vocab, use_ips=False)
    test_ds = CDDataset(test_df, vocab, use_ips=False)

    print(
        f"[certcd:{cfg.dataset}/{cfg.model}] S={vocab.num_students} E={vocab.num_exercises} "
        f"K={vocab.num_concepts} train={len(train_ds)} test={len(test_ds)} device={device}",
        flush=True,
    )

    model, best_valid_auc, best_ep = train_backbone(
        vocab, train_ds, valid_ds,
        model_type=cfg.model, epochs=cfg.epochs, lr=cfg.lr, batch_size=cfg.batch_size,
        dropout=cfg.dropout, latent_dim=cfg.latent_dim, grad_clip=cfg.grad_clip,
        patience=cfg.patience, num_workers=cfg.num_workers, device=device, seed=cfg.seed,
    )

    valid_loader = DataLoader(valid_ds, batch_size=1024, shuffle=False, collate_fn=collate, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False, collate_fn=collate, num_workers=cfg.num_workers)
    valid_probs = predict(model, valid_loader, device)
    test_probs_raw = predict(model, test_loader, device)

    # ---- calibration ----
    T = C.fit_temperature(valid_probs, valid_ds.label)
    test_probs = C.apply_temperature(test_probs_raw, T)
    ece_raw = C.ece(test_probs_raw, test_ds.label)
    ece_cal = C.ece(test_probs, test_ds.label)
    test_metrics = binary_metrics(test_ds.label, test_probs)

    # ---- mastery artifact + DOA (CD identity) ----
    mastery = model.mastery_matrix(device)
    doa_pc, doa_agg, doa_pairs = compute_doa(
        mastery, test_ds.student, test_ds.exercise, test_ds.concept_lists, test_ds.label,
        num_concepts=vocab.num_concepts, max_responders_per_item=cfg.doa_max_responders, seed=cfg.seed,
    )
    strat = stratified_doa(doa_pc, vocab.exposure_decile, pair_counts=doa_pairs)

    # ---- certificate ----
    disc = discrimination_by_exercise(model, device)
    count, purity, dwp = build_coverage(
        train_ds.student, train_ds.exercise, train_ds.concept_lists,
        vocab.num_students, vocab.num_concepts, disc_by_exercise=disc,
    )
    cert_score_cell, cert_bin_cell, concept_ident = certificate_scores(count, purity, dwp, vocab, n_min=cfg.n_min)

    cert_pred = per_prediction(cert_score_cell, test_ds.student, test_ds.concept_lists, agg="mean")
    certified_pred = per_prediction(cert_bin_cell, test_ds.student, test_ds.concept_lists, agg="min")  # all concepts certified

    # ---- abstention scores (higher = keep) ----
    score_dict = {
        "certificate": cert_pred,
        "count": S.count_scores(count, test_ds.student, test_ds.concept_lists),
        "prob_margin": S.prob_margin_scores(test_probs),
        "mc_dropout": S.mc_dropout_scores(model, test_loader, device, passes=cfg.mc_passes),
        "ability": S.ability_scores(train_ds.student, train_ds.label, vocab.num_students, test_ds.student),
        "random": S.random_scores(len(test_ds.label), seed=cfg.seed),
    }
    selective = evaluate_methods(score_dict, np.asarray(test_ds.label), test_probs)

    # ---- ECE split by certificate ----
    cmask = certified_pred >= 0.5
    ece_certified = C.ece(test_probs[cmask], np.asarray(test_ds.label)[cmask]) if cmask.any() else float("nan")
    ece_abstained = C.ece(test_probs[~cmask], np.asarray(test_ds.label)[~cmask]) if (~cmask).any() else float("nan")

    excess = selective["certificate_excess_aurc_vs"]
    kill_switch_pass = bool(excess.get("count", -1) > 0 and excess.get("mc_dropout", -1) > 0)

    print(
        f"  TEST auc {test_metrics['auc']:.4f} acc {test_metrics['acc']:.4f} | DOA {doa_agg:.4f} "
        f"| T={T:.2f} ECE {ece_raw:.4f}->{ece_cal:.4f} (cert {ece_certified:.4f} / abst {ece_abstained:.4f})",
        flush=True,
    )
    print(
        f"  AURC certificate {selective['per_method']['certificate']['aurc']:.4f} "
        f"| excess vs count {excess.get('count', float('nan')):+.4f} "
        f"vs mc_dropout {excess.get('mc_dropout', float('nan')):+.4f} "
        f"vs margin {excess.get('prob_margin', float('nan')):+.4f} "
        f"| KILL-SWITCH {'PASS' if kill_switch_pass else 'FAIL'}",
        flush=True,
    )

    result = {
        "dataset": cfg.dataset,
        "model": cfg.model,
        "seed": cfg.seed,
        "best_valid_auc": best_valid_auc,
        "best_epoch": best_ep,
        "test": test_metrics,
        "doa": {"aggregate": doa_agg, "exposure_stratified": strat},
        "calibration": {
            "temperature": T,
            "ece_uncalibrated": ece_raw,
            "ece_calibrated": ece_cal,
            "ece_certified": ece_certified,
            "ece_abstained": ece_abstained,
        },
        "certificate": {
            "n_min": cfg.n_min,
            "concept_identifiable_fraction": float(concept_ident.mean()),
            "certified_cell_fraction": float(cert_bin_cell.mean()),
            "certified_prediction_fraction": float(cmask.mean()),
        },
        "selective": selective,
        "kill_switch_pass": kill_switch_pass,
        "config": asdict(cfg),
    }

    out_dir = os.path.join(cfg.output_dir, cfg.dataset, cfg.model)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"  saved -> {out_dir}/metrics.json", flush=True)
    return result


def _parser():
    p = argparse.ArgumentParser(description="CertCD real-data pipeline")
    p.add_argument("--config", type=str, default=None)
    for fld, typ in [
        ("dataset", str), ("data-root", str), ("output-dir", str), ("model", str),
        ("epochs", int), ("lr", float), ("batch-size", int), ("dropout", float),
        ("latent-dim", int), ("grad-clip", float), ("patience", int), ("n-min", int),
        ("mc-passes", int), ("doa-max-responders", int), ("num-workers", int),
        ("seed", int), ("device", str),
    ]:
        p.add_argument(f"--{fld}", type=typ)
    return p


def cfg_from_args(args) -> RunConfig:
    cfg = RunConfig()
    if args.config:
        import yaml

        with open(args.config) as f:
            for k, v in (yaml.safe_load(f) or {}).items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
    for fld in cfg.__dataclass_fields__:
        v = getattr(args, fld, None)
        if v is not None:
            setattr(cfg, fld, v)
    return cfg


def main():
    run(cfg_from_args(_parser().parse_args()))


if __name__ == "__main__":
    main()
