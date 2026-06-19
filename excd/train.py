"""Ex-CD training runner.

Trains a cognitive-diagnosis backbone (NCDM / KaNCD) under one debiasing variant
(vanilla | ips | dr), then evaluates AUC/ACC/RMSE/BCE plus DOA and the Exposure-Stratified
DOA cliff on the test split. Exposes ``run(cfg)`` for reuse (e.g. the MNAR-stress harness)
and a ``main()`` CLI. All logging goes to stdout so server logs capture full progress.

Example:
    python -m excd.train --dataset assist_09 --model ncdm --variant ips \
        --data-root data_retricd --output-dir outputs
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from excd.data import CDDataset, build_vocab, load_splits, make_collate
from excd.losses import dr_loss, imputation_bce, ips_bce, vanilla_bce
from excd.metrics import binary_metrics, compute_doa, stratified_doa
from excd.model import CDM


@dataclass
class RunConfig:
    dataset: str = "assist_09"
    data_root: str = "data_retricd"
    output_dir: str = "outputs"
    model: str = "ncdm"               # ncdm | kancd
    variant: str = "vanilla"          # vanilla | ips | dr
    epochs: int = 15
    batch_size: int = 256
    lr: float = 2e-3
    weight_decay: float = 0.0
    latent_dim: int = 32              # kancd only
    dropout: float = 0.5
    hidden: tuple = (256, 128)
    propensity_floor: float = 0.05
    doa_max_responders: int = 80
    doa_min_pairs: int = 5
    num_workers: int = 2
    patience: int = 5                 # early stopping on valid AUC
    seed: int = 42
    device: str = "auto"
    save_mastery: bool = False
    tag: str = ""                     # optional run-name suffix


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _eval_split(model, loader, device) -> np.ndarray:
    model.eval()
    probs = []
    with torch.no_grad():
        for batch in loader:
            prob, _ = model(
                batch["student"].to(device),
                batch["exercise"].to(device),
                batch["concept_mask"].to(device),
            )
            probs.append(prob.cpu().numpy())
    return np.concatenate(probs, axis=0)


def run(cfg: RunConfig) -> Dict:
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    use_ips = cfg.variant in ("ips", "dr")

    train_df, valid_df, test_df = load_splits(cfg.dataset, cfg.data_root)
    vocab = build_vocab(train_df, propensity_floor=cfg.propensity_floor)
    collate = make_collate(vocab.num_concepts)

    train_ds = CDDataset(train_df, vocab, use_ips=use_ips)
    valid_ds = CDDataset(valid_df, vocab, use_ips=False)
    test_ds = CDDataset(test_df, vocab, use_ips=False)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate, num_workers=cfg.num_workers
    )
    valid_loader = DataLoader(valid_ds, batch_size=1024, shuffle=False, collate_fn=collate, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False, collate_fn=collate, num_workers=cfg.num_workers)

    print(
        f"[{cfg.dataset}/{cfg.model}/{cfg.variant}] students={vocab.num_students} "
        f"exercises={vocab.num_exercises} concepts={vocab.num_concepts} "
        f"train={len(train_ds)} valid={len(valid_ds)} test={len(test_ds)} device={device}",
        flush=True,
    )

    model = CDM(
        num_students=vocab.num_students,
        num_exercises=vocab.num_exercises,
        num_concepts=vocab.num_concepts,
        model_type=cfg.model,
        latent_dim=cfg.latent_dim,
        hidden=tuple(cfg.hidden),
        dropout=cfg.dropout,
        with_imputation=(cfg.variant == "dr"),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = []
    best_auc = -1.0
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epochs_without_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        t0 = time.time()
        for batch in train_loader:
            student = batch["student"].to(device)
            exercise = batch["exercise"].to(device)
            mask = batch["concept_mask"].to(device)
            label = batch["label"].to(device)
            weight = batch["weight"].to(device)

            prob, x = model(student, exercise, mask)
            if cfg.variant == "vanilla":
                loss = vanilla_bce(prob, label)
            elif cfg.variant == "ips":
                loss = ips_bce(prob, label, weight)
            else:  # dr
                imp_prob = model.impute(x.detach())
                loss = dr_loss(prob, label, weight, imp_prob) + imputation_bce(imp_prob, label, weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        valid_probs = _eval_split(model, valid_loader, device)
        valid_metrics = binary_metrics(valid_ds.label, valid_probs)
        train_loss = float(np.mean(losses)) if losses else float("nan")
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_auc": valid_metrics["auc"],
                "valid_acc": valid_metrics["acc"],
                "valid_rmse": valid_metrics["rmse"],
            }
        )
        print(
            f"  epoch {epoch:02d} | loss {train_loss:.4f} | valid_auc {valid_metrics['auc']:.4f} "
            f"acc {valid_metrics['acc']:.4f} rmse {valid_metrics['rmse']:.4f} | {time.time() - t0:.1f}s",
            flush=True,
        )

        if valid_metrics["auc"] > best_auc:
            best_auc = valid_metrics["auc"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= cfg.patience:
                print(f"  early stop at epoch {epoch} (best epoch {best_epoch}, auc {best_auc:.4f})", flush=True)
                break

    model.load_state_dict(best_state)

    # ---- final test evaluation ----
    t_eval = time.time()
    test_probs = _eval_split(model, test_loader, device)
    test_metrics = binary_metrics(test_ds.label, test_probs)

    mastery = model.mastery_matrix(device)
    doa_per_concept, doa_agg = compute_doa(
        mastery,
        test_ds.student,
        test_ds.exercise,
        test_ds.concept_lists,
        test_ds.label,
        num_concepts=vocab.num_concepts,
        max_responders_per_item=cfg.doa_max_responders,
        min_pairs=cfg.doa_min_pairs,
        seed=cfg.seed,
    )
    strat = stratified_doa(doa_per_concept, vocab.exposure_decile)
    eval_seconds = time.time() - t_eval

    test_metrics["doa"] = doa_agg
    test_metrics["exposure_stratified_doa"] = strat

    print(
        f"  TEST auc {test_metrics['auc']:.4f} acc {test_metrics['acc']:.4f} rmse {test_metrics['rmse']:.4f} "
        f"| DOA {doa_agg:.4f} | bottom(d0-2) {strat['bottom_group_doa']:.4f} top(d7-9) {strat['top_group_doa']:.4f} "
        f"group_gap {strat['group_gap']:.4f} | measurable_deciles {strat['n_measurable_deciles']}/10 "
        f"| doa_eval {eval_seconds:.1f}s",
        flush=True,
    )

    # ---- persist ----
    run_name = f"{cfg.model}_{cfg.variant}_seed{cfg.seed}" + (f"_{cfg.tag}" if cfg.tag else "")
    out_dir = os.path.join(cfg.output_dir, cfg.dataset, run_name)
    os.makedirs(out_dir, exist_ok=True)

    result = {
        "dataset": cfg.dataset,
        "model": cfg.model,
        "variant": cfg.variant,
        "seed": cfg.seed,
        "best_epoch": best_epoch,
        "best_valid_auc": best_auc,
        "epochs_run": len(history),
        "test": test_metrics,
        "doa_eval_seconds": eval_seconds,
        "config": asdict(cfg),
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(out_dir, "history.csv"), "w") as f:
        f.write("epoch,train_loss,valid_auc,valid_acc,valid_rmse\n")
        for h in history:
            f.write(f"{h['epoch']},{h['train_loss']:.6f},{h['valid_auc']:.6f},{h['valid_acc']:.6f},{h['valid_rmse']:.6f}\n")
    # per-concept DOA + exposure (so the cliff can be re-plotted without re-running)
    with open(os.path.join(out_dir, "concept_doa.csv"), "w") as f:
        f.write("concept_index,exposure_count,exposure_decile,doa\n")
        for k in range(vocab.num_concepts):
            d = doa_per_concept[k]
            f.write(
                f"{k},{int(vocab.concept_exposure[k])},{int(vocab.exposure_decile[k])},"
                f"{'' if np.isnan(d) else f'{d:.6f}'}\n"
            )
    if cfg.save_mastery:
        np.save(os.path.join(out_dir, "mastery.npy"), mastery)

    print(f"  saved -> {out_dir}", flush=True)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ex-CD training runner")
    p.add_argument("--config", type=str, default=None, help="optional YAML overriding defaults")
    p.add_argument("--dataset", type=str)
    p.add_argument("--data-root", type=str)
    p.add_argument("--output-dir", type=str)
    p.add_argument("--model", type=str, choices=["ncdm", "kancd"])
    p.add_argument("--variant", type=str, choices=["vanilla", "ips", "dr"])
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--weight-decay", type=float)
    p.add_argument("--latent-dim", type=int)
    p.add_argument("--dropout", type=float)
    p.add_argument("--propensity-floor", type=float)
    p.add_argument("--doa-max-responders", type=int)
    p.add_argument("--doa-min-pairs", type=int)
    p.add_argument("--num-workers", type=int)
    p.add_argument("--patience", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--device", type=str)
    p.add_argument("--tag", type=str)
    p.add_argument("--save-mastery", action="store_true")
    return p


def cfg_from_args(args) -> RunConfig:
    cfg = RunConfig()
    if args.config:
        import yaml

        with open(args.config) as f:
            raw = yaml.safe_load(f) or {}
        for k, v in raw.items():
            if hasattr(cfg, k):
                setattr(cfg, k, tuple(v) if k == "hidden" else v)
    for field_name in cfg.__dataclass_fields__:
        arg_val = getattr(args, field_name, None)
        if arg_val is not None and not (field_name == "save_mastery" and arg_val is False):
            setattr(cfg, field_name, arg_val)
    return cfg


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = cfg_from_args(args)
    run(cfg)


if __name__ == "__main__":
    main()
