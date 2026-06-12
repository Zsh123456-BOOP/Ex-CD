from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from retricd.collate import move_to_device
from retricd.config import DEFAULT_DATASETS, RunConfig, set_seed, term_flags_for_variant, train_support_mode_for_variant
from retricd.datasets import load_prefix_datasets
from retricd.losses import bce_loss, fidelity_margin_loss
from retricd.metrics import evaluate_fidelity_suite, evaluate_model
from retricd.model import RetriCDModel


def device_from_name(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _simple_yaml(path: Path) -> dict:
    data: dict = {}
    stack = [data]
    indents = [0]
    for raw in path.read_text().splitlines():
        if not raw.strip() or raw.strip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        key, _, value = raw.strip().partition(":")
        while indent < indents[-1]:
            stack.pop()
            indents.pop()
        if not value.strip():
            node = {}
            stack[-1][key] = node
            stack.append(node)
            indents.append(indent + 2)
            continue
        text = value.strip()
        if text in ("true", "false"):
            parsed = text == "true"
        elif text.startswith("["):
            parsed = json.loads(text)
        else:
            try:
                parsed = int(text)
            except ValueError:
                try:
                    parsed = float(text)
                except ValueError:
                    parsed = text.strip("\"'")
        stack[-1][key] = parsed
    return data


def load_config(path: str | None) -> dict:
    if not path:
        return {}
    p = Path(path)
    if p.suffix == ".json":
        return json.loads(p.read_text())
    try:
        import yaml  # type: ignore

        return yaml.safe_load(p.read_text()) or {}
    except Exception:
        return _simple_yaml(p)


def train_one_dataset(cfg: RunConfig) -> Dict[str, Dict]:
    set_seed(cfg.seed)
    device = device_from_name(cfg.device)
    run_dir = Path(cfg.output_dir) / cfg.dataset / f"seed{cfg.seed}_{cfg.variant}"
    run_dir.mkdir(parents=True, exist_ok=True)
    datasets, encoded = load_prefix_datasets(cfg.data_root, cfg.dataset, cfg.max_history_len, cfg.max_concepts)
    loaders = {
        split: DataLoader(ds, batch_size=cfg.batch_size, shuffle=(split == "train"), num_workers=cfg.num_workers)
        for split, ds in datasets.items()
    }
    term_flags = cfg.retriever_terms or term_flags_for_variant(cfg.variant)
    model = RetriCDModel(
        encoded.num_exercises,
        encoded.num_concepts,
        embed_dim=cfg.embed_dim,
        topk=cfg.topk,
        temperature=cfg.temperature,
        dropout=cfg.dropout,
        term_flags=term_flags,
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    train_support_mode = train_support_mode_for_variant(cfg.variant)
    best_auc = -1.0
    bad_epochs = 0
    history_rows = []
    best_path = run_dir / "best_model.pt"
    (run_dir / "run_config.json").write_text(json.dumps({**cfg.__dict__, "retriever_terms": term_flags}, indent=2))

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        iterator = tqdm(loaders["train"], desc=f"{cfg.dataset}:{cfg.variant} epoch {epoch}", leave=False)
        for batch in iterator:
            batch = move_to_device(batch, device)
            optim.zero_grad(set_to_none=True)
            out = model(batch, support_mode=train_support_mode)
            bce = bce_loss(out["logit"], batch["label"])
            loss = bce
            if cfg.fidelity_margin_weight > 0 and cfg.variant == "full":
                rand_out = model(batch, support_mode="random")
                rand_bce = bce_loss(rand_out["logit"], batch["label"])
                loss = loss + cfg.fidelity_margin_weight * fidelity_margin_loss(bce, rand_bce, cfg.fidelity_margin)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            losses.append(float(loss.detach().cpu()))

        valid = evaluate_model(model, loaders["valid"], device, support_mode=train_support_mode)
        valid_auc = valid["overall"]["auc"]
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"valid_{k}": v for k, v in valid["overall"].items()}}
        history_rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        score = valid_auc if not np.isnan(valid_auc) else -1.0
        if score > best_auc:
            best_auc = score
            bad_epochs = 0
            torch.save({"model": model.state_dict(), "config": cfg.__dict__, "term_flags": term_flags}, best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                break

    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model"])

    main_metrics = evaluate_model(
        model,
        loaders["test"],
        device,
        support_mode=train_support_mode,
        export_dir=run_dir / "exports" if cfg.variant == "full" else None,
        export_support_limit=cfg.export_support_limit,
        case_limit=cfg.case_limit,
    )
    metrics = {"main": main_metrics}
    if cfg.variant == "full":
        metrics["fidelity"] = evaluate_fidelity_suite(model, loaders["test"], device)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    with (run_dir / "history.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(history_rows[0].keys()) if history_rows else ["epoch"])
        writer.writeheader()
        writer.writerows(history_rows)
    print(json.dumps({"dataset": cfg.dataset, "variant": cfg.variant, "test": metrics}, ensure_ascii=False), flush=True)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RetriCD experiments.")
    parser.add_argument("--config")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--variant", default=None)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-history-len", type=int)
    parser.add_argument("--max-concepts", type=int)
    parser.add_argument("--topk", type=int)
    parser.add_argument("--embed-dim", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--fidelity-margin-weight", type=float)
    parser.add_argument("--fidelity-margin", type=float)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--export-support-limit", type=int)
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--device")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace, dataset: str) -> RunConfig:
    raw = load_config(args.config)
    flat = {
        "dataset": raw.get("dataset", dataset),
        "data_root": raw.get("data_root", "data_retricd"),
        "output_dir": raw.get("output_dir", "outputs/retricd_full"),
        "variant": raw.get("variant", raw.get("model_variant", "full")),
        "seed": raw.get("seed", 42),
        "max_history_len": raw.get("max_history_len", 128),
        "max_concepts": raw.get("max_concepts", 8),
        "topk": raw.get("topk", 16),
        "embed_dim": raw.get("embed_dim", 64),
        "batch_size": raw.get("batch_size", 256),
        "epochs": raw.get("epochs", 30),
        "patience": raw.get("patience", 5),
        "lr": raw.get("learning_rate", raw.get("lr", 1e-3)),
        "weight_decay": raw.get("weight_decay", 1e-5),
        "dropout": raw.get("dropout", 0.15),
        "temperature": raw.get("temperature", 0.5),
        "fidelity_margin_weight": raw.get("loss", {}).get("fidelity_margin_weight", raw.get("fidelity_margin_weight", 0.2)),
        "fidelity_margin": raw.get("loss", {}).get("fidelity_margin", raw.get("fidelity_margin", 0.02)),
        "num_workers": raw.get("num_workers", 0),
        "export_support_limit": raw.get("export_support_limit", 50000),
        "case_limit": raw.get("case_limit", 200),
        "device": raw.get("device", "auto"),
        "retriever_terms": raw.get("retriever", {}),
    }
    for key, value in vars(args).items():
        if key == "config" or value is None:
            continue
        flat[key.replace("-", "_")] = value
    flat["dataset"] = dataset
    return RunConfig(**flat)


def main() -> None:
    args = parse_args()
    cfg_raw = load_config(args.config)
    requested = args.dataset or cfg_raw.get("dataset") or "assist_09"
    datasets = DEFAULT_DATASETS if requested == "all" else (requested,)
    all_metrics = {}
    for dataset in datasets:
        cfg = config_from_args(args, dataset)
        all_metrics[dataset] = train_one_dataset(cfg)
    out = Path(args.output_dir or cfg_raw.get("output_dir", "outputs/retricd_full"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(all_metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
