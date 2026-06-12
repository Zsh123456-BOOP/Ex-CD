from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from retricd.config import DEFAULT_DATASETS, RunConfig, set_seed
from retricd.data import REGIME_NAMES, load_prefix_datasets
from retricd.model import RetriCDModel


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _move(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def _safe_auc(labels, probs) -> float:
    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, probs))


def _metrics(labels: List[float], probs: List[float]) -> Dict[str, float]:
    y = np.asarray(labels, dtype=np.float32)
    p = np.asarray(probs, dtype=np.float32).clip(1e-6, 1.0 - 1e-6)
    pred = (p >= 0.5).astype(np.float32)
    return {
        "auc": _safe_auc(y.tolist(), p.tolist()),
        "acc": float(accuracy_score(y, pred)),
        "rmse": float(mean_squared_error(y, p) ** 0.5),
        "bce": float(log_loss(y, p, labels=[0, 1])),
        "n": int(len(y)),
    }


def evaluate(
    model: RetriCDModel,
    loader: DataLoader,
    device: torch.device,
    export_dir: Optional[Path] = None,
    export_support_limit: int = 0,
    case_limit: int = 0,
) -> Dict[str, Dict[str, float]]:
    model.eval()
    labels: List[float] = []
    probs: List[float] = []
    regimes: List[int] = []
    short_flags: List[bool] = []
    pred_writer = support_writer = None
    pred_file = support_file = cases_file = None
    exported_queries = 0
    cases_written = 0
    if export_dir is not None:
        export_dir.mkdir(parents=True, exist_ok=True)
        pred_file = (export_dir / "predictions.csv").open("w", newline="")
        support_file = (export_dir / "supports.csv").open("w", newline="")
        cases_file = (export_dir / "cases.jsonl").open("w")
        pred_writer = csv.DictWriter(pred_file, fieldnames=["row_id", "label", "prob", "regime", "hist_len", "query_exercise"])
        support_writer = csv.DictWriter(
            support_file,
            fieldnames=["row_id", "rank", "hist_position", "weight", "hist_exercise", "hist_correct", "hist_difficulty"],
        )
        pred_writer.writeheader()
        support_writer.writeheader()

    row_id = 0
    with torch.no_grad():
        for batch in loader:
            batch = _move(batch, device)
            out = model(batch)
            batch_labels = batch["label"].detach().cpu().numpy().astype(float)
            batch_probs = out["prob"].detach().cpu().numpy().astype(float)
            batch_regimes = batch["regime"].detach().cpu().numpy().astype(int)
            batch_short = batch["is_short_history"].detach().cpu().numpy().astype(bool)
            labels.extend(batch_labels.tolist())
            probs.extend(batch_probs.tolist())
            regimes.extend(batch_regimes.tolist())
            short_flags.extend(batch_short.tolist())

            if pred_writer is None:
                row_id += len(batch_labels)
                continue

            top_idx = out["topk_idx"].detach().cpu().numpy()
            attn = out["attn"].detach().cpu().numpy()
            hist_exer = batch["hist_exercise_id"].detach().cpu().numpy()
            hist_correct = batch["hist_correct"].detach().cpu().numpy()
            hist_diff = batch["hist_difficulty"].detach().cpu().numpy()
            hist_mask = batch["hist_mask"].detach().cpu().numpy().astype(bool)
            hist_len = batch["hist_len"].detach().cpu().numpy().astype(int)
            raw_query = batch["raw_query_exercise"].detach().cpu().numpy()
            for i in range(len(batch_labels)):
                regime_name = REGIME_NAMES.get(int(batch_regimes[i]), "unknown")
                pred_writer.writerow(
                    {
                        "row_id": row_id,
                        "label": float(batch_labels[i]),
                        "prob": float(batch_probs[i]),
                        "regime": regime_name,
                        "hist_len": int(hist_len[i]),
                        "query_exercise": int(raw_query[i]),
                    }
                )
                if exported_queries < export_support_limit:
                    valid_supports = []
                    for rank, pos in enumerate(top_idx[i]):
                        if pos < 0 or pos >= hist_mask.shape[1] or not hist_mask[i, pos]:
                            continue
                        weight = float(attn[i, pos])
                        if weight <= 0:
                            continue
                        support_writer.writerow(
                            {
                                "row_id": row_id,
                                "rank": rank,
                                "hist_position": int(pos),
                                "weight": weight,
                                "hist_exercise": int(hist_exer[i, pos]),
                                "hist_correct": float(hist_correct[i, pos]),
                                "hist_difficulty": float(hist_diff[i, pos]),
                            }
                        )
                        valid_supports.append(
                            {
                                "rank": rank,
                                "hist_position": int(pos),
                                "weight": weight,
                                "hist_exercise": int(hist_exer[i, pos]),
                                "hist_correct": float(hist_correct[i, pos]),
                            }
                        )
                    if cases_written < case_limit and valid_supports:
                        cases_file.write(
                            json.dumps(
                                {
                                    "row_id": row_id,
                                    "label": float(batch_labels[i]),
                                    "prob": float(batch_probs[i]),
                                    "regime": regime_name,
                                    "hist_len": int(hist_len[i]),
                                    "query_exercise": int(raw_query[i]),
                                    "supports": valid_supports,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        cases_written += 1
                    exported_queries += 1
                row_id += 1

    for handle in (pred_file, support_file, cases_file):
        if handle is not None:
            handle.close()

    result = {"overall": _metrics(labels, probs)}
    arr_regimes = np.asarray(regimes)
    arr_short = np.asarray(short_flags)
    for code, name in REGIME_NAMES.items():
        mask = arr_regimes == code
        if mask.any():
            result[name] = _metrics(np.asarray(labels)[mask].tolist(), np.asarray(probs)[mask].tolist())
    if arr_short.any():
        result["short_history"] = _metrics(np.asarray(labels)[arr_short].tolist(), np.asarray(probs)[arr_short].tolist())
    return result


def train_one_dataset(cfg: RunConfig) -> Dict[str, Dict[str, float]]:
    set_seed(cfg.seed)
    device = _device(cfg.device)
    run_dir = Path(cfg.output_dir) / cfg.dataset / f"seed{cfg.seed}_{cfg.variant}"
    run_dir.mkdir(parents=True, exist_ok=True)
    datasets, encoded = load_prefix_datasets(cfg.data_root, cfg.dataset, cfg.max_history_len, cfg.max_concepts)
    loaders = {
        split: DataLoader(ds, batch_size=cfg.batch_size, shuffle=(split == "train"), num_workers=cfg.num_workers)
        for split, ds in datasets.items()
    }
    model = RetriCDModel(
        encoded.num_exercises,
        encoded.num_concepts,
        embed_dim=cfg.embed_dim,
        topk=cfg.topk,
        temperature=cfg.temperature,
        dropout=cfg.dropout,
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_auc = -1.0
    bad_epochs = 0
    history_rows = []
    best_path = run_dir / "best_model.pt"

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        iterator = tqdm(loaders["train"], desc=f"{cfg.dataset} epoch {epoch}", leave=False)
        for batch in iterator:
            batch = _move(batch, device)
            optim.zero_grad(set_to_none=True)
            out = model(batch)
            bce = F.binary_cross_entropy_with_logits(out["logit"], batch["label"].float())
            loss = bce
            if cfg.fidelity_margin_weight > 0:
                rand_out = model(batch, support_mode="random")
                rand_bce = F.binary_cross_entropy_with_logits(rand_out["logit"], batch["label"].float())
                fid = torch.relu(torch.tensor(cfg.fidelity_margin, device=device) + bce - rand_bce.detach())
                loss = loss + cfg.fidelity_margin_weight * fid
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            losses.append(float(loss.detach().cpu()))

        valid = evaluate(model, loaders["valid"], device)
        valid_auc = valid["overall"]["auc"]
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"valid_{k}": v for k, v in valid["overall"].items()}}
        history_rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        score = valid_auc if not np.isnan(valid_auc) else -1.0
        if score > best_auc:
            best_auc = score
            bad_epochs = 0
            torch.save({"model": model.state_dict(), "config": cfg.__dict__}, best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                break

    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
    test_metrics = evaluate(
        model,
        loaders["test"],
        device,
        export_dir=run_dir / "exports",
        export_support_limit=cfg.export_support_limit,
        case_limit=cfg.case_limit,
    )
    (run_dir / "metrics.json").write_text(json.dumps(test_metrics, indent=2, ensure_ascii=False))
    with (run_dir / "history.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(history_rows[0].keys()) if history_rows else ["epoch"])
        writer.writeheader()
        writer.writerows(history_rows)
    print(json.dumps({"dataset": cfg.dataset, "test": test_metrics}, ensure_ascii=False), flush=True)
    return test_metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RetriCD full model.")
    parser.add_argument("--dataset", choices=DEFAULT_DATASETS + ("all",), default="assist_09")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="outputs/retricd_full")
    parser.add_argument("--variant", default="full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-history-len", type=int, default=128)
    parser.add_argument("--max-concepts", type=int, default=8)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--fidelity-margin-weight", type=float, default=0.2)
    parser.add_argument("--fidelity-margin", type=float, default=0.02)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--export-support-limit", type=int, default=50000)
    parser.add_argument("--case-limit", type=int, default=200)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    datasets = DEFAULT_DATASETS if args.dataset == "all" else (args.dataset,)
    all_metrics = {}
    for dataset in datasets:
        cfg = RunConfig(dataset=dataset, **{k.replace("-", "_"): v for k, v in vars(args).items() if k != "dataset"})
        all_metrics[dataset] = train_one_dataset(cfg)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "summary.json").write_text(json.dumps(all_metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

