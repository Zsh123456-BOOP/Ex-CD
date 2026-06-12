from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, roc_auc_score

from retricd.collate import move_to_device
from retricd.datasets import REGIME_NAMES
from retricd.explain import collect_export_rows, export_cases, export_predictions, export_supports


def safe_auc(labels, probs) -> float:
    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, probs))


def binary_metrics(labels: List[float], probs: List[float]) -> Dict[str, float]:
    y = np.asarray(labels, dtype=np.float32)
    p = np.asarray(probs, dtype=np.float32).clip(1e-6, 1.0 - 1e-6)
    pred = (p >= 0.5).astype(np.float32)
    return {
        "auc": safe_auc(y.tolist(), p.tolist()),
        "acc": float(accuracy_score(y, pred)),
        "rmse": float(mean_squared_error(y, p) ** 0.5),
        "bce": float(log_loss(y, p, labels=[0, 1])),
        "n": int(len(y)),
    }


def evaluate_model(
    model,
    loader,
    device: torch.device,
    *,
    support_mode: str = "retrieval",
    export_dir: Optional[Path] = None,
    export_support_limit: int = 0,
    case_limit: int = 0,
) -> Dict[str, Dict[str, float]]:
    model.eval()
    labels: List[float] = []
    probs: List[float] = []
    regimes: List[int] = []
    short_flags: List[bool] = []
    candidate_counts: List[int] = []
    pred_rows = []
    support_rows = []
    case_rows = []
    exported = 0
    start = time.perf_counter()
    encoded = loader.dataset.encoded
    row_id = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, device)
            out = model(batch, support_mode=support_mode)
            batch_labels = batch["label"].detach().cpu().numpy().astype(float)
            batch_probs = out["prob"].detach().cpu().numpy().astype(float)
            batch_regimes = batch["regime"].detach().cpu().numpy().astype(int)
            batch_short = batch["is_short_history"].detach().cpu().numpy().astype(bool)
            labels.extend(batch_labels.tolist())
            probs.extend(batch_probs.tolist())
            regimes.extend(batch_regimes.tolist())
            short_flags.extend(batch_short.tolist())
            candidate_counts.extend(batch["hist_mask"].detach().cpu().numpy().sum(axis=1).astype(int).tolist())
            if export_dir is not None and exported < export_support_limit and support_mode == "retrieval":
                p_rows, s_rows, c_rows = collect_export_rows(
                    batch,
                    out,
                    encoded,
                    row_id,
                    max(0, case_limit - len(case_rows)),
                )
                room = max(0, export_support_limit - exported)
                pred_rows.extend(p_rows[:room])
                support_rows.extend(s_rows[: room * max(1, out["topk_idx"].shape[1])])
                case_rows.extend(c_rows[: max(0, case_limit - len(case_rows))])
                exported += len(p_rows[:room])
            row_id += len(batch_labels)
    elapsed = time.perf_counter() - start
    result = {"overall": binary_metrics(labels, probs)}
    arr_regimes = np.asarray(regimes)
    arr_short = np.asarray(short_flags)
    for code, name in REGIME_NAMES.items():
        mask = arr_regimes == code
        if mask.any():
            result[name] = binary_metrics(np.asarray(labels)[mask].tolist(), np.asarray(probs)[mask].tolist())
    if arr_short.any():
        result["short_history"] = binary_metrics(np.asarray(labels)[arr_short].tolist(), np.asarray(probs)[arr_short].tolist())
    result["runtime"] = {
        "queries": int(len(labels)),
        "seconds": float(elapsed),
        "queries_per_second": float(len(labels) / max(elapsed, 1e-9)),
        "seconds_per_10k_queries": float(elapsed * 10000.0 / max(1, len(labels))),
        "avg_memory_candidates": float(np.mean(candidate_counts)) if candidate_counts else 0.0,
    }
    if export_dir is not None and support_mode == "retrieval":
        export_predictions(pred_rows, export_dir)
        export_supports(support_rows, export_dir)
        export_cases(case_rows, export_dir)
    return result


def evaluate_fidelity_suite(model, loader, device: torch.device) -> Dict[str, Dict[str, Dict[str, float]]]:
    suite = {
        "retrieval": evaluate_model(model, loader, device, support_mode="retrieval"),
        "support_deletion": evaluate_model(model, loader, device, support_mode="zero"),
        "support_corruption": evaluate_model(model, loader, device, support_mode="random"),
        "shuffle_student": evaluate_model(model, loader, device, support_mode="shuffle_student"),
    }
    base = suite["retrieval"]["overall"]
    for name in ("support_deletion", "support_corruption", "shuffle_student"):
        cur = suite[name]["overall"]
        suite[name]["delta_vs_retrieval"] = {
            "auc_drop": float(base["auc"] - cur["auc"]) if not np.isnan(base["auc"]) and not np.isnan(cur["auc"]) else float("nan"),
            "bce_increase": float(cur["bce"] - base["bce"]),
            "rmse_increase": float(cur["rmse"] - base["rmse"]),
        }
    return suite

