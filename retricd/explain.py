from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch

from retricd.datasets import EncodedData, REGIME_NAMES


def _json_value(value):
    if hasattr(value, "item"):
        return value.item()
    return value


def write_table(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
        return path
    except Exception as exc:
        fallback = path.with_suffix(".csv")
        df.to_csv(fallback, index=False)
        path.with_suffix(path.suffix + ".unavailable.txt").write_text(
            f"Parquet export unavailable; wrote {fallback.name}. Error: {exc}\n"
        )
        return fallback


def export_predictions(rows: List[dict], export_dir: Path) -> None:
    write_table(pd.DataFrame(rows), export_dir / "predictions.parquet")


def export_supports(rows: List[dict], export_dir: Path) -> None:
    write_table(pd.DataFrame(rows), export_dir / "supports.parquet")


def export_cases(rows: List[dict], export_dir: Path) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    with (export_dir / "cases.jsonl").open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def collect_export_rows(
    batch: Dict[str, torch.Tensor],
    out: Dict[str, torch.Tensor],
    encoded: EncodedData,
    start_row_id: int,
    case_limit_left: int,
) -> tuple[list, list, list]:
    labels = batch["label"].detach().cpu().numpy()
    probs = out["prob"].detach().cpu().numpy()
    regimes = batch["regime"].detach().cpu().numpy()
    hist_len = batch["hist_len"].detach().cpu().numpy()
    row_indices = batch["row_index"].detach().cpu().numpy()
    hist_row_indices = batch["hist_row_index"].detach().cpu().numpy()
    hist_mask = batch["hist_mask"].detach().cpu().numpy().astype(bool)
    hist_distance = batch["hist_distance"].detach().cpu().numpy()
    hist_difficulty = batch["hist_difficulty"].detach().cpu().numpy()
    top_idx = out["topk_idx"].detach().cpu().numpy()
    attn = out["attn"].detach().cpu().numpy()
    score_terms = {k: v.detach().cpu().numpy() for k, v in out.get("score_terms", {}).items()}

    pred_rows = []
    support_rows = []
    case_rows = []
    for i, row_index in enumerate(row_indices):
        row_id = start_row_id + i
        regime = REGIME_NAMES.get(int(regimes[i]), "unknown")
        pred = {
            "row_id": row_id,
            "encoded_row_index": int(row_index),
            "student_id": _json_value(encoded.raw_students[row_index]),
            "query_exercise": _json_value(encoded.raw_exercises[row_index]),
            "query_concepts": str(encoded.raw_concept_text[row_index]),
            "query_difficulty": float(encoded.difficulties[row_index]),
            "label": float(labels[i]),
            "prob": float(probs[i]),
            "regime": regime,
            "hist_len": int(hist_len[i]),
        }
        pred_rows.append(pred)
        supports = []
        for rank, pos in enumerate(top_idx[i]):
            if pos < 0 or pos >= hist_mask.shape[1] or not hist_mask[i, pos]:
                continue
            hist_row = int(hist_row_indices[i, pos])
            if hist_row < 0:
                continue
            support = {
                "row_id": row_id,
                "rank": int(rank),
                "hist_position": int(pos),
                "support_weight": float(attn[i, pos]),
                "score_total": float(score_terms.get("total", [[0]])[i, pos]) if "total" in score_terms else 0.0,
                "latent_sim": float(score_terms.get("latent_sim", [[0]])[i, pos]) if "latent_sim" in score_terms else 0.0,
                "concept_overlap": float(score_terms.get("concept_overlap", [[0]])[i, pos]) if "concept_overlap" in score_terms else 0.0,
                "difficulty_sim": float(score_terms.get("difficulty_sim", [[0]])[i, pos]) if "difficulty_sim" in score_terms else 0.0,
                "recency": float(score_terms.get("recency", [[0]])[i, pos]) if "recency" in score_terms else 0.0,
                "student_id": _json_value(encoded.raw_students[row_index]),
                "query_exercise": _json_value(encoded.raw_exercises[row_index]),
                "query_concepts": str(encoded.raw_concept_text[row_index]),
                "query_difficulty": float(encoded.difficulties[row_index]),
                "hist_exercise": _json_value(encoded.raw_exercises[hist_row]),
                "hist_concepts": str(encoded.raw_concept_text[hist_row]),
                "hist_correct": float(encoded.labels[hist_row]),
                "hist_difficulty": float(hist_difficulty[i, pos]),
                "difficulty_gap": float(abs(encoded.difficulties[row_index] - hist_difficulty[i, pos])),
                "time_distance": float(hist_distance[i, pos]),
            }
            support_rows.append(support)
            supports.append(support)
        if case_limit_left > len(case_rows) and supports:
            case_rows.append({**pred, "supports": supports})
    return pred_rows, support_rows, case_rows
