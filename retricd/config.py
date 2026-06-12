from __future__ import annotations

from dataclasses import dataclass, field
import random

import numpy as np
import torch


DEFAULT_DATASETS = ("assist_09", "assist_17", "junyi", "nips34_retricd_small")
RAW_DATASETS = ("assist_09", "assist_17", "junyi", "nips34")


@dataclass
class RunConfig:
    dataset: str
    data_root: str = "data_retricd"
    output_dir: str = "outputs/retricd_full"
    variant: str = "full"
    seed: int = 42
    max_history_len: int = 128
    max_concepts: int = 8
    topk: int = 16
    embed_dim: int = 64
    batch_size: int = 256
    epochs: int = 30
    patience: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-5
    dropout: float = 0.15
    temperature: float = 0.5
    fidelity_margin_weight: float = 0.2
    fidelity_margin: float = 0.02
    num_workers: int = 0
    export_support_limit: int = 50000
    case_limit: int = 200
    device: str = "auto"
    use_student_id_embedding: bool = False
    use_text_branch: bool = False
    retriever_terms: dict = field(default_factory=dict)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def term_flags_for_variant(variant: str) -> dict:
    all_terms = {
        "latent_sim": True,
        "concept_overlap": True,
        "difficulty_sim": True,
        "recency": True,
        "text_sim": False,
    }
    if variant == "full":
        return all_terms
    if variant == "recency_only":
        return {k: k == "recency" for k in all_terms}
    if variant == "overlap_only":
        return {k: k == "concept_overlap" for k in all_terms}
    if variant == "difficulty_only":
        return {k: k == "difficulty_sim" for k in all_terms}
    if variant == "latent_only":
        return {k: k == "latent_sim" for k in all_terms}
    return all_terms


def train_support_mode_for_variant(variant: str) -> str:
    if variant == "no_retrieval":
        return "zero"
    if variant == "random_retrieval":
        return "random"
    return "retrieval"
