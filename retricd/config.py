from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
import torch


DEFAULT_DATASETS = ("assist_09", "assist_17", "junyi", "nips34")


@dataclass
class RunConfig:
    dataset: str
    data_root: str = "data"
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

