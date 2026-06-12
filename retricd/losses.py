from __future__ import annotations

import torch
import torch.nn.functional as F


def bce_loss(logit: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logit, label.float())


def fidelity_margin_loss(full_bce: torch.Tensor, random_bce: torch.Tensor, margin: float) -> torch.Tensor:
    return torch.relu(torch.tensor(margin, device=full_bce.device) + full_bce - random_bce.detach())

