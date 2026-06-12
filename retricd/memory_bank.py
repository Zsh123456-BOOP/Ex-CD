from __future__ import annotations

import torch
from torch import nn

from retricd.query_encoder import QueryEncoder


class HistoryEncoder(nn.Module):
    def __init__(self, item_encoder: QueryEncoder, embed_dim: int):
        super().__init__()
        self.item_encoder = item_encoder
        self.correct_emb = nn.Embedding(2, embed_dim)

    def forward(self, batch):
        bsz, hist_len = batch["hist_exercise_id"].shape
        item = self.item_encoder(
            batch["hist_exercise_id"].reshape(-1),
            batch["hist_concept_ids"].reshape(bsz * hist_len, -1),
            batch["hist_concept_mask"].reshape(bsz * hist_len, -1),
            batch["hist_difficulty"].reshape(-1),
        ).reshape(bsz, hist_len, -1)
        correct_idx = (batch["hist_correct"] > 0.5).long()
        return item + self.correct_emb(correct_idx)


def masked_recency_pool(values: torch.Tensor, mask: torch.Tensor, recency: torch.Tensor) -> torch.Tensor:
    weights = (mask.float() * recency.float()).unsqueeze(-1)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

