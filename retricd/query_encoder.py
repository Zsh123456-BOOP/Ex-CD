from __future__ import annotations

import torch
from torch import nn


def masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weights = mask.float()
    while weights.dim() < values.dim():
        weights = weights.unsqueeze(-1)
    total = (values * weights).sum(dim=dim)
    denom = weights.sum(dim=dim).clamp_min(1.0)
    return total / denom


class QueryEncoder(nn.Module):
    def __init__(self, num_exercises: int, num_concepts: int, embed_dim: int):
        super().__init__()
        self.exercise_emb = nn.Embedding(num_exercises, embed_dim, padding_idx=0)
        self.concept_emb = nn.Embedding(num_concepts, embed_dim, padding_idx=0)
        self.diff_proj = nn.Linear(1, embed_dim)

    def forward(self, exercise_id, concept_ids, concept_mask, difficulty):
        exercise = self.exercise_emb(exercise_id)
        concept = masked_mean(self.concept_emb(concept_ids), concept_mask, dim=-2)
        diff = self.diff_proj(difficulty.unsqueeze(-1).float())
        return torch.tanh(exercise + concept + diff)

