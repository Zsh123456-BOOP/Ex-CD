from __future__ import annotations

from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F


TERM_ORDER = ("latent_sim", "concept_overlap", "difficulty_sim", "recency")


def concept_overlap(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    q = batch["query_concept_ids"].unsqueeze(1).unsqueeze(-1)
    h = batch["hist_concept_ids"].unsqueeze(2)
    q_mask = batch["query_concept_mask"].unsqueeze(1).unsqueeze(-1)
    h_mask = batch["hist_concept_mask"].unsqueeze(2)
    matches = ((q == h) & q_mask & h_mask & (q > 1)).any(dim=-1)
    inter = matches.float().sum(dim=-1)
    q_count = batch["query_concept_mask"].float().sum(dim=-1).unsqueeze(1)
    h_count = batch["hist_concept_mask"].float().sum(dim=-1)
    union = (q_count + h_count - inter).clamp_min(1.0)
    return inter / union


class RetrieverScorer(nn.Module):
    def __init__(self, embed_dim: int, term_flags: dict | None = None):
        super().__init__()
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.hist_proj = nn.Linear(embed_dim, embed_dim)
        self.raw_score_weights = nn.Parameter(torch.tensor([1.0, 1.0, 0.25, 0.25]))
        self.term_flags = {name: True for name in TERM_ORDER}
        self.term_flags["text_sim"] = False
        if term_flags:
            self.term_flags.update(term_flags)

    def forward(self, q, hist_value, batch):
        q_proj = F.normalize(self.query_proj(q), dim=-1)
        h_proj = F.normalize(self.hist_proj(hist_value), dim=-1)
        terms = {
            "latent_sim": (q_proj.unsqueeze(1) * h_proj).sum(dim=-1),
            "concept_overlap": concept_overlap(batch),
            "difficulty_sim": -torch.abs(batch["query_difficulty"].unsqueeze(1) - batch["hist_difficulty"]),
            "recency": batch["hist_recency"],
        }
        weights = F.softplus(self.raw_score_weights)
        total = torch.zeros_like(terms["latent_sim"])
        for idx, name in enumerate(TERM_ORDER):
            if self.term_flags.get(name, False):
                total = total + weights[idx] * terms[name]
        terms["text_sim"] = torch.zeros_like(total)
        terms["total"] = total
        return terms

