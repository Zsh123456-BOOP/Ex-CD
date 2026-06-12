from __future__ import annotations

from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weights = mask.float()
    while weights.dim() < values.dim():
        weights = weights.unsqueeze(-1)
    total = (values * weights).sum(dim=dim)
    denom = weights.sum(dim=dim).clamp_min(1.0)
    return total / denom


def _concept_overlap(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
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


class RetriCDModel(nn.Module):
    def __init__(
        self,
        num_exercises: int,
        num_concepts: int,
        embed_dim: int = 64,
        topk: int = 16,
        temperature: float = 0.5,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.topk = topk
        self.temperature = temperature
        self.exercise_emb = nn.Embedding(num_exercises, embed_dim, padding_idx=0)
        self.concept_emb = nn.Embedding(num_concepts, embed_dim, padding_idx=0)
        self.diff_proj = nn.Linear(1, embed_dim)
        self.correct_emb = nn.Embedding(2, embed_dim)
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.hist_proj = nn.Linear(embed_dim, embed_dim)
        self.raw_score_weights = nn.Parameter(torch.tensor([1.0, 1.0, 0.25, 0.25]))
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim * 6, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, 1),
        )

    def _item_repr(self, exercise_id, concept_ids, concept_mask, difficulty):
        exercise = self.exercise_emb(exercise_id)
        concept = _masked_mean(self.concept_emb(concept_ids), concept_mask, dim=-2)
        diff = self.diff_proj(difficulty.unsqueeze(-1).float())
        return torch.tanh(exercise + concept + diff)

    def _encode_history(self, batch):
        bsz, hist_len = batch["hist_exercise_id"].shape
        flat_item = self._item_repr(
            batch["hist_exercise_id"].reshape(-1),
            batch["hist_concept_ids"].reshape(bsz * hist_len, -1),
            batch["hist_concept_mask"].reshape(bsz * hist_len, -1),
            batch["hist_difficulty"].reshape(-1),
        ).reshape(bsz, hist_len, -1)
        correct_idx = (batch["hist_correct"] > 0.5).long()
        return flat_item + self.correct_emb(correct_idx)

    def _retrieval_scores(self, q, hist_value, batch):
        q_proj = F.normalize(self.query_proj(q), dim=-1)
        h_proj = F.normalize(self.hist_proj(hist_value), dim=-1)
        latent_sim = (q_proj.unsqueeze(1) * h_proj).sum(dim=-1)
        overlap = _concept_overlap(batch)
        difficulty_sim = -torch.abs(batch["query_difficulty"].unsqueeze(1) - batch["hist_difficulty"])
        recency = batch["hist_recency"]
        terms = {
            "latent_sim": latent_sim,
            "concept_overlap": overlap,
            "difficulty_sim": difficulty_sim,
            "recency": recency,
        }
        weights = F.softplus(self.raw_score_weights)
        total = (
            weights[0] * latent_sim
            + weights[1] * overlap
            + weights[2] * difficulty_sim
            + weights[3] * recency
        )
        terms["total"] = total
        return terms

    def _topk_attention(self, scores, mask):
        scores = scores.masked_fill(~mask, -1e9)
        has_hist = mask.any(dim=1)
        k = min(self.topk, scores.size(1))
        top_idx = torch.topk(scores, k=k, dim=1).indices
        top_mask = torch.zeros_like(mask)
        top_mask.scatter_(1, top_idx, True)
        masked_scores = scores.masked_fill(~top_mask, -1e9) / max(self.temperature, 1e-6)
        masked_scores = torch.where(has_hist.unsqueeze(1), masked_scores, torch.zeros_like(masked_scores))
        attn = torch.softmax(masked_scores, dim=1)
        attn = torch.where(has_hist.unsqueeze(1), attn, torch.zeros_like(attn))
        return attn, top_idx

    def _random_attention(self, mask):
        rand = torch.rand(mask.shape, device=mask.device).masked_fill(~mask, -1e9)
        return self._topk_attention(rand, mask)

    def forward(self, batch: Dict[str, torch.Tensor], support_mode: str = "retrieval"):
        q = self._item_repr(
            batch["query_exercise_id"],
            batch["query_concept_ids"],
            batch["query_concept_mask"],
            batch["query_difficulty"],
        )
        hist_value = self._encode_history(batch)
        hist_mask = batch["hist_mask"].bool()
        pool_w = (batch["hist_recency"] * hist_mask.float()).unsqueeze(-1)
        global_state = (hist_value * pool_w).sum(dim=1) / pool_w.sum(dim=1).clamp_min(1.0)

        if support_mode == "random":
            score_terms = {}
            attn, top_idx = self._random_attention(hist_mask)
        else:
            score_terms = self._retrieval_scores(q, hist_value, batch)
            attn, top_idx = self._topk_attention(score_terms["total"], hist_mask)
        support = torch.bmm(attn.unsqueeze(1), hist_value).squeeze(1)
        features = torch.cat([q, global_state, support, q * support, global_state * support, q - support], dim=-1)
        logit = self.predictor(features).squeeze(-1)
        return {
            "logit": logit,
            "prob": torch.sigmoid(logit),
            "attn": attn,
            "topk_idx": top_idx,
            "score_terms": score_terms,
            "support": support,
            "global_state": global_state,
        }

