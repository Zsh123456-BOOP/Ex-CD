from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from retricd.aggregator import SupportAggregator
from retricd.memory_bank import HistoryEncoder, masked_recency_pool
from retricd.predictor import PredictionHead
from retricd.query_encoder import QueryEncoder
from retricd.retriever import RetrieverScorer


def _roll_history_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    rolled = dict(batch)
    for key in list(batch.keys()):
        if key.startswith("hist_"):
            rolled[key] = torch.roll(batch[key], shifts=1, dims=0)
    return rolled


class RetriCDModel(nn.Module):
    def __init__(
        self,
        num_exercises: int,
        num_concepts: int,
        embed_dim: int = 64,
        topk: int = 16,
        temperature: float = 0.5,
        dropout: float = 0.15,
        term_flags: dict | None = None,
    ):
        super().__init__()
        self.query_encoder = QueryEncoder(num_exercises, num_concepts, embed_dim)
        self.memory_encoder = HistoryEncoder(self.query_encoder, embed_dim)
        self.retriever = RetrieverScorer(embed_dim, term_flags=term_flags)
        self.aggregator = SupportAggregator(topk=topk, temperature=temperature)
        self.predictor = PredictionHead(embed_dim, dropout)

    def forward(self, batch: Dict[str, torch.Tensor], support_mode: str = "retrieval"):
        score_batch = _roll_history_batch(batch) if support_mode == "shuffle_student" else batch
        q = self.query_encoder(
            batch["query_exercise_id"],
            batch["query_concept_ids"],
            batch["query_concept_mask"],
            batch["query_difficulty"],
        )
        hist_value = self.memory_encoder(score_batch)
        hist_mask = score_batch["hist_mask"].bool()
        global_state = masked_recency_pool(hist_value, hist_mask, score_batch["hist_recency"])

        if support_mode == "zero":
            score_terms = {}
            attn, top_idx = self.aggregator.zero_attention(hist_mask)
        elif support_mode == "random":
            score_terms = {}
            attn, top_idx = self.aggregator.random_attention(hist_mask)
        else:
            score_terms = self.retriever(q, hist_value, score_batch)
            attn, top_idx = self.aggregator.topk_attention(score_terms["total"], hist_mask)
        support = self.aggregator.aggregate(hist_value, attn)
        logit = self.predictor(q, global_state, support)
        return {
            "logit": logit,
            "prob": torch.sigmoid(logit),
            "attn": attn,
            "topk_idx": top_idx,
            "score_terms": score_terms,
            "support": support,
            "global_state": global_state,
        }

