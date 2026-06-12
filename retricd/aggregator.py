from __future__ import annotations

import torch
from torch import nn


class SupportAggregator(nn.Module):
    def __init__(self, topk: int, temperature: float):
        super().__init__()
        self.topk = topk
        self.temperature = temperature

    def topk_attention(self, scores, mask):
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

    def random_attention(self, mask):
        rand = torch.rand(mask.shape, device=mask.device).masked_fill(~mask, -1e9)
        return self.topk_attention(rand, mask)

    def zero_attention(self, mask):
        k = min(self.topk, mask.size(1))
        return torch.zeros(mask.shape, device=mask.device), torch.zeros((mask.size(0), k), dtype=torch.long, device=mask.device)

    def aggregate(self, hist_value, attn):
        return torch.bmm(attn.unsqueeze(1), hist_value).squeeze(1)

