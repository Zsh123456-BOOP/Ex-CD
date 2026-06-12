from __future__ import annotations

import torch
from torch import nn


class PredictionHead(nn.Module):
    def __init__(self, embed_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 6, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, 1),
        )

    def forward(self, q, global_state, support):
        features = torch.cat([q, global_state, support, q * support, global_state * support, q - support], dim=-1)
        return self.net(features).squeeze(-1)
