"""Cognitive-diagnosis backbones for Ex-CD: NCDM and a KaNCD-style low-rank variant.

Both produce an interpretable per-concept mastery vector and a monotone interaction network
(positive weights), so they are validated by DOA + monotonicity, not just response prediction.
An optional imputation head supports the doubly-robust (DR) training variant.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class InteractionNet(nn.Module):
    """Monotone interaction net (canonical NeuralCDM).

    Plain Linear layers whose weights are clamped to >= 0 by ``clip_weights()`` AFTER each
    optimizer step. Clamping inside ``forward`` (the previous approach) gives negative weights
    zero gradient, so they die and the network collapses to a constant output — the
    post-step clipper keeps gradients flowing and is stable.
    """

    def __init__(self, num_concepts: int, hidden=(256, 128), dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(num_concepts, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc3 = nn.Linear(hidden[1], 1)
        self.drop = nn.Dropout(dropout)
        for fc in (self.fc1, self.fc2, self.fc3):
            nn.init.xavier_normal_(fc.weight)
            nn.init.zeros_(fc.bias)
        self.clip_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(torch.sigmoid(self.fc1(x)))
        x = self.drop(torch.sigmoid(self.fc2(x)))
        return torch.sigmoid(self.fc3(x)).squeeze(-1)

    def clip_weights(self) -> None:
        for fc in (self.fc1, self.fc2, self.fc3):
            fc.weight.data.clamp_(min=0.0)


class CDM(nn.Module):
    def __init__(
        self,
        num_students: int,
        num_exercises: int,
        num_concepts: int,
        model_type: str = "ncdm",
        latent_dim: int = 32,
        hidden=(256, 128),
        dropout: float = 0.2,
        with_imputation: bool = False,
    ):
        super().__init__()
        self.model_type = model_type
        self.num_concepts = num_concepts
        self.with_imputation = with_imputation

        if model_type == "ncdm":
            self.student_emb = nn.Embedding(num_students, num_concepts)
            self.diff_emb = nn.Embedding(num_exercises, num_concepts)
            nn.init.xavier_normal_(self.student_emb.weight)
            nn.init.xavier_normal_(self.diff_emb.weight)
        elif model_type == "kancd":
            self.student_emb = nn.Embedding(num_students, latent_dim)
            self.exer_emb = nn.Embedding(num_exercises, latent_dim)
            self.knowledge = nn.Parameter(torch.randn(num_concepts, latent_dim) * 0.1)
            nn.init.xavier_normal_(self.student_emb.weight)
            nn.init.xavier_normal_(self.exer_emb.weight)
        else:
            raise ValueError(f"unknown model_type: {model_type}")

        self.disc_emb = nn.Embedding(num_exercises, 1)
        nn.init.xavier_normal_(self.disc_emb.weight)

        self.net = InteractionNet(num_concepts, hidden, dropout)
        if with_imputation:
            self.imp = nn.Sequential(
                nn.Linear(num_concepts, hidden[1]),
                nn.ReLU(),
                nn.Linear(hidden[1], 1),
                nn.Sigmoid(),
            )

    def _proficiency_difficulty(self, student, exercise) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.model_type == "ncdm":
            h_s = torch.sigmoid(self.student_emb(student))
            h_diff = torch.sigmoid(self.diff_emb(exercise))
        else:
            stu = self.student_emb(student)
            exer = self.exer_emb(exercise)
            h_s = torch.sigmoid(stu @ self.knowledge.t())
            h_diff = torch.sigmoid(exer @ self.knowledge.t())
        return h_s, h_diff

    def forward(self, student, exercise, concept_mask):
        h_s, h_diff = self._proficiency_difficulty(student, exercise)
        disc = torch.sigmoid(self.disc_emb(exercise))  # [B, 1]
        x = disc * (h_s - h_diff) * concept_mask
        prob = self.net(x).clamp(1e-6, 1.0 - 1e-6)
        return prob, x

    def impute(self, x: torch.Tensor) -> torch.Tensor:
        if not self.with_imputation:
            raise RuntimeError("model built without imputation head (with_imputation=False)")
        return self.imp(x).squeeze(-1).clamp(1e-6, 1.0 - 1e-6)

    def clip_monotonicity(self) -> None:
        """Enforce non-negative interaction weights. Call AFTER each optimizer step."""
        self.net.clip_weights()

    def smoothing_penalty(self, propensity: torch.Tensor) -> torch.Tensor:
        """Exposure-weighted shrinkage of per-concept proficiency toward each student's mean
        ability. Rare concepts (low propensity) are under-identified; pulling them toward the
        student's general ability lets them borrow strength, which IPS reweighting cannot do.
        ``propensity`` is a [num_concepts] tensor in (0, 1]; weight = (1 - propensity)."""
        if self.model_type == "ncdm":
            prof = self.student_emb.weight                      # [S, K] pre-sigmoid proficiency
        else:
            prof = self.student_emb.weight @ self.knowledge.t()  # [S, K] derived
        row_mean = prof.mean(dim=1, keepdim=True)
        w = (1.0 - propensity).clamp(min=0.0)                    # [K]
        return (((prof - row_mean) ** 2) * w.unsqueeze(0)).mean()

    @torch.no_grad()
    def mastery_matrix(self, device, batch: int = 8192) -> np.ndarray:
        """Return the [num_students, num_concepts] per-concept mastery artifact."""
        was_training = self.training
        self.eval()
        num_students = self.student_emb.num_embeddings
        chunks = []
        for start in range(0, num_students, batch):
            idx = torch.arange(start, min(start + batch, num_students), device=device)
            if self.model_type == "ncdm":
                m = torch.sigmoid(self.student_emb(idx))
            else:
                m = torch.sigmoid(self.student_emb(idx) @ self.knowledge.t())
            chunks.append(m.cpu().numpy())
        if was_training:
            self.train()
        return np.concatenate(chunks, axis=0)
