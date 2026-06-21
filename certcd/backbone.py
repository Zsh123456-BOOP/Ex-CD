"""Backbone trainer for CertCD — reuses the validated excd CDM (NCDM/KaNCD), losses, data.

Stable training recipe carried over from the Ex-CD debugging: post-step monotonicity clipper
(not clamp-in-forward), gradient-norm clipping, lr 1e-3, dropout 0.2, early stopping on valid AUC.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch.utils.data import DataLoader

from excd.data import make_collate
from excd.losses import vanilla_bce
from excd.metrics import binary_metrics
from excd.model import CDM


@torch.no_grad()
def predict(model, loader, device) -> np.ndarray:
    model.eval()
    out = []
    for b in loader:
        p, _ = model(b["student"].to(device), b["exercise"].to(device), b["concept_mask"].to(device))
        out.append(p.cpu().numpy())
    return np.concatenate(out)


def train_backbone(
    vocab,
    train_ds,
    valid_ds,
    *,
    model_type: str = "ncdm",
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 256,
    dropout: float = 0.2,
    latent_dim: int = 32,
    grad_clip: float = 5.0,
    patience: int = 8,
    num_workers: int = 2,
    device="cpu",
    seed: int = 42,
    log=print,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    collate = make_collate(vocab.num_concepts)
    tl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=num_workers)
    vl = DataLoader(valid_ds, batch_size=1024, shuffle=False, collate_fn=collate, num_workers=num_workers)

    model = CDM(
        vocab.num_students, vocab.num_exercises, vocab.num_concepts,
        model_type=model_type, latent_dim=latent_dim, dropout=dropout,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best, best_state, best_ep, bad = -1.0, copy.deepcopy(model.state_dict()), 0, 0
    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for b in tl:
            prob, _ = model(b["student"].to(device), b["exercise"].to(device), b["concept_mask"].to(device))
            loss = vanilla_bce(prob, b["label"].to(device))
            opt.zero_grad()
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            model.clip_monotonicity()
            losses.append(float(loss.detach().cpu()))
        vm = binary_metrics(valid_ds.label, predict(model, vl, device))
        log(f"  epoch {ep:02d} | loss {np.mean(losses):.4f} | valid_auc {vm['auc']:.4f}")
        if vm["auc"] > best:
            best, best_state, best_ep, bad = vm["auc"], copy.deepcopy(model.state_dict()), ep, 0
        else:
            bad += 1
            if bad >= patience:
                log(f"  early stop @ ep{ep} (best {best:.4f} @ ep{best_ep})")
                break
    model.load_state_dict(best_state)
    return model, best, best_ep


def discrimination_by_exercise(model, device) -> np.ndarray:
    with torch.no_grad():
        return torch.sigmoid(model.disc_emb.weight).detach().cpu().numpy().reshape(-1)
