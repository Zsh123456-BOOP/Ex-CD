"""Training losses for Ex-CD variants: vanilla BCE, IPS-reweighted BCE, and a
doubly-robust (DR) estimator over the observed batch with an imputation head.
"""
from __future__ import annotations

import torch


def _bce_per_sample(prob: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    p = prob.clamp(1e-6, 1.0 - 1e-6)
    return -(label * torch.log(p) + (1.0 - label) * torch.log(1.0 - p))


def vanilla_bce(prob: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    return _bce_per_sample(prob, label).mean()


def ips_bce(prob: torch.Tensor, label: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (_bce_per_sample(prob, label) * weight).mean()


def dr_loss(prob, label, weight, imp_prob):
    """Doubly-robust loss over the observed batch.

    e_i  = BCE(prob_i, label_i)        (prediction error)
    e^_i = BCE(prob_i, imp_prob_i)     (imputed error from the imputation head, detached)
    DR_i = e^_i + w_i * (e_i - e^_i)

    Variance-reduced relative to plain IPS. The imputation head is trained separately with
    ``imputation_bce`` below. Returns (dr, imputation_target_loss).
    """
    e = _bce_per_sample(prob, label)
    e_hat = _bce_per_sample(prob, imp_prob.detach())
    dr = (e_hat + weight * (e - e_hat)).mean()
    return dr


def imputation_bce(imp_prob: torch.Tensor, label: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """IPS-weighted BCE used to fit the imputation head toward observed responses."""
    return (_bce_per_sample(imp_prob, label) * weight).mean()
