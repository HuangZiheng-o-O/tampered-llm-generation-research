"""The three pilot objectives, factored out for direct comparison with source code.

GradDiff and NPO follow OpenUnlearning's grad_diff.py, npo.py and utils.py;
RMU follows OpenUnlearning's rmu.py and the WMDP reference implementation.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def sequence_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shifted = labels[..., 1:].contiguous()
    scores = logits[..., :-1, :].contiguous()
    per_token = F.cross_entropy(scores.transpose(-1, -2).float(), shifted,
                                ignore_index=-100, reduction="none")
    return per_token.sum(dim=-1)


def graddiff(forget_nll: torch.Tensor, retain_nll: torch.Tensor,
             alpha: float) -> torch.Tensor:
    return -forget_nll + alpha * retain_nll


def npo(current_sum_nll: torch.Tensor, reference_sum_nll: torch.Tensor,
        retain_nll: torch.Tensor, beta: float, alpha: float) -> torch.Tensor:
    if beta <= 0:
        raise ValueError("NPO beta must be positive")
    forget = (-2.0 / beta) * F.logsigmoid(beta *
                                         (current_sum_nll - reference_sum_nll)).mean()
    return forget + alpha * retain_nll


def masked_activation_mse(current: torch.Tensor, target: torch.Tensor,
                          label_mask: torch.Tensor) -> torch.Tensor:
    if current.shape != target.shape or current.shape[:2] != label_mask.shape:
        raise ValueError("RMU activations or token mask have incompatible shapes")
    count = label_mask.sum(dim=-1).clamp_min(1)
    per_token = (current.float() - target.float()).square().mean(dim=-1)
    return ((per_token * label_mask).sum(dim=-1) / count).mean()
