"""
loss.py — Sparsity regularisation loss (Part 2).

Total Loss  =  CrossEntropy(logits, labels)  +  λ × SparsityLoss

SparsityLoss is the L1 norm of ALL gate values across every PrunableLinear
layer in the model:

    SparsityLoss = Σ  sigmoid(gate_score_i)   for every i

Why L1 encourages sparsity
--------------------------
The gradient of |x| with respect to x is sign(x) — a constant ±1 that
does NOT diminish as x approaches zero.  This means the optimiser keeps
pushing gate values all the way to 0 rather than merely shrinking them
(as L2 would do).  Since sigmoid output is always positive, |gate| = gate,
so the penalty is simply the sum of gate values.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from prunable_layer import PrunableLinear




def sparsity_loss(model: nn.Module) -> torch.Tensor:
    """
    Compute the L1 norm of all gate values in every PrunableLinear layer.

    A higher value means more gates are open (non-zero), so minimising this
    term drives gates toward zero — effectively pruning the corresponding
    weights.

    Parameters
    ----------
    model : nn.Module
        The SelfPruningNet (or any model containing PrunableLinear layers).

    Returns
    -------
    torch.Tensor
        Scalar tensor; differentiable w.r.t. all gate_scores parameters.
    """
    total = torch.tensor(0.0, device=next(model.parameters()).device)

    for module in model.modules():
        if isinstance(module, PrunableLinear):
            # sigmoid maps gate_scores to (0, 1)
            # sum = L1 norm because all values are positive
            gates = torch.sigmoid(module.gate_scores)
            total = total + gates.sum()

    return total


def total_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    model:  nn.Module,
    lam:    float,
) -> tuple[torch.Tensor, float, float]:
    """
    Combined classification + sparsity loss.

    Parameters
    ----------
    logits : Tensor [batch, num_classes]
        Raw (un-normalised) model output.
    labels : Tensor [batch]
        Ground-truth class indices.
    model  : nn.Module
        The model (must contain PrunableLinear layers).
    lam    : float
        Lambda — weight of the sparsity term.
        - Small λ (e.g. 1e-4) → minimal pruning, high accuracy.
        - Large λ (e.g. 1e-2) → aggressive pruning, lower accuracy.

    Returns
    -------
    loss   : Tensor   — differentiable total loss (for .backward())
    ce_val : float    — cross-entropy component (for logging)
    sp_val : float    — sparsity component (for logging)
    """
    ce_loss = F.cross_entropy(logits, labels)
    sp_loss = sparsity_loss(model)

    loss = ce_loss + lam * sp_loss

    return loss, ce_loss.item(), sp_loss.item()
