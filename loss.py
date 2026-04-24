"""
loss.py - Sparsity regularisation loss (Part 2).

Total Loss = CrossEntropy(logits, labels) + lambda * SparsityLoss


  sum()  -> SparsityLoss = 3,800,000 * 0.18 = ~684,000
            lambda * 684,000 >> CrossEntropy (~2.0)
            Sparsity term completely dominates
            Network has two outcomes:
              a) Resist sparsity -> gates stuck at 0.18 -> 0% pruned
              b) Yield to sparsity -> all gates -> 0 -> 100% pruned
            No middle ground possible.

  mean() -> SparsityLoss = 0.18  (always between 0 and 1)
            lambda * 0.18 is comparable to CrossEntropy (~2.0)
            Both losses contribute meaningfully
            Result: partial sparsity (30-80%) depending on lambda

WHY L1 (not L2) creates sparsity:
  L1 gradient = constant 1.0 regardless of gate size
             -> keeps pushing gate toward 0 even when gate is tiny
  L2 gradient = 2 * gate, shrinks as gate -> 0
             -> never fully zeros a gate, just makes it small
  L1 creates TRUE zeros; L2 creates small-but-nonzero values
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from prunable_layer import PrunableLinear


def sparsity_loss(model: nn.Module) -> torch.Tensor:
    """
    MEAN of all sigmoid(gate_scores) across all PrunableLinear layers.
    Returns a scalar in (0, 1) regardless of model size.
    Fully differentiable - gradients flow back to gate_scores.
    """
    all_gates = []

    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores)
            all_gates.append(gates.reshape(-1))

    if not all_gates:
        raise ValueError("No PrunableLinear layers found in model.")

    # MEAN not sum - this is the key fix
    # Keeps SparsityLoss in (0,1) so lambda controls the actual balance
    return torch.cat(all_gates).mean()


def total_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    model:  nn.Module,
    lam:    float,
) -> tuple:
    """
    Combined loss = CrossEntropy + lambda * SparsityLoss

    Both terms are on comparable scales:
      CrossEntropy ~ 1.0 to 3.0
      SparsityLoss ~ 0.0 to 1.0  (mean of gates)
      lambda controls what fraction of total loss is sparsity

    Returns: (loss tensor, ce float, sparsity float)
    """
    ce_loss = F.cross_entropy(logits, labels)   # ~1.0-3.0
    sp_loss = sparsity_loss(model)              # ~0.0-1.0

    loss = ce_loss + lam * sp_loss
    return loss, ce_loss.item(), sp_loss.item()
