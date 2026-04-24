"""
model.py — Feedforward neural network for CIFAR-10 using PrunableLinear layers.

Architecture
------------
Input  : 3 × 32 × 32 CIFAR-10 image  →  flattened to 3072 features
Hidden : 3072 → 1024 → 512 → 256  (each with ReLU + BatchNorm)
Output : 256 → 10  (class logits)

Every linear transformation uses PrunableLinear, so every weight has a
companion gate that can be driven toward zero by the sparsity loss.
"""

import torch
import torch.nn as nn
from prunable_layer import PrunableLinear
from config import ARCHITECTURE


class SelfPruningNet(nn.Module):
    """
    Self-pruning feedforward network for CIFAR-10 classification.

    All nn.Linear layers are replaced with PrunableLinear so that the
    sparsity regularisation loss can selectively silence weights during
    training.

    Parameters
    ----------
    architecture : list[int]
        Sequence of layer widths including input and output dimensions.
        Default is taken from config.ARCHITECTURE = [3072,1024,512,256,10].
    dropout_p : float
        Dropout probability applied after each hidden ReLU (0 = disabled).
    """

    def __init__(self, architecture=None, dropout_p: float = 0.1):
        super().__init__()

        if architecture is None:
            architecture = ARCHITECTURE

        layers = []
        for i in range(len(architecture) - 1):
            in_dim  = architecture[i]
            out_dim = architecture[i + 1]
            is_last = (i == len(architecture) - 2)

            # Prunable linear transformation
            layers.append(PrunableLinear(in_dim, out_dim))

            if not is_last:
                # Hidden layers: activation → normalisation → optional dropout
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.BatchNorm1d(out_dim))
                if dropout_p > 0:
                    layers.append(nn.Dropout(p=dropout_p))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten: [B, 3, 32, 32] → [B, 3072]
        x = x.view(x.size(0), -1)
        return self.net(x)

    def count_parameters(self) -> dict:
        """Return a breakdown of total vs gate parameters."""
        total   = sum(p.numel() for p in self.parameters())
        gates   = sum(m.gate_scores.numel()
                      for m in self.modules()
                      if isinstance(m, PrunableLinear))
        weights = sum(m.weight.numel() + m.bias.numel()
                      for m in self.modules()
                      if isinstance(m, PrunableLinear))
        return {"total": total, "weight+bias": weights, "gate_scores": gates}
