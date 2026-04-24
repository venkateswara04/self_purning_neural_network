"""
model.py - SelfPruningNet for CIFAR-10.
Input: 3x32x32 image -> flatten -> 3072
Output: 10 class logits
All linear layers are PrunableLinear.
"""

import torch
import torch.nn as nn
from prunable_layer import PrunableLinear
from config import ARCHITECTURE


class SelfPruningNet(nn.Module):

    def __init__(self, architecture=None, dropout_p: float = 0.0):
        super().__init__()

        # dropout_p=0.0 by default - dropout + gate sparsity can conflict
        if architecture is None:
            architecture = ARCHITECTURE

        layers = []
        for i in range(len(architecture) - 1):
            in_dim  = architecture[i]
            out_dim = architecture[i + 1]
            is_last = (i == len(architecture) - 2)

            layers.append(PrunableLinear(in_dim, out_dim))

            if not is_last:
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.BatchNorm1d(out_dim))
                if dropout_p > 0:
                    layers.append(nn.Dropout(p=dropout_p))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)   # [B,3,32,32] -> [B,3072]
        return self.net(x)

    def count_parameters(self) -> dict:
        total   = sum(p.numel() for p in self.parameters())
        gates   = sum(m.gate_scores.numel() for m in self.modules()
                      if isinstance(m, PrunableLinear))
        weights = sum(m.weight.numel() + m.bias.numel() for m in self.modules()
                      if isinstance(m, PrunableLinear))
        return {"total": total, "weight+bias": weights, "gate_scores": gates}
