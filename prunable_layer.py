"""
prunable_layer.py - Custom PrunableLinear module (Part 1).

KEY FIX - gate_scores initialization:
  torch.zeros  -> sigmoid(0.0) = 0.50  -> stuck in middle, never moves
  torch.full(-1) -> sigmoid(-1.0) = 0.27 -> all identical, move in lockstep
  torch.randn * 0.5 - 1.5 -> sigmoid(-1.5) = 0.18 average, WITH spread
      The spread (std=0.5) is critical: different gates get different
      gradients -> some open (important), most close (unimportant).
      Without spread: lockstep movement -> always 0% or 100% sparsity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PrunableLinear(nn.Module):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.in_features  = in_features
        self.out_features = out_features

        # Standard weight + bias (same as nn.Linear internals)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))

        # Gate scores - THE CRITICAL PART:
        # Mean = -1.5 -> sigmoid(-1.5) = 0.18 -> gates start mostly closed
        # Std  =  0.5 -> gates are SPREAD OUT, not identical
        #
        # Why spread matters:
        #   If all gate_scores = same value -> same gradient -> all move together
        #   -> entire layer either fully pruned or fully open (binary flip)
        #   With std=0.5: each gate has unique value -> unique gradient
        #   -> important gates climb to 1, unimportant gates fall to 0
        #   -> smooth partial sparsity (30%, 60%, 80%) instead of 0% or 100%
        self.gate_scores = nn.Parameter(
            torch.randn(out_features, in_features) * 0.5 - 1.5
        )

        # Kaiming init for weights (standard for ReLU networks)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: convert gate_scores to gates in (0, 1)
        gates = torch.sigmoid(self.gate_scores)

        # Step 2: silence unimportant weights
        pruned_weights = self.weight * gates

        # Step 3: standard linear transform (x @ W.T + b)
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self) -> torch.Tensor:
        """Return gate values detached (for evaluation, no grad needed)."""
        return torch.sigmoid(self.gate_scores).detach()

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"
