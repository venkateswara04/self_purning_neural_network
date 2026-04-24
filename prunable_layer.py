"""
prunable_layer.py — Custom PrunableLinear module (Part 1).

Each weight in the layer has a paired learnable 'gate_score' scalar.
During the forward pass:
    gate        = sigmoid(gate_score)   ∈ (0, 1)
    pruned_w    = weight × gate         element-wise
    output      = x @ pruned_w.T + bias

When the sparsity loss drives gate_score → -∞, gate → 0, effectively
removing that weight from the network.  Gradients flow through both
'weight' and 'gate_scores' automatically because both are nn.Parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PrunableLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear that adds a learnable gate for
    every weight.  Use exactly like nn.Linear:

        layer = PrunableLinear(in_features=512, out_features=256)
        out   = layer(x)

    Extra methods
    -------------
    get_gates() -> Tensor   : current gate values (detached, for evaluation)
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.in_features  = in_features
        self.out_features = out_features


        self.weight = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        self.bias = nn.Parameter(
            torch.zeros(out_features)
        )

        # ── Gate scores — same shape as weight ────────────────────────────
        # Initialised to 0  →  sigmoid(0) = 0.5  →  gates start half-open.
        # The sparsity loss will push most toward -∞  →  gate ≈ 0 (pruned).
        # Important weights survive because the classification loss pushes
        # their gate_scores positive  →  gate ≈ 1 (open).
        self.gate_scores = nn.Parameter(
            torch.full((out_features, in_features),-1.0)
        )

        # ── Weight initialisation (Kaiming uniform for ReLU networks) ─────
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Initialise bias the same way nn.Linear does
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(self.gate_scores)          

        pruned_weights = self.weight * gates             

        
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self) -> torch.Tensor:
        """Return current gate values as a detached tensor (for evaluation)."""
        return torch.sigmoid(self.gate_scores).detach()

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}")
