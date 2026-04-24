"""
config.py - Hyperparameters for the Self-Pruning Neural Network.

KEY INSIGHT on lambda values:
  Too small (1e-4): sparsity pressure too weak  -> 0% pruned
  Too large (1e-1): sparsity destroys everything -> 100% pruned
  Sweet spot:  1e-5 to 1e-4 when using sum(), or 1e-2 to 5e-2 when using mean()
  We use mean() in loss.py so these values are correct.
"""

import torch

# Three lambdas showing low / medium / high sparsity trade-off
LAMBDAS = [5e-3, 2e-2, 5e-2]

# Training
EPOCHS     = 40
LR         = 1e-3        # lower LR = more stable gate movement
BATCH_SIZE = 256

# Gate below this value = pruned
# 0.1 means "gate contributes less than 10% of the weight" -> effectively dead
PRUNE_THRESHOLD = 0.1

# Dataset
DATA_DIR = "./data"

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Network: input -> hidden layers -> output
ARCHITECTURE = [3072, 1024, 512, 256, 10]

# Outputs
PLOT_DIR    = "./plots"
REPORT_PATH = "./report.md"
