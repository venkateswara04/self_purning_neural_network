"""
config.py — Centralised hyperparameters for the Self-Pruning Neural Network.
Edit LAMBDAS, EPOCHS, LR, or BATCH_SIZE here; every other file imports from this module.
"""

import torch

LAMBDAS = [1e-2, 5e-2, 1e-1]

EPOCHS     = 10         
LR         = 1e-3        
BATCH_SIZE = 256         

PRUNE_THRESHOLD = 0.05

DATA_DIR   = "./data"    # CIFAR-10 will be downloaded here automatically

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


ARCHITECTURE = [3072, 1024, 512, 256, 10]

PLOT_DIR   = "./plots"   # gate-distribution histograms saved here
REPORT_PATH = "./report.md"
