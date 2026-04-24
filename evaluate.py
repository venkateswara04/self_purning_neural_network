"""
evaluate.py - Sparsity metrics, test accuracy, and gate distribution plot.
"""

import os
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')   # non-interactive backend - saves file without needing a display
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from prunable_layer import PrunableLinear
from config import DEVICE, PRUNE_THRESHOLD, PLOT_DIR


def compute_sparsity(model: nn.Module):
    """Returns (sparsity_pct, all_gates_tensor)."""
    gate_tensors = []
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gate_tensors.append(module.get_gates().flatten().cpu())

    if not gate_tensors:
        raise ValueError("No PrunableLinear layers found in model.")

    all_gates    = torch.cat(gate_tensors)
    n_pruned     = (all_gates < PRUNE_THRESHOLD).sum().item()
    sparsity_pct = n_pruned / all_gates.numel() * 100.0
    return sparsity_pct, all_gates


@torch.no_grad()
def compute_accuracy(model: nn.Module, loader, device=DEVICE) -> float:
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds    = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return correct / total * 100.0


def plot_gate_distribution(all_gates: torch.Tensor, lam: float,
                           save_path: str = None) -> None:
    """
    Histogram of gate values.
    SUCCESS looks like: big spike at 0 (pruned) + cluster near 0.8-1.0 (active)
    FAILURE looks like: single spike stuck at one value (0.18 or 0.5)
    """
    gates_np = all_gates.numpy()
    n_total  = len(gates_np)
    n_pruned = (all_gates < PRUNE_THRESHOLD).sum().item()
    sparsity = n_pruned / n_total * 100

    fig, ax = plt.subplots(figsize=(10, 5))

    counts, bins, patches = ax.hist(gates_np, bins=100,
                                    color="#3B82F6", edgecolor="none", alpha=0.85)

    # Colour pruned region red
    for patch, left in zip(patches, bins[:-1]):
        if left < PRUNE_THRESHOLD:
            patch.set_facecolor("#EF4444")
            patch.set_alpha(0.9)

    ax.axvline(PRUNE_THRESHOLD, color="#EF4444", linestyle="--",
               linewidth=1.5, label=f"Prune threshold = {PRUNE_THRESHOLD}")

    ax.set_title(
        f"Gate distribution  (lambda={lam:.0e})  |  "
        f"{sparsity:.1f}% pruned  ({n_pruned:,} / {n_total:,} weights)",
        fontsize=13, pad=12
    )
    ax.set_xlabel("Gate value = sigmoid(gate_score)  in (0, 1)", fontsize=11)
    ax.set_ylabel("Number of weights", fontsize=11)
    ax.set_xlim(-0.02, 1.02)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )

    ax.text(0.02, 0.95, f"Pruned\n{sparsity:.1f}%",
            transform=ax.transAxes, fontsize=10,
            color="#EF4444", va="top", fontweight="bold")
    ax.text(0.90, 0.95, f"Active\n{100-sparsity:.1f}%",
            transform=ax.transAxes, fontsize=10,
            color="#3B82F6", va="top", ha="center", fontweight="bold")

    ax.legend(fontsize=10)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved -> {save_path}")

    plt.close(fig)


def evaluate(model: nn.Module, test_loader, lam: float, save_plot=True) -> dict:
    print(f"  Evaluating (lambda={lam:.0e}) ...")

    acc              = compute_accuracy(model, test_loader, DEVICE)
    sparsity_pct, all_gates = compute_sparsity(model)
    n_pruned         = (all_gates < PRUNE_THRESHOLD).sum().item()

    print(f"    Test accuracy : {acc:.2f}%")
    print(f"    Sparsity      : {sparsity_pct:.2f}%"
          f"  ({n_pruned:,} / {all_gates.numel():,} gates pruned)")
    print(f"    Gate mean     : {all_gates.mean().item():.4f}")
    print(f"    Gate min/max  : {all_gates.min().item():.4f} / {all_gates.max().item():.4f}")

    if save_plot:
        plot_path = os.path.join(PLOT_DIR, f"gates_lambda_{lam:.0e}.png")
        plot_gate_distribution(all_gates, lam, save_path=plot_path)

    return {
        "lambda":   lam,
        "accuracy": acc,
        "sparsity": sparsity_pct,
        "n_gates":  all_gates.numel(),
    }
