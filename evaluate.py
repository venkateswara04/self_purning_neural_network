"""
evaluate.py — Sparsity metrics, test accuracy, and gate distribution plot.

After training, call:
    sparsity_pct, all_gates = compute_sparsity(model)
    acc                     = compute_accuracy(model, test_loader, DEVICE)
    plot_gate_distribution(all_gates, lam, save_path="gates.png")
"""

import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from prunable_layer import PrunableLinear
from config import DEVICE, PRUNE_THRESHOLD, PLOT_DIR



def compute_sparsity(model: nn.Module) -> tuple[float, torch.Tensor]:
    """
    Compute the percentage of gates below PRUNE_THRESHOLD.

    A gate value < PRUNE_THRESHOLD ≈ 0 means that weight is effectively
    pruned (its contribution to the forward pass is negligible).

    Parameters
    ----------
    model : nn.Module  (must contain PrunableLinear layers)

    Returns
    -------
    sparsity_pct : float        — percentage of pruned weights (0–100)
    all_gates    : torch.Tensor — flat tensor of every gate value
    """
    gate_tensors = []

    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gate_tensors.append(module.get_gates().flatten().cpu())

    if not gate_tensors:
        raise ValueError("No PrunableLinear layers found in model.")

    all_gates = torch.cat(gate_tensors)

    n_pruned      = (all_gates < PRUNE_THRESHOLD).sum().item()
    sparsity_pct  = n_pruned / all_gates.numel() * 100.0

    return sparsity_pct, all_gates



@torch.no_grad()
def compute_accuracy(
    model:  nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device = DEVICE,
) -> float:
    """
    Evaluate classification accuracy on a data loader.

    Returns
    -------
    float — accuracy in percent (0–100)
    """
    model.eval()
    correct = 0
    total   = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        preds  = logits.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    return correct / total * 100.0




def plot_gate_distribution(
    all_gates: torch.Tensor,
    lam:       float,
    save_path: str | None = None,
) -> None:
    """
    Plot a histogram of gate values for a trained model.

    A successful result shows:
    - A large spike at gate ≈ 0  (pruned / silenced weights)
    - A secondary cluster near gate ≈ 1  (surviving weights)
    - Very little in between  (L1 forces a binary outcome)

    Parameters
    ----------
    all_gates : flat Tensor of gate values from compute_sparsity()
    lam       : lambda used during training (displayed in title)
    save_path : if given, save the figure to this path
    """
    gates_np = all_gates.numpy()
    n_total  = len(gates_np)
    n_pruned = (all_gates < PRUNE_THRESHOLD).sum().item()
    sparsity = n_pruned / n_total * 100

    fig, ax = plt.subplots(figsize=(9, 4))

    # Main histogram
    counts, bins, patches = ax.hist(
        gates_np, bins=120, color="#3B82F6", edgecolor="none", alpha=0.85
    )

    # Colour the pruned-region bars differently
    for patch, left in zip(patches, bins[:-1]):
        if left < PRUNE_THRESHOLD:
            patch.set_facecolor("#EF4444")
            patch.set_alpha(0.9)

    # Threshold line
    ax.axvline(PRUNE_THRESHOLD, color="#EF4444", linestyle="--",
               linewidth=1.2, label=f"Prune threshold ({PRUNE_THRESHOLD})")

    # Annotations
    ax.set_title(
        f"Gate value distribution  (λ = {lam:.0e})  —  "
        f"{sparsity:.1f}% pruned  ({n_pruned:,} / {n_total:,} weights)",
        fontsize=12, pad=12
    )
    ax.set_xlabel("Gate value  σ(gate_score)  ∈ (0, 1)", fontsize=11)
    ax.set_ylabel("Number of weights", fontsize=11)
    ax.set_xlim(-0.02, 1.02)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x):,}"
    ))

    # Annotate the spike at 0
    ax.text(0.03, 0.92, f"Pruned\n{sparsity:.1f}%",
            transform=ax.transAxes, fontsize=9,
            color="#EF4444", va="top")
    ax.text(0.88, 0.92, f"Active\n{100-sparsity:.1f}%",
            transform=ax.transAxes, fontsize=9,
            color="#3B82F6", va="top", ha="center")

    ax.legend(fontsize=10)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved → {save_path}")

    plt.show()
    plt.close(fig)



def evaluate(
    model:       nn.Module,
    test_loader: torch.utils.data.DataLoader,
    lam:         float,
    save_plot:   bool = True,
) -> dict:
    """
    Run all evaluation steps and return a results dict.

    Parameters
    ----------
    model       : trained SelfPruningNet
    test_loader : DataLoader for the test split
    lam         : lambda used during training
    save_plot   : whether to save the gate histogram to PLOT_DIR

    Returns
    -------
    dict with keys: lambda, accuracy, sparsity, n_gates
    """
    print(f"  Evaluating model (λ = {lam:.0e}) …")

    acc                   = compute_accuracy(model, test_loader, DEVICE)
    sparsity_pct, all_gates = compute_sparsity(model)

    print(f"    Test accuracy : {acc:.2f}%")
    print(f"    Sparsity      : {sparsity_pct:.2f}%  "
          f"({(all_gates < PRUNE_THRESHOLD).sum().item():,} / "
          f"{all_gates.numel():,} gates pruned)")

    if save_plot:
        plot_path = os.path.join(PLOT_DIR, f"gates_lambda_{lam:.0e}.png")
        plot_gate_distribution(all_gates, lam, save_path=plot_path)

    return {
        "lambda":   lam,
        "accuracy": acc,
        "sparsity": sparsity_pct,
        "n_gates":  all_gates.numel(),
    }
