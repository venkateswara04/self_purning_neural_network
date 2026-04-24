"""
main.py — Entry point for the Self-Pruning Neural Network experiment.

Runs training + evaluation for every lambda in config.LAMBDAS, then
writes a Markdown report summarising accuracy vs sparsity.

Usage
-----
    python main.py

Outputs
-------
    ./data/            — CIFAR-10 dataset (auto-downloaded)
    ./plots/           — gate distribution histograms for each lambda
    ./report.md        — Markdown report with results table
"""

import os
import time

import torch
from train    import train, get_loaders
from evaluate import evaluate
from config   import LAMBDAS, REPORT_PATH, PLOT_DIR, DEVICE


def write_report(results: list[dict]) -> None:
    """Generate report.md with explanation + results table."""
    lines = [
        "# Self-Pruning Neural Network — Results Report",
        "",
        "## Why L1 on sigmoid gates encourages sparsity",
        "",
        "The sparsity loss is the **L1 norm** of all gate values:",
        "",
        "```",
        "SparsityLoss = Σ sigmoid(gate_score_i)   for all i",
        "```",
        "",
        "The L1 norm has a **constant gradient of ±1** regardless of how small",
        "the value is. This means the optimiser keeps pushing gate values all",
        "the way to zero, rather than merely shrinking them (as L2 would do).",
        "",
        "In contrast, the classification loss pushes gate_scores **positive**",
        "for weights that genuinely improve accuracy. The result is a tug-of-war",
        "that resolves into a binary outcome for each gate:",
        "",
        "- **Gate → 0** : weight not useful enough to justify the sparsity cost → pruned.",
        "- **Gate → 1** : weight is critical for accuracy → preserved.",
        "",
        "This is why the gate distribution shows a bimodal spike at 0 and a",
        "cluster near 1, with almost nothing in between.",
        "",
        "---",
        "",
        "## Results",
        "",
        "| Lambda | Test Accuracy (%) | Sparsity Level (%) |",
        "|--------|------------------|--------------------|",
    ]

    for r in results:
        lines.append(
            f"| `{r['lambda']:.0e}` | {r['accuracy']:.2f} | {r['sparsity']:.2f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Gate distribution analysis",
        "",
        "See `./plots/gates_lambda_*.png` for gate histograms.",
        "",
        "A **successful** run shows:",
        "- A large spike at gate ≈ 0 (pruned connections).",
        "- A secondary cluster near gate ≈ 0.8–1.0 (surviving connections).",
        "- Very little in the middle — the L1 loss forces a binary commit.",
        "",
        "Increasing λ shifts more weight into the spike at 0 (higher sparsity),",
        "at the cost of reduced test accuracy.",
        "",
        "---",
        "",
        f"*Generated automatically by main.py — device: {DEVICE}*",
        "",
    ]

    os.makedirs(os.path.dirname(REPORT_PATH) or ".", exist_ok=True)
    with open(REPORT_PATH, "w",encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nReport saved → {REPORT_PATH}")


def main():
    print("\n" + "=" * 55)
    print("  Self-Pruning Neural Network  |  CIFAR-10")
    print("=" * 55)
    print(f"  Device   : {DEVICE}")
    print(f"  Lambdas  : {LAMBDAS}")
    print(f"  Plot dir : {PLOT_DIR}\n")

    os.makedirs(PLOT_DIR, exist_ok=True)

    # Pre-build test loader (shared across all experiments)
    _, test_loader = get_loaders()

    results = []
    wall_times = []

    for lam in LAMBDAS:
        t0 = time.time()

        # Train
        model = train(lam=lam)

        # Evaluate
        result = evaluate(model, test_loader, lam, save_plot=True)
        results.append(result)

        elapsed = time.time() - t0
        wall_times.append(elapsed)
        print(f"  Wall time for λ={lam:.0e}: {elapsed/60:.1f} min\n")

        # Free GPU memory between runs
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary table
    print("\n" + "=" * 55)
    print("  FINAL RESULTS")
    print("=" * 55)
    print(f"  {'Lambda':<10}  {'Test Acc (%)':<15}  {'Sparsity (%)':<15}")
    print(f"  {'-'*8:<10}  {'-'*13:<15}  {'-'*13:<15}")
    for r in results:
        print(f"  {r['lambda']:<10.0e}  {r['accuracy']:<15.2f}  {r['sparsity']:<15.2f}")
    print()

    write_report(results)


if __name__ == "__main__":
    main()
