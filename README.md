# Self-Pruning Neural Network

A feedforward neural network that learns to prune its own weights during training using learnable gates and L1 sparsity regularisation, trained on CIFAR-10.

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run all three lambda experiments (downloads CIFAR-10 automatically)
python main.py
```

Outputs:
- `./plots/gates_lambda_*.png` — gate distribution histograms
- `./report.md` — results table and analysis

## Project structure

```
self_pruning_network/
├── config.py          # All hyperparameters (edit λ, epochs, etc. here)
├── prunable_layer.py  # PrunableLinear — the core gated weight module
├── model.py           # SelfPruningNet — feedforward net using PrunableLinear
├── loss.py            # total_loss = CrossEntropy + λ × SparsityLoss
├── train.py           # Training loop with Adam + cosine LR scheduler
├── evaluate.py        # Sparsity %, test accuracy, gate histogram plot
├── main.py            # Entry point: runs all λ experiments + writes report
└── requirements.txt
```

## How it works

1. **PrunableLinear** — every weight `w` has a paired `gate_score` parameter.
   In the forward pass: `gate = sigmoid(gate_score)`, then `output = (w × gate) @ x + b`.
   A gate of 0 silences the weight completely.

2. **Sparsity loss** — `SparsityLoss = Σ gate_i` (L1 norm of all gates).
   Added to the cross-entropy loss: `Total = CE + λ × SparsityLoss`.
   The L1 constant gradient drives most gates all the way to 0.

3. **Training** — Adam updates both weights and gate_scores simultaneously.
   No separate pruning step is needed; the network learns to prune itself.

## Expected results

| Lambda | Test Accuracy | Sparsity |
|--------|--------------|----------|
| 1e-4   | ~53%         | ~25–35%  |
| 1e-3   | ~47%         | ~55–70%  |
| 1e-2   | ~38%         | ~80–95%  |
