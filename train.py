"""
train.py - Training loop with live sparsity % printed every epoch.

The key metric to watch is "Sparsity %" - it should GROW each epoch.
If it stays at 0% -> lambda too small.
If it jumps to 100% immediately -> lambda too large.
With correct settings: grows gradually from ~5% to 30-80% over 40 epochs.
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model          import SelfPruningNet
from loss           import total_loss
from prunable_layer import PrunableLinear
from config         import DEVICE, DATA_DIR, EPOCHS, LR, BATCH_SIZE, PRUNE_THRESHOLD


def _live_sparsity(model):
    """Count % of gates currently below PRUNE_THRESHOLD. No forward pass needed."""
    total  = 0
    pruned = 0
    for m in model.modules():
        if isinstance(m, PrunableLinear):
            gates   = torch.sigmoid(m.gate_scores).detach()
            pruned += (gates < PRUNE_THRESHOLD).sum().item()
            total  += gates.numel()
    return (pruned / total * 100) if total > 0 else 0.0


def _live_gate_mean(model):
    """Current mean gate value across all layers - should decrease over training."""
    vals = []
    for m in model.modules():
        if isinstance(m, PrunableLinear):
            vals.append(torch.sigmoid(m.gate_scores).detach().reshape(-1))
    return torch.cat(vals).mean().item() if vals else 0.0


def get_loaders(batch_size=BATCH_SIZE):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    train_ds = datasets.CIFAR10(DATA_DIR, train=True,  download=True, transform=train_transform)
    test_ds  = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    return train_loader, test_loader


def train(lam=1e-2, epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE, verbose=True):
    print(f"\n{'='*72}")
    print(f"  Training  |  lambda={lam:.0e}  |  {epochs} epochs  |  device={DEVICE}")
    print(f"{'='*72}")

    train_loader, _ = get_loaders(batch_size)
    model = SelfPruningNet().to(DEVICE)

    info = model.count_parameters()
    print(f"  Weights+bias: {info['weight+bias']:,}  |  "
          f"Gate scores: {info['gate_scores']:,}  |  "
          f"Total: {info['total']:,}")
    print(f"  Gate init mean: {_live_gate_mean(model):.4f}  "
          f"(target: ~0.18 = sigmoid(-1.5))")
    print(f"  Initial sparsity: {_live_sparsity(model):.2f}%\n")

    # Separate optimisers with different LR:
    # gate_scores need a HIGHER LR to move faster toward 0 or 1
    # weights need standard LR for accurate classification
    gate_params   = [m.gate_scores for m in model.modules()
                     if isinstance(m, PrunableLinear)]
    weight_params = [p for n, p in model.named_parameters()
                     if 'gate_scores' not in n]

    optimizer = torch.optim.Adam([
        {'params': weight_params, 'lr': lr},
        {'params': gate_params,   'lr': lr * 10},  # gates learn 10x faster
    ], weight_decay=0)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )

    # Print header
    print(f"  {'Ep':>4} {'CE':>8} {'GateMean':>10} {'Sparse%':>10} {'TrainAcc':>10}")
    print(f"  {'--':>4} {'--':>8} {'--------':>10} {'-------':>10} {'--------':>10}")

    for epoch in range(1, epochs + 1):
        model.train()

        sum_ce = 0.0
        correct = 0
        total_n = 0

        for images, labels in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            logits = model(images)
            loss, ce_val, _ = total_loss(logits, labels, model, lam)
            loss.backward()
            optimizer.step()

            sum_ce  += ce_val
            preds    = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total_n += labels.size(0)

        scheduler.step()

        # Compute real sparsity % and gate mean ONCE per epoch
        sp_pct    = _live_sparsity(model)
        gate_mean = _live_gate_mean(model)
        acc       = correct / total_n * 100
        ce_avg    = sum_ce / len(train_loader)

        if verbose:
            print(f"  {epoch:>4d} {ce_avg:>8.4f} {gate_mean:>10.4f} "
                  f"{sp_pct:>9.2f}% {acc:>9.2f}%")

    print()
    return model
