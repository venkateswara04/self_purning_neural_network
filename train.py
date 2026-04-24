"""
train.py — Training loop for the self-pruning network (Part 3).

Builds CIFAR-10 data loaders, instantiates the model, and runs Adam
optimisation for a given lambda value.  The sparsity loss is applied on
every step so gate_scores are updated alongside the regular weights.
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model  import SelfPruningNet
from loss   import total_loss
from config import DEVICE, DATA_DIR, EPOCHS, LR, BATCH_SIZE


def get_loaders(batch_size: int = BATCH_SIZE):
    """
    Return (train_loader, test_loader) for CIFAR-10.

    Images are normalised to [-1, 1] per channel.  Training set uses
    basic augmentation (random horizontal flip + random crop) to improve
    generalisation.
    """
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),   # CIFAR-10 channel means
                             (0.2470, 0.2435, 0.2616)),  # CIFAR-10 channel stds
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    train_ds = datasets.CIFAR10(root=DATA_DIR, train=True,
                                download=True, transform=train_transform)
    test_ds  = datasets.CIFAR10(root=DATA_DIR, train=False,
                                download=True, transform=test_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, test_loader



def train(
    lam:        float  = 1e-3,
    epochs:     int    = EPOCHS,
    lr:         float  = LR,
    batch_size: int    = BATCH_SIZE,
    verbose:    bool   = True,
) -> SelfPruningNet:
    """
    Train a SelfPruningNet with the given sparsity coefficient λ.

    The optimiser updates BOTH the network weights and the gate_scores on
    every step.  The sparsity loss gradually drives most gate_scores negative
    so their sigmoid values (gates) approach 0 — pruning those connections.

    Parameters
    ----------
    lam        : sparsity regularisation coefficient
    epochs     : number of full passes over the training data
    lr         : initial Adam learning rate
    batch_size : mini-batch size
    verbose    : print per-epoch summary if True

    Returns
    -------
    SelfPruningNet — trained model on DEVICE
    """
    print(f"\n{'='*55}")
    print(f"  Training  |  λ = {lam:.0e}  |  {epochs} epochs  |  {DEVICE}")
    print(f"{'='*55}")

    train_loader, _ = get_loaders(batch_size)

    model = SelfPruningNet().to(DEVICE)

    param_info = model.count_parameters()
    print(f"  Parameters: {param_info['weight+bias']:,} (weights+bias) + "
          f"{param_info['gate_scores']:,} (gate_scores) = "
          f"{param_info['total']:,} total\n")

    # Adam optimiser — all parameters (weights, biases, gate_scores) included
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)

    # Cosine annealing scheduler: LR decays smoothly from lr → 0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )

    for epoch in range(1, epochs + 1):
        model.train()

        running_loss = 0.0
        running_ce   = 0.0
        running_sp   = 0.0
        correct      = 0
        total_samples = 0

        for images, labels in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            logits = model(images)

            # Total loss = CrossEntropy + λ × SparsityLoss
            loss, ce_val, sp_val = total_loss(logits, labels, model, lam)

            # Backprop — gradients flow through weights AND gate_scores
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_ce   += ce_val
            running_sp   += sp_val

            preds    = logits.argmax(dim=1)
            correct  += (preds == labels).sum().item()
            total_samples += labels.size(0)

        scheduler.step()

        if verbose and (epoch % 5 == 0 or epoch == 1):
            n_batches = len(train_loader)
            train_acc = correct / total_samples * 100
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Loss {running_loss/n_batches:.4f} | "
                  f"CE {running_ce/n_batches:.4f} | "
                  f"Sparsity {running_sp/n_batches:.1f} | "
                  f"Train acc {train_acc:.1f}%")

    print()
    return model
