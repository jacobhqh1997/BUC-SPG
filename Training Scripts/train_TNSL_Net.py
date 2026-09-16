"""
Inputs: frozen features [B,9,D], valid_mask [B,9], positions [9,2],
soft_target [B,4] (nonnegative rows summing to one), sample_weight [B].
Only the centre soft target is supervised; neighbouring tokens provide context.
No tissue-prior input, pseudotime head or survival endpoint is used.
Training weight = normalized inverse-sqrt subject patch count * reliability;
weights are prepared upstream using only the applicable training partition.
Final fitting visits all labelled centres once per epoch without replacement.
This is deployment fitting, NOT independent validation. Model selection must
precede it, using subject-disjoint validation and training-only target fitting.
No identifiers, paths, data loading, logging, saving or CLI are included.
"""

import torch
import torch.nn.functional as F


def weighted_soft_loss(logits, targets, weights, brier_lambda=0.10):
    soft_ce = -(targets * F.log_softmax(logits, dim=-1)).sum(-1)
    brier = ((logits.softmax(-1) - targets) ** 2).sum(-1)
    denominator = weights.sum().clamp_min(1e-8)
    return (weights * (soft_ce + brier_lambda * brier)).sum() / denominator


def train(model, train_loader, relative_positions, *, device, epochs,
          learning_rate=2e-4, weight_decay=3e-4, brier_lambda=0.10,
          gradient_clip=1.0):
    """Fixed-epoch final fit; loader supplies tensors, not private metadata.

    Numerical optimization only; mixed precision and export are omitted.
    """
    model.to(device)
    positions = relative_positions.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch['features'].to(device), positions,
                           batch['valid_mask'].to(device))
            loss = weighted_soft_loss(logits, batch['soft_target'].to(device),
                                      batch['sample_weight'].to(device), brier_lambda)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
        scheduler.step()
    return model
