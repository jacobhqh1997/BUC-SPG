"""

Supply an initialized eight-class model and preprocessed tensor loaders.
Frozen image embeddings: features [B, 9, D]; valid_mask [B, 9];
relative_positions [9, 2]; target [B] (integer tissue class).
Split subjects BEFORE extracting training/validation patches. Keep all patches
from the same subject in one partition. Any balancing uses training data only.
No data paths, identifiers, checkpoint export, logging services or CLI included.
Mixed precision and infrastructure are intentionally omitted for readability.
"""

from copy import deepcopy
import torch
import torch.nn.functional as F


@torch.no_grad()
def validation_macro_f1(model, loader, positions, device):
    model.eval()
    confusion = torch.zeros(8, 8, device=device)
    for batch in loader:
        target = batch['target'].to(device).long()
        logits = model(batch['features'].to(device),
                       batch['valid_mask'].to(device), positions)
        predicted = logits.argmax(-1)
        confusion += torch.bincount(target * 8 + predicted, minlength=64).reshape(8, 8)
    if confusion.sum() == 0:
        raise ValueError('Empty validation loader')
    denominator = confusion.sum(0) + confusion.sum(1)
    return float((2 * confusion.diag() / denominator.clamp_min(1)).mean())


def train(model, train_loader, validation_loader, relative_positions, *,
          device, epochs, learning_rate, weight_decay, label_smoothing,
          patience, gradient_clip=1.0):
    """Select the in-memory checkpoint using held-out macro-F1, not test data.

    Hyperparameters are caller supplied; no claim of a frozen experiment config.
    """
    model.to(device)
    positions = relative_positions.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-6)
    best_score, stale, best_state = -float('inf'), 0, None
    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch['features'].to(device),
                           batch['valid_mask'].to(device), positions)
            loss = F.cross_entropy(logits, batch['target'].to(device).long(),
                                   label_smoothing=label_smoothing)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
        score = validation_macro_f1(model, validation_loader, positions, device)
        scheduler.step(score)
        if score > best_score:
            best_score, stale = score, 0
            best_state = deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model
