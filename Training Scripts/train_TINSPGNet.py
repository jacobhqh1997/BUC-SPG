"""Anonymous TINSPGNet training logic; not a reproduction package.

One variable-length slide bag per step: frozen morphology features, coordinates,
tissue probabilities, niche probabilities and a precomputed text embedding.
The supplied model returns hazards, survival probabilities and auxiliary outputs.
time_bin is a zero-based discrete interval; censored=1 means right-censored,
censored=0 means observed event. Fit interval boundaries on training data only.
Use subject-disjoint partitions and fit/select all upstream components without
held-out outcomes. Text must be de-identified before upstream embedding.
No actual text, identifiers, institution names, paths, endpoints or weights are
embedded here. No external logging, filesystem output or executable CLI.
Selection/evaluation infrastructure and mixed precision are intentionally omitted;
epochs are fixed upstream, never selected using the external test cohort.
"""

import torch


def survival_nll(hazards, survival, time_bin, censored, alpha=0.15, eps=1e-7):
    """Project's censor-weighted discrete-time NLL; NOT Cox partial likelihood.

    Uncensored: -log(S before interval)-log(hazard in interval).
    Censored: -log(S through interval), weighted by (1-alpha).
    """
    index = time_bin.reshape(-1, 1).long()
    censor = censored.reshape(-1, 1).float()
    hazards, survival = hazards.float(), survival.float()
    padded = torch.cat([torch.ones_like(censor), survival], dim=1)
    uncensored_loss = -(1-censor) * (
        padded.gather(1, index).clamp_min(eps).log()
        + hazards.gather(1, index).clamp_min(eps).log())
    censored_loss = -censor * padded.gather(1, index+1).clamp_min(eps).log()
    return (uncensored_loss + (1-alpha)*censored_loss).mean()


def train(model, train_loader, *, device, epochs, learning_rate,
          weight_decay=1e-3, regularization_weight=3e-4, alpha=0.15):
    """Adam + classifier L1 penalty + training-loss learning-rate scheduling.

    train_loader supplies one bag and its endpoint tensors at each step.
    Returns the final model; no private checkpoint metadata is collected.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 betas=(0.9, 0.999), weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    for _ in range(epochs):
        model.train()
        total, count = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            hazards, survival, _, _ = model(
                batch['features'].to(device), batch['coordinates'].to(device),
                batch['tissue_probabilities'].to(device),
                batch['niche_probabilities'].to(device),
                batch['text_embedding'].to(device))
            nll = survival_nll(hazards, survival, batch['time_bin'].to(device),
                               batch['censored'].to(device), alpha=alpha)
            penalty = sum(parameter.abs().sum() for parameter in model.classifier.parameters())
            loss = nll + regularization_weight * penalty
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach())
            count += 1
        if count == 0:
            raise ValueError('Empty training loader')
        scheduler.step(total/count)
    return model
