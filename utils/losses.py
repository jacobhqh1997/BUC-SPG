"""Anonymous training objectives for CTP-Net, TNSL-Net and TINSPGNet.

Tensor-only utilities: no data loading, identifiers, logging or network access.
"""

import torch
from torch import nn
from torch.nn import functional as F


class TissueClassificationLoss(nn.Module):
    """CTP-Net: hard-label cross entropy with optional label smoothing.

    logits: [B,C]; targets: [B], integer class indices.
    """

    def __init__(self, label_smoothing=0.0):
        super().__init__()
        if not 0 <= label_smoothing <= 1:
            raise ValueError('label_smoothing must lie in [0,1]')
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        return F.cross_entropy(logits, targets.long(),
                               label_smoothing=self.label_smoothing)


class WeightedSoftLabelLoss(nn.Module):
    """TNSL-Net: weighted soft cross entropy + lambda * Brier score.

    logits/targets: [B,C]; target rows sum to one; weights: [B].
    Weights are normalized by their sum within each batch. Upstream weights may
    combine inverse-sqrt subject patch counts and label reliability; no subject
    identifiers are needed here. All-zero weights produce a zero gradient.
    """

    def __init__(self, brier_lambda=0.10):
        super().__init__()
        if brier_lambda < 0:
            raise ValueError('brier_lambda must be nonnegative')
        self.brier_lambda = brier_lambda

    def forward(self, logits, targets, weights):
        if logits.ndim != 2 or logits.shape != targets.shape:
            raise ValueError('logits and targets must have matching [B,C] shapes')
        if weights.shape != logits.shape[:1]:
            raise ValueError('weights must have shape [B]')
        if not torch.isfinite(targets).all() or (targets < 0).any():
            raise ValueError('targets must be finite and nonnegative')
        if not torch.allclose(targets.sum(-1), torch.ones_like(targets[:, 0]), atol=1e-5):
            raise ValueError('target rows must sum to one')
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError('weights must be finite and nonnegative')
        logits, targets, weights = logits.float(), targets.float(), weights.float()
        ce = -(targets * F.log_softmax(logits, dim=-1)).sum(-1)
        brier = (logits.softmax(-1) - targets).square().sum(-1)
        return (weights * (ce + self.brier_lambda*brier)).sum() / weights.sum().clamp_min(1e-8)


class DiscreteTimeSurvivalLoss(nn.Module):
    """TINSPGNet: censor-weighted discrete-time NLL, NOT Cox loss.

    hazards/survival: [B,K]; time_bin: [B], zero-based integer interval;
    censored: [B], 1=right-censored, 0=observed event.
    survival must equal cumprod(1-hazards) along the interval axis.
    Event loss: -log(S before interval)-log(hazard in interval).
    Censor loss: -log(S through interval), multiplied by (1-alpha).
    alpha=0 yields ordinary NLL; alpha=0.15 matches the training objective.
    Fit interval boundaries using training data only.
    """

    def __init__(self, alpha=0.15, eps=1e-7):
        super().__init__()
        if not 0 <= alpha <= 1 or not 0 < eps < 1:
            raise ValueError('Invalid alpha or eps')
        self.alpha, self.eps = alpha, eps

    def forward(self, hazards, survival, time_bin, censored):
        if hazards.ndim != 2 or hazards.shape != survival.shape:
            raise ValueError('hazards and survival must have matching [B,K] shapes')
        if time_bin.numel() != hazards.shape[0] or censored.numel() != hazards.shape[0]:
            raise ValueError('Endpoint count does not match batch size')
        if not torch.isfinite(time_bin).all() or (time_bin != time_bin.long()).any():
            raise ValueError('time_bin must contain finite integer indices')
        index = time_bin.reshape(-1, 1).long()
        if (index < 0).any() or (index >= hazards.shape[1]).any():
            raise ValueError('time_bin is outside the available intervals')
        censor = censored.reshape(-1, 1).float()
        if not ((censor == 0) | (censor == 1)).all():
            raise ValueError('censored must be binary')
        for values in (hazards, survival):
            if not torch.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                raise ValueError('Probabilities must be finite and lie in [0,1]')
        hazards, survival = hazards.float(), survival.float()
        padded = torch.cat([torch.ones_like(censor), survival], dim=1)
        event = -(1-censor) * (
            padded.gather(1, index).clamp_min(self.eps).log()
            + hazards.gather(1, index).clamp_min(self.eps).log())
        censor_loss = -censor * padded.gather(1, index+1).clamp_min(self.eps).log()
        return (event + (1-self.alpha)*censor_loss).mean()


def classifier_l1(model):
    """Unscaled L1 penalty on classifier parameters, including biases.

    Add regularization_weight * classifier_l1(model) to the survival objective.
    """
    parameters = list(model.classifier.parameters())
    if not parameters:
        raise ValueError('Classifier has no parameters')
    return torch.stack([p.abs().sum() for p in parameters]).sum()
