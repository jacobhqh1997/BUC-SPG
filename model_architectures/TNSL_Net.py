"""TNSL-Net V2 core architecture.
Sources: Networks/tnsl_net.py (TNSLNetV2),
Networks/contextual_niche_distiller.py (LocalAttentionBlock),
Networks/uni_feature_niche_mlp.py (ResidualMLPBlock).
Input: frozen UNI [B,9,1024], relative positions [9,2] or [B,9,2], mask [B,9].
Output: centre-patch logits [B,4]; softmax gives ecosystem probabilities.
Centre residual MLP plus gated contextual correction; initial gate 0.12.
Defaults are constructor defaults, not a claim about every trained checkpoint.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalAttentionBlock(nn.Module):
    """Pre-normalized local self-attention block with exposed attention maps."""

    def __init__(
        self,
        dimension: int = 256,
        n_heads: int = 4,
        ffn_expansion: int = 2,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dimension)
        self.attention = nn.MultiheadAttention(
            embed_dim=dimension,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dimension)
        self.ffn = nn.Sequential(
            nn.Linear(dimension, dimension * ffn_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dimension * ffn_expansion, dimension),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self.norm1(tokens)
        attended, attention = self.attention(
            query=normalized,
            key=normalized,
            value=normalized,
            key_padding_mask=~valid_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        tokens = tokens + self.dropout1(attended)
        tokens = tokens + self.ffn(self.norm2(tokens))
        return tokens, attention


class ResidualMLPBlock(nn.Module):
    """Pre-normalized residual block for compact feature-space modeling."""

    def __init__(self, dimension: int, expansion: int = 2, dropout: float = 0.30):
        super().__init__()
        expanded = dimension * expansion
        self.block = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, expanded),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expanded, dimension),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TNSLNetV2(nn.Module):
    """Centre-preserving residual-context TNSL-Net.

    The centre UNI embedding is modelled by the compact residual MLP that was
    stable in the earlier single-patch experiments.  A position-aware local
    transformer learns only a gated *correction* to that centre representation.
    The gate starts small, so adding context cannot immediately overwrite the
    centre morphology.  There is still exactly one four-niche output head.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        n_niches: int = 4,
        dimension: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        centre_layers: int = 2,
        dropout: float = 0.20,
        neighbor_dropout: float = 0.20,
        initial_context_gate: float = 0.12,
    ) -> None:
        super().__init__()
        if not 0.0 <= neighbor_dropout < 1.0:
            raise ValueError("neighbor_dropout must be in [0, 1)")
        if not 0.0 < initial_context_gate < 1.0:
            raise ValueError("initial_context_gate must be in (0, 1)")
        self.input_dim = int(input_dim)
        self.n_niches = int(n_niches)
        self.dimension = int(dimension)
        self.neighbor_dropout = float(neighbor_dropout)

        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, dimension),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.centre_blocks = nn.ModuleList(
            [ResidualMLPBlock(dimension, expansion=2, dropout=dropout)
             for _ in range(centre_layers)]
        )
        self.position_encoder = nn.Sequential(
            nn.Linear(2, dimension // 2),
            nn.GELU(),
            nn.Linear(dimension // 2, dimension),
        )
        self.context_blocks = nn.ModuleList(
            [LocalAttentionBlock(dimension, n_heads, 2, dropout)
             for _ in range(n_layers)]
        )
        self.context_adapter = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, dimension * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dimension * 2, dimension),
            nn.Dropout(dropout),
        )
        self.context_gate = nn.Sequential(
            nn.LayerNorm(dimension * 2),
            nn.Linear(dimension * 2, dimension),
            nn.Sigmoid(),
        )
        gate_linear = self.context_gate[1]
        nn.init.zeros_(gate_linear.weight)
        gate_bias = torch.logit(torch.tensor(float(initial_context_gate))).item()
        nn.init.constant_(gate_linear.bias, gate_bias)

        self.output_norm = nn.LayerNorm(dimension)
        self.classifier = nn.Linear(dimension, n_niches)

    def _effective_mask(self, valid_mask: torch.Tensor) -> torch.Tensor:
        mask = valid_mask.bool()
        if self.training and self.neighbor_dropout > 0:
            keep = torch.rand(mask.shape, device=mask.device) >= self.neighbor_dropout
            keep[:, 0] = True
            mask = mask & keep
        return mask

    def forward(
        self,
        uni_features: torch.Tensor,
        relative_positions: torch.Tensor,
        valid_mask: torch.Tensor,
        return_explanations: bool = False,
    ):
        if uni_features.ndim != 3 or uni_features.shape[-1] != self.input_dim:
            raise ValueError("uni_features must be [B,9,input_dim]")
        if valid_mask.shape != uni_features.shape[:2] or not valid_mask[:, 0].all():
            raise ValueError("valid_mask must be [B,9] with a valid centre")
        if relative_positions.ndim == 2:
            relative_positions = relative_positions.unsqueeze(0).expand(
                uni_features.shape[0], -1, -1
            )
        if relative_positions.shape != (*uni_features.shape[:2], 2):
            raise ValueError("relative_positions must be [B,9,2] or [9,2]")

        effective_mask = self._effective_mask(valid_mask)
        projected = self.input_projection(self.input_norm(uni_features))
        centre = projected[:, 0]
        for block in self.centre_blocks:
            centre = block(centre)

        tokens = projected + self.position_encoder(
            relative_positions.to(projected.dtype)
        )
        tokens = tokens.masked_fill(~effective_mask.unsqueeze(-1), 0.0)
        attention_maps = []
        for block in self.context_blocks:
            tokens, attention = block(tokens, effective_mask)
            tokens = tokens.masked_fill(~effective_mask.unsqueeze(-1), 0.0)
            attention_maps.append(attention)

        contextual_centre = tokens[:, 0]
        context_delta = self.context_adapter(contextual_centre - projected[:, 0])
        gate = self.context_gate(torch.cat([centre, contextual_centre], dim=-1))
        fused = centre + gate * context_delta
        logits = self.classifier(self.output_norm(fused))
        if not return_explanations:
            return logits
        return {
            "logits": logits,
            "probabilities": logits.softmax(-1),
            "centre_attention": torch.stack(
                [x[:, :, 0, :] for x in attention_maps], dim=1
            ),
            "context_gate": gate,
            "effective_valid_mask": effective_mask,
        }
