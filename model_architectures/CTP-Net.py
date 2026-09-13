"""CTP-Net core architecture.
Source: Networks/contextual_niche_distiller.py; instantiated by
CTP_data_preparation/train_ctp_patient_separated.py.
Display class renamed CTPNet; original n_niches parameter means tissue classes here
and defaults to 8. Other structure and forward operations are preserved.
Input: frozen UNI features [B,9,1024], mask [B,9], positions [9,2].
Output: centre-patch logits [B,8]; softmax gives tissue probabilities.
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


class CTPNet(nn.Module):
    """Predict eight tissue-class logits for the centre patch.

    Input shapes
    ------------
    features:
        ``[batch, n_tokens, input_dim]``. Token 0 must be the centre patch.
    valid_mask:
        ``[batch, n_tokens]``. Missing neighbours are False; centre is True.
    relative_positions:
        ``[n_tokens, 2]`` in grid-step units, e.g. (-1, 0), (1, 1).

    The output is logits. Apply softmax to obtain the eight-dimensional
    tissue probability vector.  ``return_explanations=True`` also returns centre-to-neighbour
    attention and the centre/context gate.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        n_niches: int = 8,
        dimension: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.20,
        neighbor_dropout: float = 0.20,
    ) -> None:
        super().__init__()
        if not 0.0 <= neighbor_dropout < 1.0:
            raise ValueError("neighbor_dropout must be in [0, 1)")
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
        self.position_encoder = nn.Sequential(
            nn.Linear(2, dimension // 2),
            nn.GELU(),
            nn.Linear(dimension // 2, dimension),
        )
        self.blocks = nn.ModuleList(
            [
                LocalAttentionBlock(
                    dimension=dimension,
                    n_heads=n_heads,
                    ffn_expansion=2,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        # The gate decides how much the prediction should trust contextualized
        # morphology versus the original centre morphology, feature by feature.
        self.context_gate = nn.Sequential(
            nn.LayerNorm(dimension * 2),
            nn.Linear(dimension * 2, dimension),
            nn.Sigmoid(),
        )
        self.output_block = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, dimension * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dimension * 2, dimension),
            nn.Dropout(dropout),
        )
        self.output_norm = nn.LayerNorm(dimension)
        self.classifier = nn.Linear(dimension, n_niches)

    def _drop_neighbors(self, valid_mask: torch.Tensor) -> torch.Tensor:
        if not self.training or self.neighbor_dropout <= 0:
            return valid_mask
        random_keep = torch.rand(valid_mask.shape, device=valid_mask.device) >= self.neighbor_dropout
        random_keep[:, 0] = True
        return valid_mask & random_keep

    def forward(
        self,
        features: torch.Tensor,
        valid_mask: torch.Tensor,
        relative_positions: torch.Tensor,
        return_explanations: bool = False,
    ):
        if features.ndim != 3:
            raise ValueError(f"Expected [B,K,D] features, got {tuple(features.shape)}")
        if valid_mask.shape != features.shape[:2]:
            raise ValueError("valid_mask shape does not match feature tokens")
        if relative_positions.shape != (features.shape[1], 2):
            raise ValueError("relative_positions must have shape [K,2]")
        if not valid_mask[:, 0].all():
            raise ValueError("Centre token must always be valid")

        effective_mask = self._drop_neighbors(valid_mask.bool())
        projected = self.input_projection(self.input_norm(features))
        centre_morphology = projected[:, 0]
        position = self.position_encoder(relative_positions.to(projected.dtype))
        tokens = projected + position.unsqueeze(0)
        tokens = tokens.masked_fill(~effective_mask.unsqueeze(-1), 0.0)

        attention_maps = []
        for block in self.blocks:
            tokens, attention = block(tokens, effective_mask)
            tokens = tokens.masked_fill(~effective_mask.unsqueeze(-1), 0.0)
            attention_maps.append(attention)

        contextualized_centre = tokens[:, 0]
        gate = self.context_gate(
            torch.cat([centre_morphology, contextualized_centre], dim=-1)
        )
        fused = (1.0 - gate) * centre_morphology + gate * contextualized_centre
        fused = fused + self.output_block(fused)
        logits = self.classifier(self.output_norm(fused))

        if not return_explanations:
            return logits
        # [B, layers, heads, K], describing which neighbours informed the
        # centre token. Invalid/dropout neighbours receive zero attention.
        centre_attention = torch.stack(
            [attention[:, :, 0, :] for attention in attention_maps], dim=1
        )
        return {
            "logits": logits,
            "probabilities": logits.softmax(dim=-1),
            "centre_attention": centre_attention,
            "context_gate": gate,
            "effective_valid_mask": effective_mask,
        }
