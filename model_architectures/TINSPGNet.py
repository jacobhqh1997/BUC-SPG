"""TINSPGNet core architecture. Source: Networks/TINSPGNet.py.
Includes native morphology pooling, tissue/niche prototypes, semantic attention,
niche-query/tissue-key-value cross-attention, spatial encoders and text gating.
Inputs: UNI features [N,1024], coordinates [N,2], tissue map [H,W,8],
niche map [H,W,4], precomputed ClinicalBERT embedding [1,768].
UNI and ClinicalBERT feature extraction are upstream, not included here.
Output: hazards, survival probabilities, predicted interval and explanations.
Full model is the default; original structural ablation switches are retained.
No training, checkpoint loading or execution entry point is included.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MicroAttentionPool(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, 1),
        )
        self.output = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
        )

    def forward(self, features):
        scores = self.attention(features).transpose(0, 1)
        weights = F.softmax(scores, dim=1)
        pooled = torch.mm(weights, features)
        return self.output(pooled), weights


class PriorGuidedPrototypeAggregator(nn.Module):
    """Aggregate one UNI morphology prototype for every prior class."""

    def __init__(self, n_classes, feature_dim=256):
        super().__init__()
        self.n_classes = n_classes
        self.visual_transform = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
        )
        self.visual_gate = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, n_classes),
        )

    def forward(self, features, prior_at_patch):
        # features:       [N, D]
        # prior_at_patch: [N, K]
        visual = self.visual_transform(features)
        visual_gate = torch.sigmoid(self.visual_gate(features))
        unnormalized = prior_at_patch * visual_gate
        denominator = unnormalized.sum(dim=0, keepdim=True)
        attention = unnormalized / (denominator + 1e-6)
        prototypes = torch.mm(attention.transpose(0, 1), visual).unsqueeze(0)
        presence = (prior_at_patch.sum(dim=0, keepdim=True) > 1e-6).float()
        prototypes = prototypes * presence.unsqueeze(-1)
        return prototypes, attention.transpose(0, 1).unsqueeze(0), presence


class SemanticTokenEncoder(nn.Module):
    def __init__(self, n_classes, feature_dim=256, num_heads=4):
        super().__init__()
        self.class_embedding = nn.Parameter(torch.randn(1, n_classes, feature_dim) * 0.02)
        self.attention = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(feature_dim * 4, feature_dim),
        )
        self.norm2 = nn.LayerNorm(feature_dim)

    def forward(self, tokens, presence):
        tokens = tokens + self.class_embedding * presence.unsqueeze(-1)
        attended, attention = self.attention(
            tokens, tokens, tokens, average_attn_weights=True
        )
        tokens = self.norm1(tokens + attended)
        tokens = self.norm2(tokens + self.ffn(tokens))
        tokens = tokens * presence.unsqueeze(-1)
        return tokens, attention


class NicheToTissueCrossAttention(nn.Module):
    def __init__(self, feature_dim=256, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(feature_dim * 4, feature_dim),
        )
        self.norm2 = nn.LayerNorm(feature_dim)

    def forward(self, niche_tokens, tissue_tokens, niche_presence):
        context, attention = self.attention(
            query=niche_tokens,
            key=tissue_tokens,
            value=tissue_tokens,
            average_attn_weights=True,
        )
        niche_tokens = self.norm1(niche_tokens + context)
        niche_tokens = self.norm2(niche_tokens + self.ffn(niche_tokens))
        niche_tokens = niche_tokens * niche_presence.unsqueeze(-1)
        return niche_tokens, attention


class MultiScaleSpatialEncoder(nn.Module):
    def __init__(self, in_channels, feature_dim=256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
        )
        self.local = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.ReLU(),
        )
        self.wide = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=4, dilation=4, bias=False),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(64, 64, 1, bias=False),
            nn.ReLU(),
        )
        self.average_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.output = nn.Sequential(
            nn.Linear(128, feature_dim),
            nn.ReLU(),
        )

    def forward(self, prior_map):
        features = self.stem(prior_map)
        features = self.fusion(torch.cat([self.local(features), self.wide(features)], dim=1))
        average = self.average_pool(features).flatten(1)
        maximum = self.max_pool(features).flatten(1)
        return self.output(torch.cat([average, maximum], dim=1))


class PriorBranchPool(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, 1),
        )
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
        )

    def forward(self, tokens, spatial_context, presence):
        scores = self.score(tokens).squeeze(-1)
        scores = scores.masked_fill(presence <= 0, -1e4)
        weights = F.softmax(scores, dim=1) * presence
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)
        pooled = torch.sum(tokens * weights.unsqueeze(-1), dim=1)
        representation = self.fusion(torch.cat([pooled, spatial_context], dim=1))
        branch_available = (presence.sum(dim=1, keepdim=True) > 0).float()
        representation = representation * branch_available
        return representation, weights, branch_available


class ClinicalTextEncoder(nn.Module):
    def __init__(self, text_dim=768, feature_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, 512),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(512, feature_dim),
            nn.LayerNorm(feature_dim),
        )

    def forward(self, text):
        return self.encoder(text)


class TextConditionedModalityFusion(nn.Module):
    """Use diagnosis text to choose micro/tissue/niche evidence."""

    def __init__(self, feature_dim=256):
        super().__init__()
        self.modality_score = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
        )
        self.no_text_score = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, 1),
        )
        self.text_interaction = nn.Sequential(
            nn.Linear(feature_dim * 4, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(feature_dim * 2, feature_dim),
        )
        self.text_gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.text_gate[2].bias, -2.0)
        self.output_norm = nn.LayerNorm(feature_dim)

    def forward(self, modality_tokens, availability, text_embedding=None):
        if text_embedding is None:
            logits = self.no_text_score(modality_tokens).squeeze(-1)
        else:
            text_tokens = text_embedding.unsqueeze(1).expand(-1, modality_tokens.shape[1], -1)
            logits = self.modality_score(
                torch.cat([modality_tokens, text_tokens], dim=-1)
            ).squeeze(-1)
        logits = logits.masked_fill(availability <= 0, -1e4)
        modality_weights = F.softmax(logits, dim=1)
        pathology = torch.sum(modality_tokens * modality_weights.unsqueeze(-1), dim=1)

        text_gate = pathology.new_zeros(pathology.shape)
        if text_embedding is None:
            fused = self.output_norm(pathology)
        else:
            interaction = self.text_interaction(
                torch.cat(
                    [
                        pathology,
                        text_embedding,
                        pathology * text_embedding,
                        torch.abs(pathology - text_embedding),
                    ],
                    dim=1,
                )
            )
            text_gate = self.text_gate(torch.cat([pathology, text_embedding], dim=1))
            fused = self.output_norm(pathology + text_gate * interaction)
        return fused, modality_weights, pathology, text_gate


class TINSPGNet(nn.Module):
    """Tissue-Niche Spatial-Prior-Guided Text-Conditioned survival network."""

    def __init__(
        self,
        n_time_bins=4,
        n_tissue_classes=8,
        n_niche_classes=4,
        text_dim=768,
        feature_dim=256,
        variant="full",
    ):
        super().__init__()
        valid_variants = {
            "full", "no_niche", "no_tissue", "no_text", "no_spatial",
            "micro_text", "micro_only",
        }
        if variant not in valid_variants:
            raise ValueError(f"Unknown TINSPGNet variant: {variant}")
        self.variant = variant
        self.use_tissue = variant not in {"no_tissue", "micro_text", "micro_only"}
        self.use_niche = variant not in {"no_niche", "micro_text", "micro_only"}
        self.use_text = variant not in {"no_text", "micro_only"}
        self.use_spatial = variant != "no_spatial"

        self.micro_projection = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, feature_dim),
            nn.GELU(),
            nn.Dropout(0.20),
        )
        self.micro_pool = MicroAttentionPool(feature_dim)

        if self.use_tissue:
            self.tissue_aggregator = PriorGuidedPrototypeAggregator(
                n_tissue_classes, feature_dim
            )
            self.tissue_token_encoder = SemanticTokenEncoder(
                n_tissue_classes, feature_dim
            )
            self.tissue_pool = PriorBranchPool(feature_dim)
            self.tissue_spatial_encoder = (
                MultiScaleSpatialEncoder(n_tissue_classes, feature_dim)
                if self.use_spatial else None
            )
        if self.use_niche:
            self.niche_aggregator = PriorGuidedPrototypeAggregator(
                n_niche_classes, feature_dim
            )
            self.niche_token_encoder = SemanticTokenEncoder(
                n_niche_classes, feature_dim
            )
            self.niche_pool = PriorBranchPool(feature_dim)
            self.niche_spatial_encoder = (
                MultiScaleSpatialEncoder(n_niche_classes, feature_dim)
                if self.use_spatial else None
            )
        self.niche_to_tissue = (
            NicheToTissueCrossAttention(feature_dim)
            if self.use_tissue and self.use_niche else None
        )

        self.text_encoder = (
            ClinicalTextEncoder(text_dim, feature_dim) if self.use_text else None
        )
        self.fusion = TextConditionedModalityFusion(feature_dim)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(feature_dim, n_time_bins),
        )

    @staticmethod
    def _remove_loader_batch(features, coords, tissue, niche, text):
        if features.dim() == 3:
            features = features.squeeze(0)
        if coords.dim() == 3:
            coords = coords.squeeze(0)
        if tissue.dim() == 4:
            tissue = tissue.squeeze(0)
        if niche.dim() == 4:
            niche = niche.squeeze(0)
        if text.dim() == 3:
            text = text.squeeze(0)
        if text.dim() == 1:
            text = text.unsqueeze(0)
        return features, coords, tissue, niche, text

    def forward(self, features, coords, tissue_probs, niche_probs, text_features):
        features, coords, tissue_probs, niche_probs, text_features = self._remove_loader_batch(
            features, coords, tissue_probs, niche_probs, text_features
        )
        features = features.float()
        coords = coords.long()
        tissue_probs = torch.nan_to_num(tissue_probs.float(), nan=0.0)
        niche_probs = torch.nan_to_num(niche_probs.float(), nan=0.0)
        text_features = text_features.float()

        if tissue_probs.dim() != 3 or tissue_probs.shape[-1] != 8:
            raise ValueError(f"tissue_probs must be [H,W,8], got {tuple(tissue_probs.shape)}")
        if niche_probs.dim() != 3 or niche_probs.shape[-1] != 4:
            raise ValueError(f"niche_probs must be [H,W,4], got {tuple(niche_probs.shape)}")
        height, width = tissue_probs.shape[:2]
        if niche_probs.shape[:2] != (height, width):
            raise ValueError("Tissue and niche grids are not aligned")
        x, y = coords[:, 0], coords[:, 1]
        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        if not valid.all():
            features, coords = features[valid], coords[valid]
            x, y = coords[:, 0], coords[:, 1]
        if len(features) == 0:
            raise ValueError("No UNI features inside the prior grid")

        micro_features = self.micro_projection(features)
        micro_representation, micro_attention = self.micro_pool(micro_features)
        modality_tokens = [micro_representation]
        availability = [micro_representation.new_ones((1, 1))]

        occupancy = tissue_probs.new_zeros(height, width)
        occupancy[y, x] = 1.0
        tissue_map = tissue_probs * occupancy.unsqueeze(-1)
        niche_map = niche_probs * occupancy.unsqueeze(-1)

        explanations = {"micro_attention": micro_attention}
        tissue_tokens = None
        tissue_presence = None

        if self.use_tissue:
            tissue_at_patch = tissue_map[y, x]
            tissue_tokens, tissue_patch_attention, tissue_presence = self.tissue_aggregator(
                micro_features, tissue_at_patch
            )
            tissue_tokens, tissue_self_attention = self.tissue_token_encoder(
                tissue_tokens, tissue_presence
            )
            if self.use_spatial:
                tissue_spatial = self.tissue_spatial_encoder(
                    tissue_map.permute(2, 0, 1).unsqueeze(0)
                )
            else:
                tissue_spatial = micro_representation.new_zeros(micro_representation.shape)
            tissue_representation, tissue_token_weights, tissue_available = self.tissue_pool(
                tissue_tokens, tissue_spatial, tissue_presence
            )
            modality_tokens.append(tissue_representation)
            availability.append(tissue_available)
            explanations.update({
                "tissue_patch_attention": tissue_patch_attention,
                "tissue_self_attention": tissue_self_attention,
                "tissue_token_weights": tissue_token_weights,
                "tissue_presence": tissue_presence,
            })

        if self.use_niche:
            niche_at_patch = niche_map[y, x]
            niche_tokens, niche_patch_attention, niche_presence = self.niche_aggregator(
                micro_features, niche_at_patch
            )
            niche_tokens, niche_self_attention = self.niche_token_encoder(
                niche_tokens, niche_presence
            )
            niche_tissue_attention = None
            if tissue_tokens is not None:
                niche_tokens, niche_tissue_attention = self.niche_to_tissue(
                    niche_tokens, tissue_tokens, niche_presence
                )
            if self.use_spatial:
                niche_spatial = self.niche_spatial_encoder(
                    niche_map.permute(2, 0, 1).unsqueeze(0)
                )
            else:
                niche_spatial = micro_representation.new_zeros(micro_representation.shape)
            niche_representation, niche_token_weights, niche_available = self.niche_pool(
                niche_tokens, niche_spatial, niche_presence
            )
            modality_tokens.append(niche_representation)
            availability.append(niche_available)
            explanations.update({
                "niche_patch_attention": niche_patch_attention,
                "niche_self_attention": niche_self_attention,
                "niche_to_tissue_attention": niche_tissue_attention,
                "niche_token_weights": niche_token_weights,
                "niche_presence": niche_presence,
            })

        modality_tokens = torch.stack(modality_tokens, dim=1)
        availability = torch.cat(availability, dim=1)
        text_embedding = self.text_encoder(text_features) if self.use_text else None
        fused, modality_weights, pathology, text_gate = self.fusion(
            modality_tokens, availability, text_embedding
        )

        logits = self.classifier(fused)
        hazards = torch.sigmoid(logits)
        survs = torch.cumprod(1.0 - hazards, dim=1)
        Y = F.softmax(logits, dim=1)
        explanations.update({
            "modality_weights": modality_weights,
            "modality_availability": availability,
            "text_gate": text_gate,
            "pathology_representation": pathology,
            "fused_representation": fused,
        })
        return hazards, survs, Y, explanations
