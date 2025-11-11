from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from gym3.types import DictType

from algorithms.foundation_dagger.vpt_model.action_head import make_action_head
from algorithms.foundation_dagger.vpt_model.action_mapping import CameraHierarchicalMapping


def _camel_case(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_"))

def _resolve_resnet(name: str, pretrained: bool) -> nn.Module:
    normalized = name.lower()
    weights = None
    if not hasattr(models, normalized):
        raise ValueError(f"Unsupported torchvision backbone '{name}'.")
    if pretrained:
        weight_attr_candidates = (
            f"{normalized}_Weights",
            f"{_camel_case(normalized)}_Weights",
        )
        for attr in weight_attr_candidates:
            weight_enum = getattr(models, attr, None)
            if weight_enum is not None:
                weights = weight_enum.DEFAULT
                break
    backbone = getattr(models, normalized)(weights=weights)
    return backbone


class SimpleVisionBackbone(nn.Module):
    """
    Thin wrapper around torchvision classification heads that emits pooled features.
    """

    def __init__(
        self,
        name: str = "resnet50",
        pretrained: bool = True,
        trainable: bool = True,
        out_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        backbone = _resolve_resnet(name, pretrained)
        feature_dim = backbone.fc.in_features
        modules = list(backbone.children())[:-1]
        self.encoder = nn.Sequential(*modules)
        if not trainable:
            for param in self.encoder.parameters():
                param.requires_grad_(False)
        self.feature_dim = feature_dim
        self.project = None
        if out_dim is not None and out_dim != feature_dim:
            self.project = nn.Linear(feature_dim, out_dim)
            nn.init.trunc_normal_(self.project.weight, std=0.02)
            nn.init.zeros_(self.project.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(images)
        feats = feats.flatten(1)
        if self.project is not None:
            feats = self.project(feats)
        return feats

class TemporalEncoder(nn.Module):
    """Turns per-frame features into per-step temporal features."""
    def __init__(self, dim: int, n_layers: int = 2, n_heads: int = 4, seq_dropout: float = 0.0):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=n_heads, batch_first=True)
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.drop = nn.Dropout(seq_dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, T, D]
        x = self.enc(x, src_key_padding_mask=attn_mask)  # attn_mask: True for PAD
        return self.drop(x)

@dataclass
class ActionHeadConfig:
    """
    Configuration shim so Hydra configs that referenced the legacy head still load.

    Only ``camera_bins`` and ``temperature`` are consulted by the current VPT-style head;
    the remaining fields are kept for backwards compatibility with older configs.
    """

    buttons_classes: int                     # legacy no-op
    camera_bins: int = 11                    # discretization bins for the mapper
    use_camera_gate: bool = True             # retained for compatibility (unused)
    buttons_multilabel: bool = False         # retained for compatibility (unused)
    temperature: float = 1.0                 # softmax temperature passed to the VPT head

def _default_action_head_cfg() -> "ActionHeadConfig":
    # Import lazily to avoid adding a hard dependency on gym3 during lightweight usage.
    buttons_classes = 80
    try:
        from algorithms.foundation_dagger.vpt_model.action_mapping import CameraHierarchicalMapping

        buttons_classes = len(CameraHierarchicalMapping.BUTTONS_COMBINATIONS)
    except Exception:
        pass
    return ActionHeadConfig(buttons_classes=buttons_classes, camera_bins=11, use_camera_gate=True)


@dataclass
class PolicyConfig:
    backbone: str = "resnet50"
    pretrained_backbone: bool = True
    backbone_trainable: bool = True
    backbone_dim: Optional[int] = None
    hidden_dim: int = 512
    head_layers: int = 4
    dropout: float = 0.0
    activation: Literal["gelu", "relu"] = "gelu"

    # temporal_layers: int = 2
    # temporal_heads: int = 4
    # temporal_dropout: float = 0.0
    action: ActionHeadConfig = field(default_factory=_default_action_head_cfg)


class FoundationBCPolicy(nn.Module):
    """
    Simple encoder-policy head used for BC + DAgger.
    """

    def __init__(self, cfg: PolicyConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = SimpleVisionBackbone(
            name=cfg.backbone,
            pretrained=cfg.pretrained_backbone,
            trainable=cfg.backbone_trainable,
            out_dim=cfg.backbone_dim,
        )
        feat_dim = cfg.backbone_dim or self.encoder.feature_dim
        # self.temporal = TemporalEncoder(
        #     dim=feat_dim,
        #     n_layers=cfg.temporal_layers,
        #     n_heads=cfg.temporal_heads,
        #     seq_dropout=cfg.temporal_dropout,
        # )
        mapper_bins = cfg.action.camera_bins
        self.action_mapper = CameraHierarchicalMapping(n_camera_bins=mapper_bins)
        action_space = DictType(**self.action_mapper.get_action_space_update())
        self.action_head = make_action_head(
            action_space,
            feat_dim,
            temperature=getattr(cfg.action, "temperature", 1.0),
        )
        self.esc_head = nn.Linear(feat_dim, 2)

    def forward(self, frames: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor]:
        """
        frames: [B, T, C, H, W]  (T can be 1 for single frame)
        pad_mask: [B, T] with True where PAD (ignored by attention & loss)
        """
        B, T, C, H, W = frames.shape
        x = frames.reshape(B * T, C, H, W)
        feats = self.encoder(x)              # [B*T, D]
        feats = feats.view(B, T, -1)         # [B, T, D]
        # feats = self.temporal(feats, attn_mask=pad_mask)
        logits = self.action_head(feats)     # action-space dict of log-probs
        logits["esc"] = F.log_softmax(self.esc_head(feats), dim=-1)
        return logits
