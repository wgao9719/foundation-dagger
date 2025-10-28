from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
from torchvision import models


def _resolve_resnet(name: str, pretrained: bool) -> nn.Module:
    name = name.lower()
    weights = None
    if pretrained:
        weight_enum = getattr(models, f"{name}_Weights", None)
        if weight_enum is not None:
            weights = weight_enum.DEFAULT
    if not hasattr(models, name):
        raise ValueError(f"Unsupported torchvision backbone '{name}'.")
    backbone = getattr(models, name)(weights=weights)
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


class PolicyHead(nn.Module):
    """
    Lightweight MLP with GELU activations for policy logits.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        action_dim: int,
        layers: int = 4,
        dropout: float = 0.0,
        activation: Literal["gelu", "relu"] = "gelu",
    ) -> None:
        super().__init__()
        activation_layer = nn.GELU if activation == "gelu" else nn.ReLU
        mlp: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(layers - 1):
            mlp.append(nn.Linear(in_dim, hidden_dim))
            mlp.append(activation_layer())
            if dropout > 0:
                mlp.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        mlp.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*mlp)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


@dataclass
class PolicyConfig:
    backbone: str = "resnet50"
    pretrained_backbone: bool = True
    backbone_trainable: bool = True
    backbone_dim: Optional[int] = None
    hidden_dim: int = 512
    action_dim: int = 4
    head_layers: int = 4
    dropout: float = 0.0
    activation: Literal["gelu", "relu"] = "gelu"


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
        head_input = cfg.backbone_dim or self.encoder.feature_dim
        self.policy = PolicyHead(
            input_dim=head_input,
            hidden_dim=cfg.hidden_dim,
            action_dim=cfg.action_dim,
            layers=cfg.head_layers,
            dropout=cfg.dropout,
            activation=cfg.activation,
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # expects (B, C, H, W)
        features = self.encoder(frames)
        return self.policy(features)
