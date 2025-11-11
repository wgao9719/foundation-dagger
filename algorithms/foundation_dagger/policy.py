from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

import torch
import torch.nn as nn
from torchvision import models


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
    # buttons: either categorical over combos OR multi-label bits
    buttons_classes: int                     # e.g., num combo ids (recommended)
    camera_bins: int = 11                    # per-axis bins (e.g., 11)
    use_camera_gate: bool = True             # predict "camera active?" bit
    buttons_multilabel: bool = False         # set True if you want BCE per button bit

def _default_action_head_cfg() -> "ActionHeadConfig":
    # Import lazily to avoid adding a hard dependency on gym3 during lightweight usage.
    buttons_classes = 80
    try:
        from algorithms.foundation_dagger.vpt_model.action_mapping import CameraHierarchicalMapping

        buttons_classes = len(CameraHierarchicalMapping.BUTTONS_COMBINATIONS)
    except Exception:
        pass
    return ActionHeadConfig(buttons_classes=buttons_classes, camera_bins=11, use_camera_gate=True)

def _get_activation(name: Literal["gelu", "relu"]) -> Callable[[], nn.Module]:
    lowered = name.lower()
    if lowered == "gelu":
        return nn.GELU
    if lowered == "relu":
        return lambda: nn.ReLU(inplace=True)
    raise ValueError(f"Unsupported activation '{name}'")


class ActionHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        cfg: ActionHeadConfig,
        layers: int = 2,
        dropout: float = 0.0,
        activation: Literal["gelu", "relu"] = "gelu",
    ):
        super().__init__()
        act_factory = _get_activation(activation)
        # shared MLP trunk
        trunk = []
        d = input_dim
        for _ in range(layers - 1):
            trunk += [nn.Linear(d, hidden_dim), act_factory()]
            if dropout > 0:
                trunk += [nn.Dropout(dropout)]
            d = hidden_dim
        self.trunk = nn.Sequential(*trunk) if trunk else nn.Identity()

        # heads
        self.buttons = nn.Linear(d or input_dim, cfg.buttons_classes if not cfg.buttons_multilabel else cfg.buttons_classes)
        self.cam_x   = nn.Linear(d or input_dim, cfg.camera_bins)
        self.cam_y   = nn.Linear(d or input_dim, cfg.camera_bins)
        self.gate    = nn.Linear(d or input_dim, 2) if cfg.use_camera_gate else None
        self.cfg     = cfg

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        # h: [B, T, D]
        z = self.trunk(h)
        buttons_logits = self.buttons(z)
        cam_x_logits = self.cam_x(z)
        cam_y_logits = self.cam_y(z)
        camera_joint = (
            cam_x_logits.unsqueeze(-1) + cam_y_logits.unsqueeze(-2)
        ).reshape(z.shape[0], z.shape[1], -1)
        out = {
            "buttons": buttons_logits,    # [B, T, C] (or bits)
            "camera_x": cam_x_logits,     # [B, T, n_bins]
            "camera_y": cam_y_logits,     # [B, T, n_bins]
            "camera": camera_joint,       # [B, T, n_bins^2]
        }
        if self.cfg.use_camera_gate:
            out["camera_gate"] = self.gate(z)    # [B, T, 2]
        return out


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
        self.action_head = ActionHead(
            input_dim=feat_dim,
            hidden_dim=cfg.hidden_dim,
            cfg=cfg.action,
            layers=cfg.head_layers,
            dropout=cfg.dropout,
            activation=cfg.activation,
        )

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
        return self.action_head(feats)           # dict of logits [B, T, ·]
