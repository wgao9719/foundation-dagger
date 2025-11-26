"""
Policy architecture for MineRL ObtainDiamond/ObtainIronPickaxe.

Key differences from the VPT-style FoundationBCPolicy:
- Takes inventory + equipped item as additional observations
- Uses factored multi-head action output instead of joint button combinations
- Optimized for 64x64 input resolution
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def _camel_case(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_"))


def _resolve_resnet(name: str, pretrained: bool) -> nn.Module:
    """Load a torchvision ResNet model."""
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


class SmallCNN(nn.Module):
    """
    Lightweight CNN backbone for 64x64 inputs.
    
    Much faster than ResNet for this resolution and sufficient for the task.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        out_dim: int = 256,
    ):
        super().__init__()
        # 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(base_channels * 4, base_channels * 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(base_channels * 8, out_dim)
        self.feature_dim = out_dim
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class ResNetBackbone(nn.Module):
    """
    Thin wrapper around torchvision ResNet that emits pooled features.
    """
    
    def __init__(
        self,
        name: str = "resnet18",
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
        self.feature_dim = out_dim or feature_dim
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


class InventoryEncoder(nn.Module):
    """
    Encodes inventory counts and equipped item into a fixed-size vector.
    """
    
    def __init__(
        self,
        num_inventory_items: int = 18,
        num_equipped_types: int = 3,
        equipped_embed_dim: int = 16,
        hidden_dim: int = 64,
        out_dim: int = 64,
    ):
        super().__init__()
        self.num_inventory_items = num_inventory_items
        
        # Embed equipped item type
        self.equipped_embed = nn.Embedding(num_equipped_types, equipped_embed_dim)
        
        # Process inventory counts (log-scale + linear)
        # Input: inventory counts [B, T, 18] + equipped embed [B, T, 16] = 34
        input_dim = num_inventory_items + equipped_embed_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )
        self.out_dim = out_dim
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
    
    def forward(
        self,
        inventory: torch.Tensor,
        equipped_type: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            inventory: [B, T, num_items] int32 item counts
            equipped_type: [B, T] int64 equipped item type index
            
        Returns:
            [B, T, out_dim] encoded observation features
        """
        # Log-scale inventory counts for better gradient flow
        inv_float = torch.log1p(inventory.float())  # [B, T, 18]
        
        # Embed equipped type
        eq_embed = self.equipped_embed(equipped_type)  # [B, T, embed_dim]
        
        # Concatenate and encode
        combined = torch.cat([inv_float, eq_embed], dim=-1)  # [B, T, 34]
        return self.mlp(combined)  # [B, T, out_dim]


class MultiHeadActionOutput(nn.Module):
    """
    Multi-head action output for MineRL factored action space.
    
    Produces logits for:
    - 8 binary buttons (each 2-class)
    - 2D camera (binned, or continuous)
    - 5 categorical actions (place, equip, craft, nearby_craft, nearby_smelt)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_camera_bins: int = 11,
        num_place_classes: int = 7,
        num_equip_classes: int = 6,
        num_craft_classes: int = 5,
        num_nearby_craft_classes: int = 7,
        num_nearby_smelt_classes: int = 3,
        use_continuous_camera: bool = False,
    ):
        super().__init__()
        self.n_camera_bins = n_camera_bins
        self.use_continuous_camera = use_continuous_camera
        
        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # Binary button heads (8 separate 2-class heads)
        # Note: use "fwd" instead of "forward" to avoid conflict with nn.Module.forward
        self.button_names = [
            "fwd", "left", "back", "right", 
            "jump", "sneak", "sprint", "attack"
        ]
        self.button_heads = nn.ModuleDict({
            name: nn.Linear(hidden_dim, 2) for name in self.button_names
        })
        
        # Camera head
        if use_continuous_camera:
            # Predict mean for 2D Gaussian (pitch, yaw)
            self.camera_head = nn.Linear(hidden_dim, 2)
        else:
            # Predict bins for each axis independently
            self.camera_pitch_head = nn.Linear(hidden_dim, n_camera_bins)
            self.camera_yaw_head = nn.Linear(hidden_dim, n_camera_bins)
        
        # Categorical action heads
        self.place_head = nn.Linear(hidden_dim, num_place_classes)
        self.equip_head = nn.Linear(hidden_dim, num_equip_classes)
        self.craft_head = nn.Linear(hidden_dim, num_craft_classes)
        self.nearby_craft_head = nn.Linear(hidden_dim, num_nearby_craft_classes)
        self.nearby_smelt_head = nn.Linear(hidden_dim, num_nearby_smelt_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: [B, T, D] or [B, D] input features
            
        Returns:
            Dict of logits for each action head
        """
        x = self.trunk(features)
        
        outputs = {}
        
        # Binary buttons - log probabilities
        for name in self.button_names:
            outputs[f"button_{name}"] = F.log_softmax(
                self.button_heads[name](x), dim=-1
            )
        
        # Camera
        if self.use_continuous_camera:
            outputs["camera_continuous"] = self.camera_head(x)
        else:
            outputs["camera_pitch"] = F.log_softmax(self.camera_pitch_head(x), dim=-1)
            outputs["camera_yaw"] = F.log_softmax(self.camera_yaw_head(x), dim=-1)
        
        # Categorical actions - log probabilities
        outputs["place"] = F.log_softmax(self.place_head(x), dim=-1)
        outputs["equip"] = F.log_softmax(self.equip_head(x), dim=-1)
        outputs["craft"] = F.log_softmax(self.craft_head(x), dim=-1)
        outputs["nearby_craft"] = F.log_softmax(self.nearby_craft_head(x), dim=-1)
        outputs["nearby_smelt"] = F.log_softmax(self.nearby_smelt_head(x), dim=-1)
        
        return outputs


@dataclass
class MineRLPolicyConfig:
    """Configuration for MineRLPolicy."""
    
    # Vision backbone
    backbone: Literal["small_cnn", "resnet18", "resnet34", "resnet50"] = "small_cnn"
    pretrained_backbone: bool = True
    backbone_trainable: bool = True
    vision_dim: int = 256  # Output dim of vision encoder
    
    # Inventory encoder
    inventory_hidden_dim: int = 64
    inventory_out_dim: int = 64
    
    # Action head
    action_hidden_dim: int = 256
    n_camera_bins: int = 11
    use_continuous_camera: bool = False
    
    # Architecture
    use_temporal_encoder: bool = False
    temporal_layers: int = 2
    temporal_heads: int = 4
    dropout: float = 0.1


class MineRLPolicy(nn.Module):
    """
    Policy network for MineRL ObtainDiamond/ObtainIronPickaxe.
    
    Architecture:
        frames [B, T, H, W, C] -> vision encoder -> [B, T, vision_dim]
        inventory [B, T, 18] + equipped [B, T] -> inventory encoder -> [B, T, inv_dim]
        concat -> [B, T, vision_dim + inv_dim]
        (optional) temporal encoder -> [B, T, D]
        action heads -> multi-head logits
    """
    
    def __init__(
        self,
        cfg: MineRLPolicyConfig,
        action_space_info: Optional[Dict[str, int]] = None,
        num_inventory_items: int = 18,
        num_equipped_types: int = 3,
    ):
        super().__init__()
        self.cfg = cfg
        
        # Default action space info
        if action_space_info is None:
            action_space_info = {
                "num_place_classes": 7,
                "num_equip_classes": 6,
                "num_craft_classes": 5,
                "num_nearby_craft_classes": 7,
                "num_nearby_smelt_classes": 3,
            }
        
        # Vision encoder
        if cfg.backbone == "small_cnn":
            self.vision_encoder = SmallCNN(
                in_channels=3,
                base_channels=32,
                out_dim=cfg.vision_dim,
            )
        else:
            self.vision_encoder = ResNetBackbone(
                name=cfg.backbone,
                pretrained=cfg.pretrained_backbone,
                trainable=cfg.backbone_trainable,
                out_dim=cfg.vision_dim,
            )
        
        # Inventory encoder
        self.inventory_encoder = InventoryEncoder(
            num_inventory_items=num_inventory_items,
            num_equipped_types=num_equipped_types,
            hidden_dim=cfg.inventory_hidden_dim,
            out_dim=cfg.inventory_out_dim,
        )
        
        # Combined feature dimension
        combined_dim = cfg.vision_dim + cfg.inventory_out_dim
        
        # Optional temporal encoder
        self.temporal_encoder = None
        if cfg.use_temporal_encoder:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=combined_dim,
                nhead=cfg.temporal_heads,
                dim_feedforward=combined_dim * 4,
                dropout=cfg.dropout,
                batch_first=True,
            )
            self.temporal_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=cfg.temporal_layers,
            )
        
        # Action heads
        self.action_head = MultiHeadActionOutput(
            input_dim=combined_dim,
            hidden_dim=cfg.action_hidden_dim,
            n_camera_bins=cfg.n_camera_bins,
            num_place_classes=action_space_info["num_place_classes"],
            num_equip_classes=action_space_info["num_equip_classes"],
            num_craft_classes=action_space_info["num_craft_classes"],
            num_nearby_craft_classes=action_space_info["num_nearby_craft_classes"],
            num_nearby_smelt_classes=action_space_info["num_nearby_smelt_classes"],
            use_continuous_camera=cfg.use_continuous_camera,
        )
    
    def forward(
        self,
        frames: torch.Tensor,
        inventory: torch.Tensor,
        equipped_type: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            frames: [B, T, H, W, C] uint8 or [B, T, C, H, W] float (pre-normalized)
            inventory: [B, T, num_items] int32 item counts
            equipped_type: [B, T] int64 equipped item type
            
        Returns:
            Dict of action logits
        """
        B, T = frames.shape[:2]
        
        # Handle frame format - expect [B, T, H, W, C] uint8, convert to [B*T, C, H, W] float
        if frames.dim() == 5 and frames.shape[-1] == 3:
            # [B, T, H, W, C] -> [B, T, C, H, W]
            frames = frames.permute(0, 1, 4, 2, 3)
        
        # Normalize if uint8
        if frames.dtype == torch.uint8:
            frames = frames.float() / 255.0
        
        # Reshape for vision encoder
        frames_flat = frames.reshape(B * T, *frames.shape[2:])  # [B*T, C, H, W]
        
        # Vision encoding
        vision_feats = self.vision_encoder(frames_flat)  # [B*T, vision_dim]
        vision_feats = vision_feats.view(B, T, -1)  # [B, T, vision_dim]
        
        # Inventory encoding
        inv_feats = self.inventory_encoder(inventory, equipped_type)  # [B, T, inv_dim]
        
        # Combine features
        combined = torch.cat([vision_feats, inv_feats], dim=-1)  # [B, T, combined_dim]
        
        # Optional temporal encoding
        if self.temporal_encoder is not None:
            combined = self.temporal_encoder(combined)
        
        # Action prediction
        logits = self.action_head(combined)
        
        return logits
    
    def predict_action(
        self,
        frames: torch.Tensor,
        inventory: torch.Tensor,
        equipped_type: torch.Tensor,
        deterministic: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict actions for inference.
        
        Returns dict with predicted action indices/values.
        """
        with torch.no_grad():
            logits = self.forward(frames, inventory, equipped_type)
        
        # Take last timestep
        actions = {}
        
        # Binary buttons (map "fwd" back to "forward" for external API)
        button_name_map = {"fwd": "forward"}  # internal -> external
        for name in self.action_head.button_names:
            log_probs = logits[f"button_{name}"][:, -1]  # [B, 2]
            external_name = button_name_map.get(name, name)
            if deterministic:
                actions[external_name] = log_probs.argmax(dim=-1)
            else:
                actions[external_name] = torch.distributions.Categorical(logits=log_probs).sample()
        
        # Camera
        if self.cfg.use_continuous_camera:
            actions["camera"] = logits["camera_continuous"][:, -1]  # [B, 2]
        else:
            pitch_probs = logits["camera_pitch"][:, -1]
            yaw_probs = logits["camera_yaw"][:, -1]
            if deterministic:
                actions["camera_pitch"] = pitch_probs.argmax(dim=-1)
                actions["camera_yaw"] = yaw_probs.argmax(dim=-1)
            else:
                actions["camera_pitch"] = torch.distributions.Categorical(logits=pitch_probs).sample()
                actions["camera_yaw"] = torch.distributions.Categorical(logits=yaw_probs).sample()
        
        # Categorical actions
        for name in ["place", "equip", "craft", "nearby_craft", "nearby_smelt"]:
            log_probs = logits[name][:, -1]
            if deterministic:
                actions[name] = log_probs.argmax(dim=-1)
            else:
                actions[name] = torch.distributions.Categorical(logits=log_probs).sample()
        
        return actions


if __name__ == "__main__":
    # Quick test
    cfg = MineRLPolicyConfig(
        backbone="small_cnn",
        vision_dim=256,
        use_temporal_encoder=False,
    )
    
    model = MineRLPolicy(cfg)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    B, T, H, W, C = 2, 8, 64, 64, 3
    frames = torch.randint(0, 256, (B, T, H, W, C), dtype=torch.uint8)
    inventory = torch.randint(0, 64, (B, T, 18), dtype=torch.int32)
    equipped_type = torch.randint(0, 3, (B, T), dtype=torch.int64)
    
    logits = model(frames, inventory, equipped_type)
    
    print("\nOutput shapes:")
    for key, value in logits.items():
        print(f"  {key}: {value.shape}")
    
    # Test inference
    actions = model.predict_action(frames, inventory, equipped_type)
    print("\nPredicted actions:")
    for key, value in actions.items():
        print(f"  {key}: {value.shape}")

