from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
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
class BasePolicyConfig:
    type: Literal["mlp", "vpt_causal"] = "mlp"
    action_dim: int = 4


@dataclass
class PolicyConfig(BasePolicyConfig):
    type: Literal["mlp"] = "mlp"
    backbone: str = "resnet50"
    pretrained_backbone: bool = True
    backbone_trainable: bool = True
    backbone_dim: Optional[int] = None
    hidden_dim: int = 512
    head_layers: int = 4
    dropout: float = 0.0
    activation: Literal["gelu", "relu"] = "gelu"


@dataclass
class VPTPolicyConfig(BasePolicyConfig):
    type: Literal["vpt_causal"] = "vpt_causal"
    backbone: str = "resnet50"
    pretrained_backbone: bool = True
    backbone_trainable: bool = True
    embed_dim: int = 768
    ffn_dim: int = 2048
    n_layers: int = 6
    n_heads: int = 8
    dropout: float = 0.1
    attn_dropout: float = 0.1
    mem_len: int = 64
    layer_norm_eps: float = 1e-5


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


class PositionalEmbedding(nn.Module):
    """
    Sinusoidal positional embeddings used for relative attention.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, positions: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        sinusoid = torch.einsum("i,j->ij", positions, self.inv_freq)
        pos_emb = torch.cat((sinusoid.sin(), sinusoid.cos()), dim=-1)
        if pos_emb.shape[-1] < self.dim:
            pad = pos_emb.new_zeros(pos_emb.shape[0], self.dim - pos_emb.shape[-1])
            pos_emb = torch.cat([pos_emb, pad], dim=-1)
        return pos_emb.to(dtype=dtype)


def _rel_shift(x: torch.Tensor) -> torch.Tensor:
    """
    Perform relative shift to align relative attention logits.
    """

    bsz, n_head, qlen, klen = x.size()
    zero_pad = x.new_zeros(bsz, n_head, qlen, 1)
    x_padded = torch.cat([zero_pad, x], dim=3)
    x_padded = x_padded.view(bsz, n_head, klen + 1, qlen)
    x = x_padded[:, :, 1:].view(bsz, n_head, qlen, klen)
    return x


class RelMultiHeadAttention(nn.Module):
    """
    Multi-head attention with Transformer-XL-style relative positional encoding and memory.
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        attn_dropout: float = 0.1,
        resid_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by n_head for relative attention.")
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.scale = 1.0 / math.sqrt(self.d_head)

        head_dim = n_head * self.d_head
        self.q_proj = nn.Linear(d_model, head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, head_dim, bias=False)
        self.r_proj = nn.Linear(d_model, head_dim, bias=False)
        self.out_proj = nn.Linear(head_dim, d_model, bias=False)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.resid_dropout = nn.Dropout(resid_dropout)

        self.r_r_bias = nn.Parameter(torch.zeros(n_head, self.d_head))
        self.r_w_bias = nn.Parameter(torch.zeros(n_head, self.d_head))

    def forward(
        self,
        x: torch.Tensor,
        pos_emb: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        mem: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, qlen, _ = x.size()
        if mem is not None:
            k_input = torch.cat([mem, x], dim=1)
        else:
            k_input = x
        klen = k_input.size(1)

        w_head_q = self.q_proj(x).view(bsz, qlen, self.n_head, self.d_head).transpose(1, 2)
        w_head_k = self.k_proj(k_input).view(bsz, klen, self.n_head, self.d_head).transpose(1, 2)
        w_head_v = self.v_proj(k_input).view(bsz, klen, self.n_head, self.d_head).transpose(1, 2)
        r_head_k = self.r_proj(pos_emb).view(klen, self.n_head, self.d_head).permute(1, 0, 2)

        rw_head_q = w_head_q + self.r_w_bias.unsqueeze(0).unsqueeze(2)
        AC = torch.einsum("bnqd,bnkd->bnqk", rw_head_q, w_head_k)

        rr_head_q = w_head_q + self.r_r_bias.unsqueeze(0).unsqueeze(2)
        BD = torch.einsum("bnqd,nkd->bnqk", rr_head_q, r_head_k)
        BD = _rel_shift(BD)

        attn_score = (AC + BD) * self.scale

        if attn_mask is not None:
            attn_score = attn_score.masked_fill(attn_mask, float("-inf"))

        attn_prob = torch.softmax(attn_score, dim=-1)
        attn_prob = self.attn_dropout(attn_prob)

        attn_vec = torch.einsum("bnqk,bnkd->bnqd", attn_prob, w_head_v)
        attn_vec = attn_vec.transpose(1, 2).contiguous().view(bsz, qlen, self.n_head * self.d_head)
        attn_out = self.out_proj(attn_vec)
        attn_out = self.resid_dropout(attn_out)
        return attn_out


class PositionwiseFFN(nn.Module):
    """
    Feed-forward network with GELU activation.
    """

    def __init__(self, d_model: int, d_inner: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_inner)
        self.fc2 = nn.Linear(d_inner, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerXLBlock(nn.Module):
    """
    Residual Transformer block with causal masking and memory.
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        d_inner: int,
        dropout: float,
        attn_dropout: float,
        mem_len: int,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.attention = RelMultiHeadAttention(
            d_model=d_model,
            n_head=n_head,
            attn_dropout=attn_dropout,
            resid_dropout=dropout,
        )
        self.ffn = PositionwiseFFN(d_model=d_model, d_inner=d_inner, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.mem_len = mem_len

    def forward(
        self,
        x: torch.Tensor,
        pos_emb: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        mem: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_input = self.norm1(x)
        attn_out = self.attention(attn_input, pos_emb, attn_mask=attn_mask, mem=mem)
        x = x + attn_out
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out
        new_mem = self._update_memory(mem, x)
        return x, new_mem

    def _update_memory(
        self,
        mem: Optional[torch.Tensor],
        hidden: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if self.mem_len <= 0:
            return None
        new_mem = hidden.detach()
        if new_mem.size(1) > self.mem_len:
            new_mem = new_mem[:, -self.mem_len :, :]
        if mem is None:
            return new_mem
        cat = torch.cat([mem, new_mem], dim=1)
        if cat.size(1) > self.mem_len:
            cat = cat[:, -self.mem_len :, :]
        return cat


class VPTCausalPolicy(nn.Module):
    """
    Transformer-XL-style behavioral cloning policy inspired by VPT.
    """

    def __init__(self, cfg: VPTPolicyConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = SimpleVisionBackbone(
            name=cfg.backbone,
            pretrained=cfg.pretrained_backbone,
            trainable=cfg.backbone_trainable,
            out_dim=cfg.embed_dim,
        )
        self.input_norm = nn.LayerNorm(cfg.embed_dim, eps=cfg.layer_norm_eps)
        self.dropout = nn.Dropout(cfg.dropout)
        self.positional_embedding = PositionalEmbedding(cfg.embed_dim)
        self.layers = nn.ModuleList(
            [
                TransformerXLBlock(
                    d_model=cfg.embed_dim,
                    n_head=cfg.n_heads,
                    d_inner=cfg.ffn_dim,
                    dropout=cfg.dropout,
                    attn_dropout=cfg.attn_dropout,
                    mem_len=cfg.mem_len,
                    layer_norm_eps=cfg.layer_norm_eps,
                )
                for _ in range(cfg.n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(cfg.embed_dim, eps=cfg.layer_norm_eps)
        self.policy = nn.Linear(cfg.embed_dim, cfg.action_dim)

    def _causal_mask(self, qlen: int, mlen: int, device: torch.device) -> Optional[torch.Tensor]:
        if qlen <= 0:
            return None
        future_mask = torch.triu(
            torch.ones(qlen, qlen, device=device, dtype=torch.bool),
            diagonal=1,
        )
        if mlen > 0:
            mem_mask = torch.zeros(qlen, mlen, device=device, dtype=torch.bool)
            mask = torch.cat([mem_mask, future_mask], dim=1)
        else:
            mask = future_mask
        return mask.unsqueeze(0).unsqueeze(1)

    def _encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, c, h, w = frames.shape
        flat_frames = frames.reshape(bsz * seq_len, c, h, w)
        features = self.encoder(flat_frames)
        return features.reshape(bsz, seq_len, -1)

    def forward(
        self,
        frames: torch.Tensor,
        mems: Optional[Sequence[Optional[torch.Tensor]]] = None,
        return_mems: bool = False,
        return_all_tokens: bool = False,
    ):
        if frames.dim() == 4:
            frames = frames.unsqueeze(1)
        if frames.dim() != 5:
            raise ValueError("Expected input of shape (B, T, C, H, W) or (B, C, H, W).")

        hidden = self._encode_frames(frames)
        hidden = self.dropout(self.input_norm(hidden))

        num_layers = len(self.layers)
        if mems is None:
            mems = [None] * num_layers
        elif len(mems) != num_layers:
            raise ValueError(f"Expected {num_layers} memory tensors, received {len(mems)}.")

        mlen = 0
        if mems and mems[0] is not None:
            mlen = mems[0].size(1)

        seq_len = hidden.size(1)
        klen = mlen + seq_len
        device = hidden.device
        pos_seq = torch.arange(klen - 1, -1, -1, device=device, dtype=hidden.dtype)
        pos_emb = self.positional_embedding(pos_seq, hidden.dtype)
        attn_mask = self._causal_mask(seq_len, mlen, device)

        new_mems: List[Optional[torch.Tensor]] = []
        output = hidden
        for layer, mem in zip(self.layers, mems):
            output, new_mem = layer(output, pos_emb, attn_mask=attn_mask, mem=mem)
            new_mems.append(new_mem)

        logits_all = self.policy(self.final_norm(output))
        logits = logits_all if return_all_tokens else logits_all[:, -1]

        if return_mems:
            return logits, new_mems
        return logits


def parse_policy_config(cfg_dict: dict) -> BasePolicyConfig:
    """
    Instantiate the appropriate policy config dataclass from a config dictionary.
    """

    policy_type = cfg_dict.get("type", "mlp")
    if policy_type == "vpt_causal":
        return VPTPolicyConfig(**cfg_dict)
    if policy_type == "mlp":
        return PolicyConfig(**cfg_dict)
    raise ValueError(f"Unsupported policy type '{policy_type}'.")


def build_policy(cfg: BasePolicyConfig) -> nn.Module:
    """
    Build a policy module from a parsed configuration.
    """

    if isinstance(cfg, VPTPolicyConfig):
        return VPTCausalPolicy(cfg)
    if isinstance(cfg, PolicyConfig):
        return FoundationBCPolicy(cfg)
    raise TypeError(f"Unsupported policy configuration type: {type(cfg).__name__}")
