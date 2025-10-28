from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from omegaconf import DictConfig, OmegaConf

from algorithms.dfot import DFoTVideo
from utils.ckpt_utils import download_pretrained


@dataclass
class WorldModelConfig:
    checkpoint: str = "pretrained:DFoT_MCRAFT.ckpt"
    algorithm: DictConfig | None = None
    device: str = "cuda"
    context_frames: int = 8


class DFoTWorldModel:
    """
    Thin wrapper that loads a pretrained DFoT checkpoint for closed-loop rollouts.
    """

    def __init__(self, cfg: WorldModelConfig):
        self.cfg = cfg
        if cfg.algorithm is None:
            raise ValueError("WorldModelConfig.algorithm must be provided.")
        algo_cfg = DictConfig(OmegaConf.to_container(cfg.algorithm, resolve=True))
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.model = DFoTVideo(algo_cfg)
        ckpt_path = self._resolve_checkpoint(cfg.checkpoint)
        state = torch.load(ckpt_path, map_location="cpu")
        state_dict = state.get("state_dict", state)
        if any("diffusion_model._orig_mod." in key for key in state_dict):
            state_dict = {
                key.replace("diffusion_model._orig_mod.", "diffusion_model."): value
                for key, value in state_dict.items()
            }
        load_status = self.model.load_state_dict(state_dict, strict=False)
        missing_keys = [
            key
            for key in load_status.missing_keys
            if not key.startswith("metrics_")
            and key not in {"data_mean", "data_std"}
        ]
        if missing_keys:
            raise RuntimeError(f"DFoT checkpoint missing keys: {sorted(missing_keys)}")
        self.model.eval().to(self.device)
        if self.model.is_latent_diffusion and not self.model.is_latent_online:
            self.model._load_vae()

    def _resolve_checkpoint(self, checkpoint: str) -> Path:
        if checkpoint.startswith("pretrained:") or checkpoint.startswith("full:"):
            return Path(download_pretrained(checkpoint))
        return Path(checkpoint)

    @torch.no_grad()
    def rollout(
        self,
        context: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the DFoT world model conditioned on context and action tokens.
        Args:
            context: (B, T_ctx, C, H, W) tensor in [0, 1].
            actions: (B, T_h, A) action conditioning (one-hot).
        Returns:
            predicted frames for the rollout window (B, T_h, C, H, W).
        """
        device = self.device
        self.model = self.model.to(device)
        context = context.to(device)
        horizon = actions.shape[1]
        total = context.shape[1] + horizon
        if total > self.model.max_tokens:
            raise ValueError(
                f"Requested rollout length {total} exceeds DFoT max tokens {self.model.max_tokens}."
            )
        pad = torch.zeros(
            (context.shape[0], horizon, *context.shape[2:]), device=device
        )
        xs = torch.cat([context, pad], dim=1)
        xs = self.model._normalize_x(xs)
        if xs.shape[1] < self.model.max_tokens:
            pad_tokens = self.model.max_tokens - xs.shape[1]
            xs = torch.cat(
                [xs, xs[:, -1:].repeat(1, pad_tokens, 1, 1, 1)], dim=1
            )
        cond = actions.to(device)
        if cond.shape[2] != self.model.external_cond_dim:
            raise ValueError(
                f"Action dim {cond.shape[2]} != DFoT external_cond_dim {self.model.external_cond_dim}"
            )
        if cond.shape[1] < self.model.max_tokens:
            pad_len = self.model.max_tokens - cond.shape[1]
            cond = torch.cat([cond, cond[:, -1:].repeat(1, pad_len, 1)], dim=1)
        elif cond.shape[1] > self.model.max_tokens:
            cond = cond[:, : self.model.max_tokens]
        preds = self.model._predict_videos(xs, conditions=cond)
        preds = self.model._unnormalize_x(preds)
        if self.model.is_latent_diffusion:
            preds = self.model._decode(preds)
        return preds[:, context.shape[1] : context.shape[1] + horizon]
