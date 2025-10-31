from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from omegaconf import DictConfig, OmegaConf

from .policy import BasePolicyConfig, build_policy, parse_policy_config


def _policy_cfg_from_dict(cfg: DictConfig | Dict) -> BasePolicyConfig:
    data = OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else cfg
    return parse_policy_config(data)


@dataclass
class OptimizationConfig:
    lr: float = 1e-4
    weight_decay: float = 0.0
    warmup_steps: int = 0
    max_steps: Optional[int] = None


class FoundationDaggerModule(pl.LightningModule):
    """
    LightningModule for supervised BC within the DAgger loop.
    """

    def __init__(
        self,
        cfg: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        self.cfg = cfg
        policy_cfg = _policy_cfg_from_dict(cfg.policy)
        self.policy = build_policy(policy_cfg)
        self.criterion = nn.CrossEntropyLoss()
        self.optim_cfg = OptimizationConfig(**OmegaConf.to_container(cfg.optim, resolve=True))
        self.action_dim = policy_cfg.action_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.policy(images)

    def step(self, batch: Dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        images = batch["observations"]
        actions = batch["actions"]
        logits = self.forward(images)
        loss = self.criterion(logits, actions)
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            acc = (preds == actions).float().mean()
        self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/acc", acc, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def training_step(self, batch: Dict[str, torch.Tensor], _: int) -> torch.Tensor:
        return self.step(batch, "training")

    def validation_step(self, batch: Dict[str, torch.Tensor], _: int) -> torch.Tensor:
        return self.step(batch, "validation")

    def test_step(self, batch: Dict[str, torch.Tensor], _: int) -> torch.Tensor:
        return self.step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.optim_cfg.lr,
            weight_decay=self.optim_cfg.weight_decay,
        )
        if self.optim_cfg.warmup_steps <= 0:
            return optimizer

        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=self.optim_cfg.warmup_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
