from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from torch import nn

from .vlm import VisionLanguageScorer
from algorithms.mineworld.model import MineWorldModel


class MPCPlanner:
    def __init__(
        self,
        world_model: MineWorldModel, 
        horizon: int = 256,
    ):
        self.world_model = world_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.horizon = horizon

    def bc_policy_step(self, context: torch.Tensor) -> torch.Tensor:
        """
        Build the initial per-step categorical distribution by querying the BC policy
        on the final context frame. 
        """
        horizon = self.cfg.horizon
        try:
            logits = self.policy(context)
        except Exception as exc:  # pragma: no cover - policy is user provided.
            warnings.warn(f"BC policy failed during planner warm start: {exc}")
            logits = None

        probs = torch.softmax(logits / max(self.cfg.temperature, 1e-6), dim=-1)
        probs = probs.unsqueeze(1).expand(-1, horizon, -1).contiguous()
        return probs

    def world_model_step(self, action: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Roll out next frame based on action from bc policy step
        """
        return self.world_model.rollout(context, action)
    
    def rollout(self, context: torch.Tensor, horizon: int) -> torch.Tensor:
        """
        Roll out the context for the given horizon steps
        """
        actions = torch.zeros(horizon, self.action_dim)
        contexts = torch.zeros(horizon, *context.shape)
        for i in range(horizon):
            action = self.bc_policy_step(context)
            actions[i] = action
            context = self.world_model_step(action, context)
            contexts[i] = context
        return actions, contexts

    def n_rollouts(self, context: torch.Tensor, n: int, horizon: int) -> torch.Tensor:
        """
        Do n rollouts of the context for the given horizon
        """
        trajectories = []
        for i in range(n):
            actions = torch.zeros(n, self.action_dim)
            contexts = torch.zeros(n, *context.shape)
            for j in range(horizon):
                action = self.bc_policy_step(context)
                actions[i] = action
                context = self.world_model_step(action, context)
                contexts[i] = context
            trajectories.append((actions, contexts))
        return trajectories