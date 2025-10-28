from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from .vlm import VisionLanguageScorer
from .world_model import DFoTWorldModel


@dataclass
class MPCConfig:
    horizon: int = 8
    num_candidates: int = 64
    elite_frac: float = 0.125
    iterations: int = 4


class CEMPlanner:
    """
    Cross-Entropy Method planner leveraging DFoT rollouts + a VLM reward.
    """

    def __init__(
        self,
        world_model: DFoTWorldModel,
        vlm: VisionLanguageScorer,
        action_dim: int,
        cfg: MPCConfig,
    ) -> None:
        self.cfg = cfg
        self.world_model = world_model
        self.vlm = vlm
        self.action_dim = action_dim
        self.device = world_model.device

    def _one_hot(self, indices: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.one_hot(indices, num_classes=self.action_dim).float()

    def _eval_sequences(
        self,
        context: torch.Tensor,
        sequences: torch.Tensor,
    ) -> torch.Tensor:
        one_hot = self._one_hot(sequences).to(self.device)
        rollout = self.world_model.rollout(context, one_hot)
        terminal_frames = rollout[:, -1]
        return self.vlm.score_frames(terminal_frames)

    def plan(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return best action sequence + scores for a batch of contexts.
        """
        batch = context.shape[0]
        horizon = self.cfg.horizon
        probs = torch.full(
            (batch, horizon, self.action_dim),
            1 / self.action_dim,
            device=self.device,
        )

        for _ in range(self.cfg.iterations):
            samples = torch.distributions.Categorical(probs=probs).sample(
                (self.cfg.num_candidates,)
            )
            samples = samples.permute(1, 0, 2)  # (B, num_candidates, horizon)
            flat_samples = samples.reshape(batch * self.cfg.num_candidates, horizon)
            ctx = context.repeat_interleave(self.cfg.num_candidates, dim=0)
            scores = self._eval_sequences(ctx, flat_samples).view(
                batch, self.cfg.num_candidates
            )
            elite_k = max(1, int(self.cfg.num_candidates * self.cfg.elite_frac))
            elite_scores, elite_idx = torch.topk(scores, elite_k, dim=1)
            elite = torch.gather(
                samples,
                dim=1,
                index=elite_idx.unsqueeze(-1).expand(-1, -1, horizon),
            )
            probs.zero_()
            for action_id in range(self.action_dim):
                mask = elite == action_id
                probs[:, :, action_id] = mask.float().sum(dim=1) / elite_k
            probs = torch.clamp(probs, min=1e-4)
            probs = probs / probs.sum(dim=-1, keepdim=True)

        best_idx = probs.argmax(dim=-1)
        best_scores = self._eval_sequences(context, best_idx)
        return best_idx, best_scores
