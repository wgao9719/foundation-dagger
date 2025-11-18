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


def _to_numpy_image(frame: torch.Tensor) -> torch.Tensor:
    """
    Convert a BCHW tensor in [0, 1] to uint8 HWC for visualization.
    """
    frame = frame.detach()
    if frame.dim() == 4:
        frame = frame.squeeze(0)
    frame = frame.clamp(0, 1).mul(255).byte()
    if frame.size(0) in (1, 3):
        frame = frame.permute(1, 2, 0)
    return frame.cpu()


@dataclass
class RolloutVizConfig:
    """
    Optional GIF/MP4 logging for MineWorld rollouts produced during planning.
    """

    output_dir: Optional[str] = None
    max_videos: int = 4
    fps: int = 6
    prefix: str = "planner"

    def is_enabled(self) -> bool:
        return bool(self.output_dir)


@dataclass
class MPCConfig:
    horizon: int = 8
    num_candidates: int = 64
    elite_frac: float = 0.125
    iterations: int = 4
    temperature: float = 1.0
    policy_dropout: float = 0.05
    instructions: Optional[str] = None
    visualization: RolloutVizConfig = field(default_factory=RolloutVizConfig)


class CEMPlanner:
    """
    Cross-Entropy Method planner that samples action sequences from a BC policy,
    renders them with MineWorld, and ranks them with the Gemini VLM scorer.
    """

    def __init__(
        self,
        world_model: MineWorldModel,
        vlm: VisionLanguageScorer,
        action_dim: int,
        cfg: MPCConfig,
        policy: Optional[nn.Module] = None,
    ) -> None:
        self.cfg = cfg
        self.world_model = world_model
        self.vlm = vlm
        self.action_dim = action_dim
        self.device = getattr(world_model, "device", torch.device("cpu"))
        self.policy = policy or getattr(world_model, "policy", None)
        self._viz_step = 0
        self.latest_rollouts: List[List[torch.Tensor]] = []

    def _one_hot(self, indices: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.one_hot(indices, num_classes=self.action_dim).float()

    def _initial_probs(self, context: torch.Tensor) -> torch.Tensor:
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

    def _simulate_rollouts(
        self,
        context: torch.Tensor,
        sequences: torch.Tensor,
    ) -> torch.Tensor:
        """
        Dispatch to either the legacy DFoT rollout API or the MineWorld renderer.
        """
        if hasattr(self.world_model, "render_rollout"):
            rollouts = self.world_model.render_rollout(context, sequences)
            if isinstance(rollouts, list):
                rollouts = torch.stack(rollouts, dim=0)
            return rollouts
        if hasattr(self.world_model, "rollout"):
            one_hot = self._one_hot(sequences).to(self.device)
            return self.world_model.rollout(context, one_hot)
        raise AttributeError(
            "MineWorldModel must expose `render_rollout` or `rollout` for planning."
        )

    def _score_rollouts(
        self,
        rollouts: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[List[torch.Tensor]]]:
        """
        Score each rollout trajectory with the Gemini VLM; also return raw frames
        for optional visualization.
        """
        batch_rollouts: List[List[torch.Tensor]] = []
        scores: List[float] = []
        for traj in rollouts:
            frames = list(torch.unbind(traj, dim=0))
            batch_rollouts.append(frames)
            try:
                vlm_scores = self.vlm.score_frames(frames, instructions=self.cfg.instructions)
                score = float(vlm_scores.mean().item())
            except Exception as exc:  # pragma: no cover - defensive fallback.
                warnings.warn(f"VLM scoring failed, falling back to luminance: {exc}")
                stacked = torch.stack(frames, dim=0)
                score = float(stacked.mean().item())
            scores.append(score)
        return torch.tensor(scores, device=self.device), batch_rollouts

    def _maybe_visualize(self, rollouts: Sequence[Sequence[torch.Tensor]]) -> None:
        viz_cfg = self.cfg.visualization
        if not viz_cfg.is_enabled():
            return
        try:
            import imageio.v3 as iio  # type: ignore
        except ImportError:  # pragma: no cover - optional dependency.
            warnings.warn("imageio is not installed; skipping rollout visualization.")
            return

        output_root = Path(viz_cfg.output_dir).expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        for idx, frames in enumerate(itertools.islice(rollouts, viz_cfg.max_videos)):
            name = f"{viz_cfg.prefix}_{self._viz_step:04d}_{idx:02d}.mp4"
            target = output_root / name
            video = [_to_numpy_image(frame).numpy() for frame in frames]
            iio.imwrite(target, video, fps=viz_cfg.fps)
        self._viz_step += 1

    def _eval_sequences(
        self,
        context: torch.Tensor,
        sequences: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[List[torch.Tensor]]]:
        rollout = self._simulate_rollouts(context, sequences)
        scores, frames = self._score_rollouts(rollout)
        return scores, frames

    def plan(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the best action sequence + VLM scores for each context in the batch.
        """
        if context.ndim != 5:
            raise ValueError("Planner expects BCHWT context tensors.")
        context = context.to(self.device)
        batch = context.shape[0]
        horizon = self.cfg.horizon
        probs = self._initial_probs(context)

        for _ in range(self.cfg.iterations):
            samples = torch.distributions.Categorical(probs=probs).sample(
                (self.cfg.num_candidates,)
            )
            samples = samples.permute(1, 0, 2)  # (B, num_candidates, horizon)
            flat_samples = samples.reshape(batch * self.cfg.num_candidates, horizon)
            ctx = context.repeat_interleave(self.cfg.num_candidates, dim=0)
            scores, frames = self._eval_sequences(ctx, flat_samples)
            scores = scores.view(batch, self.cfg.num_candidates)

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
            if self.cfg.policy_dropout > 0 and self.policy is not None:
                uniform = 1 / self.action_dim
                probs = (1 - self.cfg.policy_dropout) * probs + self.cfg.policy_dropout * uniform
            probs = torch.clamp(probs, min=1e-4)
            probs = probs / probs.sum(dim=-1, keepdim=True)

        best_idx = probs.argmax(dim=-1)
        best_scores, best_rollouts = self._eval_sequences(context, best_idx)
        self.latest_rollouts = [list(seq) for seq in best_rollouts]
        self._maybe_visualize(self.latest_rollouts)
        return best_idx, best_scores
