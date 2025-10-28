from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import torch
from PIL import Image

try:
    from transformers import pipeline
except ImportError:  # pragma: no cover - transformers is optional at runtime.
    pipeline = None


def _to_pil(image: torch.Tensor) -> Image.Image:
    if isinstance(image, torch.Tensor):
        tensor = image.detach().cpu()
        if tensor.dim() == 3 and tensor.size(0) in (1, 3):
            tensor = tensor.clamp(0, 1)
            tensor = tensor.mul(255).byte()
            if tensor.size(0) == 1:
                tensor = tensor.repeat(3, 1, 1)
            tensor = tensor.permute(1, 2, 0).numpy()
            return Image.fromarray(tensor)
    raise ValueError("Expected BCHW tensor scaled to [0, 1] for VLM scoring.")


@dataclass
class VLMConfig:
    model_id: Optional[str] = None
    device: Optional[str] = None
    good_prompts: Sequence[str] = ("a successful rollout",)
    bad_prompts: Sequence[str] = ("a failed rollout",)
    mode: str = "zero-shot"


class VisionLanguageScorer:
    """
    Wraps a VLM or heuristic scorer for evaluating rollouts.
    """

    def __init__(self, cfg: VLMConfig) -> None:
        self.cfg = cfg
        self.good_prompts = list(cfg.good_prompts)
        self.bad_prompts = list(cfg.bad_prompts)
        self.pipeline = None
        if cfg.model_id and pipeline is not None:
            self.pipeline = pipeline(
                "zero-shot-image-classification",
                model=cfg.model_id,
                device=cfg.device,
            )

    @torch.no_grad()
    def score_frames(self, frames: Iterable[torch.Tensor]) -> torch.Tensor:
        """
        Score a list of frames; positive scores mean success.
        """
        if self.pipeline is None:
            # fallback: simple luminance heuristic for offline development
            stacked = torch.stack([frame.mean() for frame in frames])
            return stacked

        pil_frames = [_to_pil(frame) for frame in frames]
        scores = []
        for image in pil_frames:
            result = self.pipeline(
                image,
                candidate_labels=list(self.good_prompts) + list(self.bad_prompts),
            )
            score = 0.0
            for item in result:
                label = item["label"]
                value = float(item["score"])
                if label in self.good_prompts:
                    score += value
                elif label in self.bad_prompts:
                    score -= value
            scores.append(score)
        return torch.tensor(scores)
