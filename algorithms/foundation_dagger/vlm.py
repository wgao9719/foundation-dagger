from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Union

import torch
from PIL import Image

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover - gemini client is optional at runtime.
    genai = None

InstructionInput = Union[str, Sequence[str]]


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
    model_id: Optional[str] = "gemini-1.5-flash"
    api_key: Optional[str] = None
    api_key_env: str = "GOOGLE_API_KEY"
    instructions: str = "build a minecraft village house"
    default_instruction: str = (
        "You are an expert minecraft player evaluating another player's ability to complete a task."
        f"The task has the following instructions: {instructions}"
        "Score how successful the depicted rollout is on a scale from -1.0 (bad) "
        "to 1.0 (good)."
    )


class VisionLanguageScorer:
    """
    Wraps a Gemini VLM or heuristic scorer for evaluating rollouts.
    """

    def __init__(self, cfg: VLMConfig) -> None:
        self.cfg = cfg
        self._model = None
        api_key = cfg.api_key or (os.environ.get(cfg.api_key_env or "") if cfg.api_key_env else None)
        if cfg.model_id and genai is not None and api_key:
            genai.configure(api_key=api_key)
            self._model = genai.GenerativeModel(cfg.model_id)

    @torch.no_grad()
    def score_frames(
        self,
        frames: Iterable[torch.Tensor],
        instructions: Optional[InstructionInput] = None,
    ) -> torch.Tensor:
        """
        Score a sequence of frames with Gemini. Custom instructions override the default prompt.
        """
        frame_tensors = list(frames)
        if not frame_tensors:
            return torch.empty(0)
        if self._model is None:
            return self._luminance_scores(frame_tensors)

        prompt_parts = self._normalize_instructions(instructions)
        pil_frames = [_to_pil(frame) for frame in frame_tensors]
        scores = []
        for tensor, image in zip(frame_tensors, pil_frames):
            try:
                scores.append(self._score_frame(image, prompt_parts))
            except Exception:
                scores.append(float(self._luminance_scores([tensor])[0]))
        return torch.tensor(scores, dtype=torch.float32)

    def _normalize_instructions(self, instructions: Optional[InstructionInput]) -> Sequence[str]:
        if instructions is None:
            instructions = self.cfg.default_instruction
        if isinstance(instructions, str):
            instructions = [instructions]
        instructions = [part for part in instructions if part]
        if not instructions:
            raise ValueError("At least one instruction string is required for Gemini scoring.")
        return instructions

    def _score_frame(self, image: Image.Image, prompt_parts: Sequence[str]) -> float:
        prompt = list(prompt_parts) + [image]
        response = self._model.generate_content(prompt)
        text = (response.text or "").strip()
        score = self._parse_score(text)
        return float(max(min(score, 1.0), -1.0))

    @staticmethod
    def _parse_score(raw_text: str) -> float:
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            parsed = raw_text

        if isinstance(parsed, dict) and "score" in parsed:
            return float(parsed["score"])
        if isinstance(parsed, (int, float)):
            return float(parsed)
        if isinstance(parsed, str):
            match = re.search(r"-?\d+(?:\.\d+)?", parsed)
            if match:
                return float(match.group(0))
        raise ValueError(f"Unable to parse score from Gemini response: {raw_text}")

    @staticmethod
    def _luminance_scores(frames: Sequence[torch.Tensor]) -> torch.Tensor:
        values = [float(frame.detach().mean().cpu()) for frame in frames]
        return torch.tensor(values, dtype=torch.float32)
