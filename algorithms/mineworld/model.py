from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import cv2
import numpy as np
import torch
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from torchvision import transforms

from datasets.mineworld_data.mcdataset import MCDataset
from utils import load_model, print0
from utils.ckpt_utils import download_pretrained

SAFE_GLOBALS: Dict[str, object] = {"array": np.array}


@dataclass
class GenerationResult:
    video_path: Path
    actions_path: Optional[Path]
    output_path: Path
    elapsed: float
    generated_frames: int
    token_count: int
    skipped: bool = False


class MineWorldModel:
    """
    Thin wrapper around the MineWorld Llama LVM for Hydra integration.
    """

    def __init__(self, cfg: DictConfig) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("MineWorld inference requires a CUDA-capable GPU.")
        self.cfg = cfg
        self.device = torch.device(cfg.get("device", "cuda"))
        algo_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        checkpoint = algo_cfg.get("checkpoint")
        if checkpoint is None:
            raise ValueError(
                "MineWorld algorithm config must specify `checkpoint`. "
                "Provide a local path or `pretrained:` reference."
            )
        ckpt_path = self._resolve_checkpoint(checkpoint)

        print0(f"[bold cyan][MineWorld][/bold cyan] Loading model from {ckpt_path}")
        self.model = load_model(algo_cfg, ckpt_path, gpu=True, eval_mode=True)
        self.model.transformer.eval().to(self.device)
        self.tokenizer = self.model.tokenizer

        self.token_per_image = int(algo_cfg.get("token_per_image", 336))
        self.action_vocab_offset = int(algo_cfg.get("action_vocab_offset", 8192))
        self.action_encoder = MCDataset()
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def generate_sequence(
        self,
        *,
        video_path: Path,
        actions_path: Optional[Path],
        output_path: Path,
        context_frames: int,
        prediction_frames: int,
        accelerate_algo: str,
        window_size: int,
        sampler: Dict[str, Optional[float]],
        fps: int,
        overwrite: bool,
        copy_actions: bool,
    ) -> GenerationResult:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and not overwrite:
            print0(
                f"[yellow][MineWorld][/yellow] Skipping {output_path.name}: output exists and overwrite disabled."
            )
            return GenerationResult(
                video_path=video_path,
                actions_path=actions_path,
                output_path=output_path,
                elapsed=0.0,
                generated_frames=0,
                token_count=0,
                skipped=True,
            )

        image_input = self._tokenize_context(video_path, context_frames)
        action_tokens = self._prepare_actions(
            actions_path, context_frames, prediction_frames
        )

        temperature = sampler.get("temperature", 1.0)
        top_p = sampler.get("top_p")
        top_k = sampler.get("top_k")

        accelerate_algo = accelerate_algo.lower()
        if accelerate_algo not in {"naive", "image_diagd"}:
            raise ValueError(
                f"Unsupported accelerate algorithm '{accelerate_algo}'. "
                "Expected 'naive' or 'image_diagd'."
            )

        max_new_tokens = self.token_per_image * prediction_frames
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad(), torch.autocast(
            device_type=self.device.type, dtype=torch.float16
        ):
            if accelerate_algo == "naive":
                outputs = self.model.transformer.naive_generate(
                    input_ids=image_input,
                    max_new_tokens=max_new_tokens,
                    action_all=action_tokens,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                )
            else:
                outputs = self.model.transformer.img_diagd_generate(
                    input_ids=image_input,
                    max_new_tokens=max_new_tokens,
                    action_all=action_tokens,
                    windowsize=window_size,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                )
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end) / 1000.0

        tokens = outputs.tolist()[0]
        frames = self._tokens_to_frames(tokens)
        self._write_video(frames, output_path, fps)
        if copy_actions and actions_path and actions_path.exists():
            copy_target = output_path.with_suffix(".jsonl")
            copy_target.write_bytes(actions_path.read_bytes())

        return GenerationResult(
            video_path=video_path,
            actions_path=actions_path,
            output_path=output_path,
            elapsed=elapsed,
            generated_frames=len(frames),
            token_count=len(tokens),
            skipped=False,
        )

    def _resolve_checkpoint(self, checkpoint: str) -> Path:
        if checkpoint.startswith(("pretrained:", "full:")):
            return Path(download_pretrained(checkpoint))
        return Path(checkpoint).expanduser().resolve()

    def _tokenize_context(self, video_path: Path, context_frames: int) -> torch.Tensor:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file {video_path}")
        frames = []
        try:
            for idx in range(context_frames):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                success, frame = cap.read()
                if not success:
                    raise RuntimeError(
                        f"Video {video_path} does not contain {context_frames} frames."
                    )
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = torch.from_numpy(np.asarray(frame, dtype=np.uint8))
                frames.append(frame)
        finally:
            cap.release()

        stacked = torch.stack(frames, dim=0).to(self.device)
        stacked = stacked.permute(0, 3, 1, 2).float() / 255.0
        stacked = self.normalize(stacked)

        with torch.no_grad(), torch.autocast(
            device_type=self.device.type, dtype=torch.float16
        ):
            tokens = self.model.tokenizer.tokenize_images(stacked)

        tokens = rearrange(tokens, "(b t) h w -> b t (h w)", b=1)
        return rearrange(tokens, "b t c -> b (t c)")

    def _prepare_actions(
        self,
        actions_path: Optional[Path],
        context_frames: int,
        prediction_frames: int,
    ) -> torch.Tensor:
        if actions_path is None or not actions_path.exists():
            raise FileNotFoundError(
                f"MineWorld requires an action file, but none was found for {actions_path}."
            )
        action_sequences = self._load_action_tokens(actions_path)
        required = context_frames + prediction_frames
        if len(action_sequences) < required:
            raise ValueError(
                f"Action file {actions_path} only contains {len(action_sequences)} steps, "
                f"but {required} are required for context ({context_frames}) + rollout ({prediction_frames})."
            )
        future_actions = action_sequences[context_frames : context_frames + prediction_frames]
        action_tensor = torch.tensor(future_actions, dtype=torch.long, device=self.device)
        return action_tensor.unsqueeze(1)

    def _load_action_tokens(self, action_path: Path) -> list[list[int]]:
        tokens: list[list[int]] = []
        with action_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                action_dict = eval(line, {"__builtins__": None}, SAFE_GLOBALS)  # noqa: S307
                action_dict["camera"] = np.array(action_dict["camera"])
                indices = self.action_encoder.get_action_index_from_actiondict(
                    action_dict, action_vocab_offset=self.action_vocab_offset
                )
                tokens.append([int(x) for x in indices])
        if not tokens:
            raise ValueError(f"No actions parsed from {action_path}")
        return tokens

    def _tokens_to_frames(self, tokens: Iterable[int]) -> list[np.ndarray]:
        tokens = list(tokens)
        if len(tokens) % self.token_per_image != 0:
            raise ValueError(
                f"Generated token count {len(tokens)} is not a multiple of {self.token_per_image}."
            )
        frames: list[np.ndarray] = []
        for idx in range(len(tokens) // self.token_per_image):
            chunk = tokens[
                idx * self.token_per_image : (idx + 1) * self.token_per_image
            ]
            code = torch.tensor(chunk, dtype=torch.long, device=self.device)
            img = self.tokenizer.token2image(code)
            frames.append(img)
        return frames

    def _write_video(self, frames: list[np.ndarray], output_path: Path, fps: int) -> None:
        if not frames:
            raise ValueError("No frames generated to write.")
        height, width = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        try:
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
