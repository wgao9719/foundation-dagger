"""Evaluate a ResNet-based FoundationBCPolicy inside a MineRL BASALT environment."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import gym
import minerl  # noqa: F401 - ensures MineRL envs are registered
import numpy as np
import torch

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from algorithms.foundation_dagger.policy import (
    ActionHeadConfig,
    FoundationBCPolicy,
    PolicyConfig,
)
from algorithms.foundation_dagger.vpt_model.action_mapping import CameraHierarchicalMapping
from algorithms.foundation_dagger.vpt_model.actions import ActionTransformer
from evaluation.agent import ACTION_TRANSFORMER_KWARGS, AGENT_RESOLUTION, resize_image

ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT_DIR / "configurations"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs" / "minerl_eval"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)


def _reset_env(env, *, seed=None):
    if seed is not None:
        try:
            env.seed(seed)
        except (AttributeError, TypeError):
            pass
    reset_out = env.reset()
    if isinstance(reset_out, tuple):
        obs, info = reset_out
    else:
        obs, info = reset_out, {}
    return obs, info


def _step_env(env, action):
    step_out = env.step(action)
    if len(step_out) == 5:
        obs, reward, terminated, truncated, info = step_out
        done = bool(terminated or truncated)
    else:
        obs, reward, done, info = step_out
    return obs, reward, done, info


def _load_hydra_configs(
    dataset_name: str,
    algorithm_name: str,
    experiment_name: str,
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    overrides: list[str] = []
    if dataset_name:
        overrides.append(f"dataset={dataset_name}")
    if algorithm_name:
        overrides.append(f"algorithm={algorithm_name}")
    if experiment_name:
        overrides.append(f"experiment={experiment_name}")

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="config", overrides=overrides)

    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    algorithm_cfg = OmegaConf.to_container(cfg.algorithm, resolve=True)
    experiment_cfg = OmegaConf.to_container(cfg.experiment, resolve=True)
    return dataset_cfg, algorithm_cfg, experiment_cfg


def _coerce_policy_config(raw_cfg: Dict[str, Any]) -> PolicyConfig:
    valid_fields = {field.name for field in fields(PolicyConfig)}
    filtered_cfg = {key: value for key, value in raw_cfg.items() if key in valid_fields}

    action_cfg = filtered_cfg.get("action")
    if isinstance(action_cfg, dict):
        filtered_cfg["action"] = ActionHeadConfig(**action_cfg)
    elif not isinstance(action_cfg, ActionHeadConfig):
        filtered_cfg["action"] = ActionHeadConfig(buttons_classes=80, camera_bins=11, use_camera_gate=False)
    return PolicyConfig(**filtered_cfg)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _extract_frame(obs: Any) -> Optional[Any]:
    if isinstance(obs, dict):
        frame = obs.get("pov")
        if frame is not None:
            return frame
    return None


class EpisodeVideoRecorder:
    """Simple RGB video recorder backed by OpenCV."""

    def __init__(self, path: Path, fps: int) -> None:
        self.path = path
        self.fps = fps
        self._writer = None

    def append(self, frame: Any) -> None:
        if frame is None:
            return
        if self._writer is None:
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, (width, height))
            if not self._writer.isOpened():
                self._writer.release()
                self._writer = None
                raise RuntimeError(f"Could not open video writer at {self.path}")
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self._writer.write(bgr_frame)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


class BCPolicyAgent:
    def __init__(
        self,
        policy_cfg: PolicyConfig,
        checkpoint_path: Path,
        device: Optional[str],
        context_frames: int,
    ) -> None:
        dev = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.device = dev
        self.policy = FoundationBCPolicy(policy_cfg).to(dev)
        state_dict = torch.load(checkpoint_path, map_location=dev)
        self.policy.load_state_dict(state_dict)
        self.policy.eval()

        self.context_frames = max(1, int(context_frames))
        self.frame_buffer: deque[torch.Tensor] = deque(maxlen=self.context_frames)
        self.channel_mean = IMAGENET_MEAN.view(3, 1, 1).to(dev)
        self.channel_std = IMAGENET_STD.view(3, 1, 1).to(dev)

        self.action_mapper = CameraHierarchicalMapping(n_camera_bins=policy_cfg.action.camera_bins)
        self.action_transformer = ActionTransformer(**ACTION_TRANSFORMER_KWARGS)

    def reset(self) -> None:
        self.frame_buffer.clear()

    def _prepare_frame(self, obs: Dict[str, Any]) -> torch.Tensor:
        frame = resize_image(obs["pov"], AGENT_RESOLUTION)
        tensor = torch.from_numpy(frame).to(self.device, dtype=torch.float32).permute(2, 0, 1)
        tensor = tensor.div(255.0)
        tensor = (tensor - self.channel_mean) / self.channel_std
        return tensor

    def _stack_frames(self, new_frame: torch.Tensor) -> torch.Tensor:
        if not self.frame_buffer:
            for _ in range(self.context_frames):
                self.frame_buffer.append(new_frame.clone())
        else:
            self.frame_buffer.append(new_frame)
        frames = torch.stack(list(self.frame_buffer), dim=0)  # [T, C, H, W]
        return frames.unsqueeze(0)  # [1, T, C, H, W]

    @staticmethod
    def _format_action(env_action: Dict[str, Any]) -> Dict[str, Any]:
        formatted: Dict[str, Any] = {}
        for key, value in env_action.items():
            arr = np.asarray(value)
            if arr.shape[0] == 1:
                arr = arr[0]
            if isinstance(arr, np.ndarray) and arr.ndim == 0:
                formatted[key] = arr.item()
            else:
                formatted[key] = arr
        return formatted

    def get_action(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        frame_tensor = self._prepare_frame(obs)
        frames = self._stack_frames(frame_tensor)

        with torch.no_grad():
            logits = self.policy(frames)
            last_logits = {key: value[:, -1] for key, value in logits.items()}
            buttons_idx = torch.argmax(last_logits["buttons"], dim=-1).view(-1, 1).cpu().numpy()
            camera_idx = torch.argmax(last_logits["camera"], dim=-1).view(-1, 1).cpu().numpy()

            factored = self.action_mapper.to_factored({"buttons": buttons_idx, "camera": camera_idx})
            env_action = self.action_transformer.policy2env(factored)
            env_action = self._format_action(env_action)
            print("env_action", env_action)

            esc_value = 0
            esc_logits = last_logits.get("esc")
            if esc_logits is not None:
                esc_value = int(torch.argmax(esc_logits, dim=-1).item())
            env_action["ESC"] = esc_value
        return env_action


def evaluate_policy(args: argparse.Namespace) -> None:
    dataset_cfg, algorithm_cfg, _ = _load_hydra_configs(
        dataset_name=args.dataset,
        algorithm_name=args.algorithm,
        experiment_name=args.experiment,
    )

    policy_cfg_dict = algorithm_cfg.get("policy")
    if policy_cfg_dict is None:
        raise ValueError(f"Algorithm config '{args.algorithm}' does not define a 'policy' block.")
    policy_cfg = _coerce_policy_config(policy_cfg_dict)

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    context_frames = int(dataset_cfg.get("context_frames", 1))
    env = gym.make(args.env_id)

    output_root = _ensure_dir(Path(args.output_dir or DEFAULT_OUTPUT_DIR).expanduser().resolve())
    run_dir = _ensure_dir(output_root / f"{checkpoint_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    agent = BCPolicyAgent(
        policy_cfg=policy_cfg,
        checkpoint_path=checkpoint_path,
        device=args.device,
        context_frames=context_frames,
    )

    for episode_idx in range(args.episodes):
        obs, _ = _reset_env(env, seed=(args.seed + episode_idx) if args.seed is not None else None)
        agent.reset()

        total_reward = 0.0
        steps = 0
        done = False
        video_path = run_dir / f"episode_{episode_idx:03d}.mp4"
        recorder = EpisodeVideoRecorder(video_path, fps=args.video_fps)
        recorder.append(_extract_frame(obs))

        try:
            while not done and (args.max_steps is None or steps < args.max_steps):
                action = agent.get_action(obs)
                obs, reward, done, _ = _step_env(env, action)
                if args.render:
                    env.render()

                recorder.append(_extract_frame(obs))

                total_reward += float(reward)
                steps += 1
        finally:
            recorder.close()

        print(f"Episode {episode_idx + 1}: reward={total_reward:.2f}, steps={steps}, video={video_path}")

    env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a FoundationBCPolicy checkpoint (.ckpt).")
    parser.add_argument("--dataset", type=str, default="mineworld_frames", help="Hydra dataset config to load.")
    parser.add_argument("--algorithm", type=str, default="policy_bc", help="Hydra algorithm config that defines the policy.")
    parser.add_argument("--experiment", type=str, default="mineworld_bc_train", help="Hydra experiment config for defaults.")
    parser.add_argument("--env-id", type=str, default="MineRLBasaltFindCave-v0", help="MineRL environment id.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Directory to store recorded evaluation videos.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of evaluation episodes.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional hard limit on steps per episode.")
    parser.add_argument("--video-fps", type=int, default=20, help="Frames per second for recorded videos.")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default=None, help="Computation device override.")
    parser.add_argument("--seed", type=int, default=None, help="Base random seed for environment resets.")
    parser.add_argument("--render", action="store_true", help="Render the MineRL environment locally during evaluation.")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate_policy(parse_args())
