from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import gym
import cv2
import minerl
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from algorithms.foundation_dagger.vpt_model.load_vpt_config import VPTConfig, parse_policy_config
from evaluation.agent import MineRLAgent

ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT_DIR / "configurations"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs" / "minerl_eval"


def _reset_env(env, *, seed=None):
    # Handle seed separately for compatibility with MineRL environments
    if seed is not None:
        try:
            env.seed(seed)
        except (AttributeError, TypeError):
            pass  # Environment doesn't support seeding
    
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
    algorithm_name: str,
) -> Dict[str, Any]:
    overrides: list[str] = []
    if algorithm_name:
        overrides.append(f"algorithm={algorithm_name}")

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="config", overrides=overrides)

    policy_cfg = OmegaConf.to_container(cfg.algorithm, resolve=True)
    return policy_cfg


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


def evaluate_policy(args: argparse.Namespace) -> None:
    policy_cfg_dict = _load_hydra_configs(algorithm_name=args.algorithm)
    try:
        policy_cfg: VPTConfig = parse_policy_config(policy_cfg_dict)
    except TypeError as exc:
        raise ValueError(
            f"Algorithm config '{args.algorithm}' is not compatible with the VPT loader (missing VPT fields)."
        ) from exc

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    #action_space = CameraHierarchicalMapping(n_camera_bins=11)
    model_args = policy_cfg.model.get("args", {})
    policy_kwargs = model_args.get("net", {}).get("args", {})
    pi_head_kwargs = model_args.get("pi_head_opts", {})

    env = gym.make(args.env_id)

    output_root = _ensure_dir(Path(args.output_dir or DEFAULT_OUTPUT_DIR).expanduser().resolve())
    run_dir = _ensure_dir(output_root / f"{checkpoint_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    agent = MineRLAgent(env, device=args.device, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(checkpoint_path)
    
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
                action["ESC"] = 0
                obs, reward, done, _ = _step_env(env, action)
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
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to VPT weights file (.weights).")
    parser.add_argument("--algorithm", type=str, default="vpt_bc", help="Hydra algorithm config to load.")
    parser.add_argument("--env-id", type=str, default="MineRLBasaltFindCave-v0", help="MineRL environment id.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Directory to store recorded evaluation videos.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of evaluation episodes.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional hard limit on steps per episode.")
    parser.add_argument("--video-fps", type=int, default=20, help="Frames per second for recorded videos.")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default=None, help="Computation device override.")
    parser.add_argument("--seed", type=int, default=None, help="Base random seed for environment resets.")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate_policy(parse_args())
