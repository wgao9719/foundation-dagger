"""Evaluate a MineWorld BC policy inside a MineRL BASALT environment.

The script loads a `FoundationBCPolicy`, performs greedy rollouts in a chosen
MineRL-Gym task, and saves episode videos for qualitative inspection.

python -m scripts.minerl_evaluation --checkpoint checkpoints/bc_policy.ckpt --env-id MineRLBasaltFindCave-v0
"""

from __future__ import annotations

import argparse
import copy
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Tuple

import gym
import imageio.v2 as imageio
import minerl  # noqa: F401 - needed to register MineRL environments
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from torch import Tensor
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from algorithms.foundation_dagger.policy import (
    BasePolicyConfig,
    VPTCausalPolicy,
    build_policy,
    parse_policy_config,
)
from datasets.mineworld_data.mcdataset import MCDataset


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT_DIR / "configurations"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _load_hydra_configs(
    dataset_name: str,
    algorithm_name: str,
    experiment_name: str,
) -> Tuple[Dict, Dict, Dict]:
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
    policy_cfg = OmegaConf.to_container(cfg.algorithm.policy, resolve=True)
    exp_cfg = OmegaConf.to_container(cfg.experiment, resolve=True)
    return dataset_cfg, policy_cfg, exp_cfg


def _build_transform(resize: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((resize, resize), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _prep_frame(frame: np.ndarray, transform: transforms.Compose, device: torch.device) -> Tensor:
    tensor = transform(frame)
    return tensor.unsqueeze(0).to(device, non_blocking=True)


def _legacy_camera_bins_from_tokens(tokens: Iterable[str]) -> np.ndarray:
    camera_bins = [0, 0]
    for token in tokens:
        if token.startswith("cam_0_"):
            try:
                camera_bins[0] = int(token.rsplit("_", 1)[-1])
            except (ValueError, IndexError):
                camera_bins[0] = 0
        elif token.startswith("cam_1_"):
            try:
                camera_bins[1] = int(token.rsplit("_", 1)[-1])
            except (ValueError, IndexError):
                camera_bins[1] = 0
    return np.array(camera_bins, dtype=np.int64)


def _decode_action(
    token_indices: Iterable[int],
    mc_dataset: MCDataset,
    noop_template: Dict,
) -> Dict:
    # Start from a fresh noop action to ensure compatibility with the chosen environment.
    action = copy.deepcopy(noop_template)
    available_keys = set(action.keys())
    camera_value = action.get("camera", np.zeros(2, dtype=np.float32))
    if not isinstance(camera_value, np.ndarray):
        camera_value = np.asarray(camera_value, dtype=np.float32)
    action["camera"] = camera_value.astype(np.float32, copy=False)

    gate_token_idx = int(token_indices[1]) if len(token_indices) > 1 else None
    joint_token_idx = int(token_indices[2]) if len(token_indices) > 2 else None
    inv_vocab = {v: k for k, v in mc_dataset.action_vocab.items()}
    tokens = [inv_vocab.get(idx, "<null_act>") for idx in token_indices]

    if len(tokens) < 3:
        return action
    if gate_token_idx is not None and joint_token_idx is not None and hasattr(mc_dataset, "camera_tokens_to_bins"):
        cam_bins = mc_dataset.camera_tokens_to_bins(gate_token_idx, joint_token_idx)
    else:
        cam_bins = _legacy_camera_bins_from_tokens(tokens[1:3])
    camera = mc_dataset.camera_quantizer.undiscretize(cam_bins)
    action["camera"][0] = float(camera[0])
    action["camera"][1] = float(camera[1])

    hotbar_token = tokens[3] if len(tokens) > 3 else "<null_act>"
    if hotbar_token.startswith("hotbar.") and hotbar_token in available_keys:
        action[hotbar_token] = 1

    def _set_binary(token: str, positive_key: str, negative_key: str | None = None) -> None:
        if token == positive_key and positive_key in available_keys:
            action[positive_key] = 1
        elif negative_key and token == negative_key and negative_key in available_keys:
            action[negative_key] = 1

    if len(tokens) > 4:
        _set_binary(tokens[4], "forward", "back")
    if len(tokens) > 5:
        _set_binary(tokens[5], "left", "right")
    if len(tokens) > 6:
        _set_binary(tokens[6], "sprint", "sneak")
    if len(tokens) > 7:
        _set_binary(tokens[7], "use", "attack")
    if len(tokens) > 8 and tokens[8] == "jump" and "jump" in available_keys:
        action["jump"] = 1
    if len(tokens) > 9:
        _set_binary(tokens[9], "drop", "pickItem")

    # Reset inventory-related toggles that would pause the game.
    if "inventory" in action:
        action["inventory"] = 0
    if "ESC" in action:
        action["ESC"] = 0
    return action


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def evaluate_policy(args: argparse.Namespace) -> None:
    dataset_cfg, policy_cfg_dict, exp_cfg = _load_hydra_configs(
        dataset_name=args.dataset,
        algorithm_name=args.algorithm,
        experiment_name=args.experiment,
    )

    policy_cfg: BasePolicyConfig = parse_policy_config(policy_cfg_dict)
    mc_dataset = MCDataset()
    mc_dataset.make_action_vocab(action_vocab_offset=0)
    action_length = mc_dataset.action_length
    action_vocab_size = len(mc_dataset.action_vocab)
    policy_cfg.action_dim = action_length * action_vocab_size

    resize = int(dataset_cfg.get("resize", dataset_cfg.get("resolution", 256)))
    transform = _build_transform(resize)
    context_frames = int(dataset_cfg.get("context_frames", 1))

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    policy = build_policy(policy_cfg).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(state_dict)
    policy.eval()

    video_root = _ensure_dir(Path(args.video_dir or (ROOT_DIR / "outputs" / "minerl_eval")))
    run_dir = _ensure_dir(video_root / f"{checkpoint_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    env = gym.make(args.env_id)
    noop_template = env.action_space.noop()

    for episode_idx in range(args.episodes):
        if args.seed is not None:
            reset_out = env.reset(seed=args.seed + episode_idx)
        else:
            reset_out = env.reset()

        if isinstance(reset_out, tuple):
            obs = reset_out[0]
        else:
            obs = reset_out

        video_path = run_dir / f"episode_{episode_idx:03d}.mp4"
        writer = imageio.get_writer(video_path, fps=args.video_fps)

        frame_buffer: deque[Tensor] = deque(maxlen=context_frames)
        mems = None

        total_reward = 0.0
        steps = 0
        done = False

        while not done and (args.max_steps is None or steps < args.max_steps):
            frame = obs["pov"]
            writer.append_data(frame)

            with torch.no_grad():
                input_tensor = _prep_frame(frame, transform, device)
                frame_buffer.append(input_tensor.squeeze(0))
                if len(frame_buffer) < context_frames:
                    padded = [frame_buffer[0]] * (context_frames - len(frame_buffer)) + list(frame_buffer)
                else:
                    padded = list(frame_buffer)
                stacked = torch.stack(padded, dim=0).unsqueeze(0)

                if isinstance(policy, VPTCausalPolicy):
                    logits, mems = policy(stacked, mems=mems, return_mems=True)
                else:
                    logits = policy(stacked)

                logits = logits.view(1, action_length, action_vocab_size)
                token_indices = logits.argmax(dim=-1).squeeze(0).tolist()

            action = _decode_action(token_indices, mc_dataset, noop_template)
            step_out = env.step(action)

            if len(step_out) == 5:
                obs, reward, terminated, truncated, _ = step_out
                done = bool(terminated or truncated)
            else:
                obs, reward, done, _ = step_out

            total_reward += float(reward)
            steps += 1

        if isinstance(obs, dict) and "pov" in obs:
            writer.append_data(obs["pov"])

        writer.close()
        print(f"Episode {episode_idx + 1}: reward={total_reward:.2f}, steps={steps}, video={video_path}")

    env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a trained BC checkpoint (.ckpt).")
    parser.add_argument("--dataset", type=str, default="mineworld_frames", help="Hydra dataset config to load.")
    parser.add_argument("--algorithm", type=str, default="mineworld_bc", help="Hydra algorithm config to load.")
    parser.add_argument("--experiment", type=str, default="mineworld_bc_train", help="Hydra experiment config to load.")
    parser.add_argument("--env-id", type=str, default="MineRLBasaltFindCave-v0", help="MineRL environment id.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of evaluation episodes.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional hard limit on steps per episode.")
    parser.add_argument("--video-dir", type=str, default=None, help="Directory to store recorded evaluation videos.")
    parser.add_argument("--video-fps", type=int, default=20, help="Frames per second for output videos.")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default=None, help="Computation device override.")
    parser.add_argument("--seed", type=int, default=None, help="Base random seed for environment resets.")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate_policy(parse_args())
