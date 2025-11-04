from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import gym
import minerl 
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from algorithms.foundation_dagger.vpt_model.load_vpt_config import VPTConfig, parse_policy_config
from evaluation.agent import MineRLAgent

ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT_DIR / "configurations"

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

    policy_cfg = OmegaConf.to_container(cfg.algorithm.policy, resolve=True)
    return policy_cfg


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

    agent = MineRLAgent(env, device=args.device, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(checkpoint_path)
    
    for episode_idx in range(args.episodes):
        if args.seed is not None:
            obs = env.reset(seed=args.seed + episode_idx)
        else:
            obs = env.reset()
        agent.reset()

        total_reward = 0.0
        steps = 0
        done = False

        while not done and (args.max_steps is None or steps < args.max_steps):
            action = agent.get_action(obs)
            action["ESC"] = 0
            obs, reward, done, _ = env.step(action)
            env.render()

            total_reward += float(reward)
            steps += 1

        print(f"Episode {episode_idx + 1}: reward={total_reward:.2f}, steps={steps}")

    env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to VPT weights file (.weights).")
    parser.add_argument("--algorithm", type=str, default="vpt_bc", help="Hydra algorithm config to load.")
    parser.add_argument("--env-id", type=str, default="MineRLBasaltFindCave-v0", help="MineRL environment id.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of evaluation episodes.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional hard limit on steps per episode.")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default=None, help="Computation device override.")
    parser.add_argument("--seed", type=int, default=None, help="Base random seed for environment resets.")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate_policy(parse_args())
