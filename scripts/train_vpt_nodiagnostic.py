"""Train an initial BC policy on MineWorld rollouts.

The script ingests MineWorld-style trajectories (paired ``.mp4`` videos and
``.jsonl`` action logs) and produces a supervised policy checkpoint compatible
with the ``foundation_dagger`` module. Each training sample stacks the full
``N``-frame context window and maximises the joint log-likelihood of the final
frame's hierarchical action under the VPT policy head, enabling
Transformer-style policies to leverage temporal attention while still
supporting single-frame heads. Only a user-specified fraction of the available
samples is used to keep the bootstrap stage quick.

Example
-------
python -m scripts.train_vpt \
  --algorithm vpt_bc
  --dataset mineworld_frames \
  --experiment mineworld_bc_train \
  --data-root /workspace/foundation-dagger/foundation-dagger/data/mineworld \
  --fraction 0.3 \
  --epochs 20 \
  --batch-size 32 \
  --output checkpoints/bc_checkpoints/bc_policy.ckpt
"""


from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from typing import Any, Dict, List, Optional
from gym3.types import DictType

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
import wandb

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from datasets.mineworld_data.mineworld_frame_dataset import MineWorldFrameDataset

CONFIG_DIR = ROOT_DIR / "configurations"

from algorithms.foundation_dagger.vpt_model.policy import MinecraftAgentPolicy
from algorithms.foundation_dagger.vpt_model.action_mapping import CameraHierarchicalMapping
from algorithms.foundation_dagger.vpt_model.load_vpt_config import VPTConfig, parse_policy_config
from algorithms.foundation_dagger.vpt_model.tree_util import tree_map, tree_multimap

from utils.SequentialBatchSampler import PerVideoSequentialBatchSampler

def _load_hydra_configs(
    dataset_name: str,
    algorithm_name: str,
    experiment_name: str,
) -> tuple[dict, dict, dict]:
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
    policy_cfg = OmegaConf.to_container(cfg.algorithm, resolve=True)
    exp_cfg = OmegaConf.to_container(cfg.experiment, resolve=True)
    return dataset_cfg, policy_cfg, exp_cfg

def train_vpt_bc(
    data_root: Path,
    fraction: float,
    epochs: int,
    batch_size: int,
    context_frames: int,
    recursive: bool,
    resize: int,
    output: Path,
    dataset_cfg: Dict,
    policy_cfg_dict: Dict,
    algorithm_name: str,
    experiment_name: str,
    dataset_name: str,
    exp_cfg: Dict,
) -> None:
    # get configs
    policy_cfg: VPTConfig = parse_policy_config(policy_cfg_dict)

    #initialize dataset
    dataset = MineWorldFrameDataset(
        data_root=data_root,
        context_frames=context_frames,
        recursive=recursive,
        max_open_captures=int(dataset_cfg.get("max_open_captures", 2)),
    )
    train_cfg = exp_cfg.get("training", {})
    opt_cfg = exp_cfg.get("optimizer", {})

    # get subset length and indices
    subset_len = max(1, int(len(dataset) * fraction))
    indices = list(range(len(dataset)))
    random_seed = int(train_cfg.get("seed", dataset_cfg.get("seed", 0)))
    random.seed(random_seed)

    # Shuffle at the video level while preserving chronological ordering per video.
    video_to_indices: Dict[int, List[int]] = {}
    for idx, (_, vid, _) in enumerate(dataset.samples):
        video_to_indices.setdefault(int(vid), []).append(idx)
    video_ids = list(video_to_indices.keys())
    random.shuffle(video_ids)
    indices = []
    for vid in video_ids:
        indices.extend(video_to_indices[vid])

    # split dataset into training and validation
    val_fraction = float(train_cfg.get("val_fraction", dataset_cfg.get("val_fraction", 0.1)))
    if subset_len <= 1 or val_fraction <= 0:
        val_fraction = 0.0
        val_split = 0
    else:
        val_split = max(1, int(round(subset_len * val_fraction)))
        if val_split >= subset_len:
            val_split = max(1, subset_len - 1)
    val_indices = indices[:val_split]
    train_indices = indices[val_split:subset_len]
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    train_video_to_positions: Dict[int, List[int]] = {}
    for subset_idx, dataset_idx in enumerate(train_indices):
        _, vid, _ = dataset.samples[dataset_idx]
        train_video_to_positions.setdefault(int(vid), []).append(subset_idx)

    # load in training hparameters
    train_batch_size = batch_size
    train_epochs = epochs
    train_workers = int(train_cfg.get("num_workers", 4))
    val_batch_size = int(train_cfg.get("val_batch_size", train_batch_size))
    val_workers = int(train_cfg.get("val_num_workers", max(1, train_workers // 2)))
    checkpoint_interval = int(train_cfg.get("checkpoint_interval", 0))
    grad_clip_norm = float(train_cfg.get("grad_clip_norm", 1.0))
    
    # initialize wandb
    wandb.init(
        project="foundation-dagger",
        job_type="bc-bootstrap",
        config={
            "dataset": dataset_name,
            "algorithm": algorithm_name,
            "experiment": experiment_name,
            "data_root": str(data_root),
            "fraction": fraction,
            "epochs": train_epochs,
            "batch_size": train_batch_size,
            "context_frames": context_frames,
            "resize": resize,
            "val_fraction": val_fraction,
            "seed": random_seed,
            "train_samples": len(train_subset),
            "val_samples": len(val_subset),
            "grad_clip_norm": grad_clip_norm if grad_clip_norm is not None else 0.0,
        },
    )

    # initialize data loader
    prefetch_factor = int(train_cfg.get("prefetch_factor", 2))
    train_batch_sampler = PerVideoSequentialBatchSampler(
        train_video_to_positions=train_video_to_positions,
        batch_size=train_batch_size,
        seed=random_seed,
    )

    loader_kwargs = dict(
        batch_sampler=train_batch_sampler,
        num_workers=train_workers,
        pin_memory=True,
        persistent_workers=False,  # Disabled to avoid stale VideoCapture objects across epochs
    )
    if train_workers > 0:
        loader_kwargs["prefetch_factor"] = max(prefetch_factor, 1)
    loader = DataLoader(train_subset, **loader_kwargs)

    # initialize model
    action_mapper = CameraHierarchicalMapping(n_camera_bins=11)
    action_space = action_mapper.get_action_space_update()
    action_space = DictType(**action_space)

    model_args = policy_cfg.model.get("args", {})
    policy_kwargs = model_args.get("net", {}).get("args", {})
    pi_head_kwargs = model_args.get("pi_head_opts", {})

    model = MinecraftAgentPolicy(action_space, policy_kwargs, pi_head_kwargs)

    # initialize weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # path = "/Users/willi1/foundation-dagger/diffusion-forcing-transformer/checkpoints/vpt/foundation-model-1x.weights"
    path = "/workspace/foundation-dagger/foundation-dagger/checkpoints/vpt/foundation-model-1x.weights"
    # model.load_state_dict(torch.load(path, map_location=device), strict=False)
    state_dict = torch.load(path, map_location=device)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        warn_lines = ["VPT checkpoint load was non-strict:"]
        if missing_keys:
            warn_lines.append(f"  missing keys ({len(missing_keys)}): {sorted(missing_keys)[:10]}{' ...' if len(missing_keys) > 10 else ''}")
        if unexpected_keys:
            warn_lines.append(f"  unexpected keys ({len(unexpected_keys)}): {sorted(unexpected_keys)[:10]}{' ...' if len(unexpected_keys) > 10 else ''}")
        print("\n".join(warn_lines))
    model = model.to(device)

    def _move_state_to_device(state):
        if state is None:
            return None
        if isinstance(state, (list, tuple)):
            return type(state)(_move_state_to_device(s) for s in state)
        return state.to(device, non_blocking=True)

    def _stack_states_for_batch(state_list):
        if not state_list or state_list[0] is None:
            return None
        if len(state_list) == 1:
            return state_list[0]
        return tree_multimap(lambda *xs: torch.cat(xs, dim=0), state_list[0], *state_list[1:])

    def _slice_state(state, index):
        if state is None:
            return None
        return tree_map(lambda x: x[index : index + 1].detach().clone(), state)

    def _detach_state(state):
        if state is None:
            return None
        return tree_map(lambda x: x.detach(), state)

    def save_checkpoint(suffix: str | None = None) -> Path:
        if suffix:
            checkpoint_path = output.with_name(f"{output.stem}_{suffix}{output.suffix}")
        else:
            checkpoint_path = output
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        wandb.save(str(checkpoint_path))
        return checkpoint_path

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(opt_cfg.get("lr", 3e-4)),
        betas=tuple(opt_cfg.get("betas", [0.9, 0.99])),
        weight_decay=float(opt_cfg.get("weight_decay", 1e-4)),
    )

    val_prefetch_factor = int(train_cfg.get("val_prefetch_factor", prefetch_factor))
    val_loader_kwargs = dict(
        batch_size=val_batch_size,
        num_workers=val_workers,
        pin_memory=True,
        shuffle=False,
        persistent_workers=False,  # Disabled to avoid stale VideoCapture objects across epochs
    )
    if val_workers > 0:
        val_loader_kwargs["prefetch_factor"] = max(val_prefetch_factor, 1)
    val_loader = DataLoader(val_subset, **val_loader_kwargs)

    def run_epoch(data_loader, train: bool):
        running_loss = 0.0
        running_examples = 0
        if train:
            model.train()
        else:
            model.eval()

        iterator = tqdm(data_loader, desc="train" if train else "val", leave=False, ncols=80)
        episode_hidden_states: Dict[int, Any] = {}
        last_seen_frame: Dict[int, int] = {}

        for batch in iterator:
            observations, _, video_ids, frame_indices = batch
            observations = observations.to(device, non_blocking=True).contiguous()
            video_ids_list = video_ids.cpu().tolist()
            frame_indices_list = frame_indices.cpu().tolist()
            agent_action_list = [
                dataset.get_agent_action(vid, frame_idx)
                for vid, frame_idx in zip(video_ids_list, frame_indices_list)
            ]
            action_keys = agent_action_list[0].keys()
            agent_actions = {
                key: torch.stack([action[key] for action in agent_action_list], dim=0).to(device, non_blocking=True)
                for key in action_keys
            }
            batch_size = observations.size(0)
            seq_len = observations.size(1) if observations.dim() >= 2 else 1
            with torch.set_grad_enabled(train):
                state_inputs = []
                first = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
                for idx, (vid, frame_idx) in enumerate(zip(video_ids_list, frame_indices_list)):
                    stored_state = episode_hidden_states.get(vid)
                    last_frame = last_seen_frame.get(vid)
                    needs_reset = stored_state is None or last_frame is None or frame_idx <= last_frame
                    if needs_reset:
                        stored_state = _move_state_to_device(model.initial_state(1))
                    if stored_state is not None:
                        episode_hidden_states[vid] = stored_state
                    state_inputs.append(stored_state)
                    if needs_reset:
                        first[idx, 0] = True
                state_in = _stack_states_for_batch(state_inputs)
                obs_dict = {"img": observations}
                policy_tuple, state_out = model(obs_dict, first, state_in)
                pd_params = policy_tuple[0]
                final_pd_params = tree_map(lambda x: x[:, -1], pd_params)
                log_prob = model.pi_head.logprob(agent_actions, final_pd_params)
                loss = -log_prob.mean()
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    optimizer.step()
                if state_out is not None:
                    detached_state = _detach_state(state_out)
                    for idx, vid in enumerate(video_ids_list):
                        episode_hidden_states[vid] = _slice_state(detached_state, idx)
                for vid, frame_idx in zip(video_ids_list, frame_indices_list):
                    last_seen_frame[vid] = frame_idx
            running_loss += (-log_prob.detach()).sum().item()
            running_examples += log_prob.numel()
            if running_examples > 0:
                iterator.set_postfix(
                    loss=f"{running_loss / running_examples:.4f}",
                )
        if running_examples == 0:
            return 0.0, {}
        return running_loss / running_examples, {}

    for epoch in range(train_epochs):
        train_batch_sampler.set_epoch(epoch)
        train_loss, _ = run_epoch(loader, train=True)
        if len(val_subset) > 0:
            val_loss, _ = run_epoch(val_loader, train=False)
        else:
            val_loss, _ = None, {}
        metrics = {
            "train/loss": train_loss,
        }
        if val_loss is not None:
            metrics["val/loss"] = val_loss
        wandb.log(metrics, step=epoch + 1)
        if val_loss is not None:
            print(
                f"Epoch {epoch + 1}/{epochs} - loss: {train_loss:.4f} "
                f"- val_loss: {val_loss:.4f}"
            )
        else:
            print(
                f"Epoch {epoch + 1}/{epochs} - loss: {train_loss:.4f}"
            )
        if checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0:
            ckpt_path = save_checkpoint(f"epoch{epoch + 1:03d}")
            print(f"Saved checkpoint to {ckpt_path}")

    final_path = save_checkpoint(None)
    wandb.finish()
    print(f"Saved checkpoint to {final_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algorithm",
        type=str,
        default="vpt_bc",
        help="Hydra algorithm config name.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mineworld_frames",
        help="Hydra dataset config name.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="mineworld_bc_train",
        help="Hydra experiment config name.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Directory containing MineWorld .mp4 videos and matching .jsonl logs. Overrides config if provided.",
    )
    parser.add_argument(
        "--fraction", 
        type=float, 
        default=1.0 / 6, 
        help="Dataset fraction to use"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=5, 
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=32, 
        help="Training batch size"
    )
    parser.add_argument(
        "--context-frames",
        type=int,
        default=None,
        help="Number of frames in the conditioning window. Overrides config if provided.",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=None,
        help="Spatial size to resize frames to. Overrides config if provided.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Disable recursive search for videos under the data root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs") / "bootstrap_bc.ckpt",
        help="Path to save the trained checkpoint.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    dataset_cfg, policy_cfg_dict, exp_cfg = _load_hydra_configs(
        dataset_name=args.dataset, algorithm_name=args.algorithm, experiment_name=args.experiment
    )
    algorithm_name = args.algorithm or policy_cfg_dict.get("name", "vpt_bc")
    dataset_name = args.dataset or dataset_cfg.get("name", "mineworld_frames")
    experiment_name = args.experiment or exp_cfg.get("name", "mineworld_bc_train")

    data_root_value = args.data_root or dataset_cfg.get("data_root")
    if data_root_value is None:
        raise ValueError("Dataset config must define data_root; alternatively pass --data-root.")
    data_root = (Path(data_root_value)
                 if Path(data_root_value).is_absolute()
                 else (ROOT_DIR / data_root_value).resolve())

    if args.fraction is not None:
        fraction = args.fraction
    else:
        fraction = float(dataset_cfg.get("fraction", 0.1))

    if args.epochs is not None:
        epochs = args.epochs
    else:
        epochs = int(dataset_cfg.get("epochs", 5))

    if args.batch_size is not None:
        batch_size = args.batch_size
    else:
        batch_size = int(dataset_cfg.get("batch_size", 32))

    if args.context_frames is not None:
        context_frames = args.context_frames
    else:
        context_frames = int(dataset_cfg.get("context_frames", 8))

    if args.resize is not None:
        resize = args.resize
    else:
        resize = int(dataset_cfg.get("resize", dataset_cfg.get("resolution", 256)))

    if args.output is not None:
        output = args.output
    else:
        output = dataset_cfg.get("output", "checkpoints/bc_policy.ckpt")

    recursive = dataset_cfg.get("recursive", True)
    if args.no_recursive:
        recursive = False

    train_vpt_bc(
        data_root=data_root,
        fraction=fraction,
        epochs=epochs,
        batch_size=batch_size,
        context_frames=context_frames,
        recursive=recursive,
        resize=resize,
        output=output,
        dataset_cfg=dataset_cfg,
        policy_cfg_dict=policy_cfg_dict,
        algorithm_name=algorithm_name,
        experiment_name=experiment_name,
        dataset_name=dataset_name,
        exp_cfg=exp_cfg,
    )
