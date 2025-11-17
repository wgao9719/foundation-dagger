"""Train an initial BC policy on MineWorld rollouts.

The script ingests MineWorld-style trajectories (paired ``.mp4`` videos and
``.jsonl`` action logs) and produces a supervised policy checkpoint compatible
with the ``foundation_dagger`` module. Each training sample stacks the full
``N``-frame context window and predicts the action token sequence associated
with the final frame, enabling Transformer-style policies to leverage temporal
attention while still supporting single-frame heads. Only a user-specified
fraction of the available samples is used to keep the bootstrap stage quick.

Example
-------
python scripts/train_initial_bc.py \
    --dataset mineworld_frames \
    --data-root /Users/willi1/foundation-dagger/diffusion-forcing-transformer/data/mineworld \
    --fraction 0.1 \
    --epochs 10 \
    --batch-size 32 \
    --output checkpoints/bc_policy.ckpt \
    --label-smoothing 0.1
"""


from __future__ import annotations

import argparse
import random
import sys
from copy import deepcopy
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from dataclasses import fields
from typing import Dict, List, Optional
from collections import Counter

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
import wandb
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from datasets.mineworld_data.mineworld_frame_dataset import MineWorldFrameDataset
from utils.SequentialBatchSampler import PerVideoSequentialBatchSampler

CONFIG_DIR = ROOT_DIR / "configurations"

from algorithms.foundation_dagger.policy import (
    ActionHeadConfig,
    PolicyConfig,
    FoundationBCPolicy,
)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

IGNORE_INDEX = -100

def _logprob_cross_entropy(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    smoothing: float = 0.0,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Cross-entropy between log-probabilities and integer targets with optional label smoothing.
    """
    if smoothing and smoothing > 0.0:
        n_classes = log_probs.size(-1)
        confidence = 1.0 - smoothing
        smooth = smoothing / max(n_classes - 1, 1)
        true_dist = torch.full_like(log_probs, smooth)
        true_dist.scatter_(-1, targets.unsqueeze(-1), confidence)
        loss = (-true_dist * log_probs).sum(dim=-1)
        if weights is not None:
            loss = loss * weights.gather(0, targets.view(-1)).view_as(loss)
        return loss.mean()
    nll = F.nll_loss(log_probs, targets, reduction="none")
    if weights is not None:
        nll = nll * weights.gather(0, targets.view(-1)).view_as(nll)
    return nll.mean()

def _squeeze_action_dim(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() >= 2 and tensor.shape[-2] == 1:
        return tensor.squeeze(-2)
    return tensor

def _entropy_from_logprobs(log_probs: torch.Tensor) -> torch.Tensor:
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy.mean()

def _estimate_button_class_weights(
    dataset: MineWorldFrameDataset,
    video_to_indices: Dict[int, List[int]],
    max_videos: int = 3,
    weight_temperature: float = 1.0,
) -> torch.Tensor:
    num_classes = len(dataset.action_mapper.BUTTONS_COMBINATIONS)
    weights = torch.ones(num_classes, dtype=torch.float32)
    if not video_to_indices:
        return weights
    selected_videos = sorted(video_to_indices.keys())[:max(1, max_videos)]
    counts: Counter[int] = Counter()
    for vid in selected_videos:
        indices = video_to_indices.get(int(vid), [])
        for dataset_idx in indices:
            _, sample_vid, end_idx = dataset.samples[dataset_idx]
            sample_vid = int(sample_vid)
            end_idx = int(end_idx)
            step_infos = dataset.video_step_infos.get(sample_vid)
            if step_infos is None or end_idx >= len(step_infos):
                continue
            frame_idx = int(step_infos[end_idx]["frame_idx"])
            agent_action = dataset.agent_actions.get((sample_vid, frame_idx))
            if not agent_action:
                continue
            buttons_value = agent_action.get("buttons")
            if buttons_value is None:
                continue
            button_idx = int(np.asarray(buttons_value).reshape(-1)[0])
            counts[button_idx] += 1
    if not counts:
        return weights
    mean_count = sum(counts.values()) / len(counts)
    for cls, freq in counts.items():
        scaled = float(mean_count / max(freq, 1))
        weights[cls] = scaled ** weight_temperature
    return weights

def _load_hydra_configs(
    dataset_name: str = "mineworld_frames",
    algorithm_name: str = "policy_bc",
    experiment_name: str = "mineworld_bc_train"
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
    policy_cfg = OmegaConf.to_container(cfg.algorithm.policy, resolve=True)
    exp_cfg = OmegaConf.to_container(cfg.experiment, resolve=True)
    return dataset_cfg, policy_cfg, exp_cfg

def _coerce_policy_config(raw_cfg: Dict, dataset: MineWorldFrameDataset) -> PolicyConfig:
    cfg = deepcopy(raw_cfg)
    valid_fields = {field.name for field in fields(PolicyConfig)}
    filtered_cfg = {key: value for key, value in cfg.items() if key in valid_fields}

    num_button_combos = len(dataset.action_mapper.BUTTONS_COMBINATIONS)
    num_camera_bins = dataset.action_mapper.n_camera_bins
    action_cfg = filtered_cfg.get("action")
    if isinstance(action_cfg, dict):
        action_cfg.setdefault("buttons_classes", num_button_combos)
        action_cfg.setdefault("camera_bins", num_camera_bins)
        filtered_cfg["action"] = ActionHeadConfig(**action_cfg)
    elif isinstance(action_cfg, ActionHeadConfig):
        pass
    else:
        filtered_cfg["action"] = ActionHeadConfig(
            buttons_classes=num_button_combos,
            camera_bins=num_camera_bins,
            use_camera_gate=True,
        )
    return PolicyConfig(**filtered_cfg)

def train_initial_bc(
    data_root: Path,
    fraction: float,
    epochs: int,
    batch_size: int,
    context_frames: int,
    recursive: bool,
    output: Path,
    dataset_cfg: Dict,
    policy_cfg_dict: Dict,
    algorithm_name: str,
    experiment_name: str,
    dataset_name: str,
    exp_cfg: Dict,
    label_smoothing: Optional[float],
) -> None:

    #initialize dataset
    dataset = MineWorldFrameDataset(
        data_root=data_root,
        context_frames=context_frames,
        recursive=recursive,
        max_open_captures=int(dataset_cfg.get("max_open_captures", 2))
    )
    policy_cfg = _coerce_policy_config(policy_cfg_dict, dataset)
    train_cfg = exp_cfg.get("training", {})
    opt_cfg = exp_cfg.get("optimizer", {})

    total_samples = len(dataset)
    if total_samples == 0:
        raise ValueError(f"No samples found under {data_root}")
    subset_len = max(1, int(total_samples * fraction))
    random_seed = int(train_cfg.get("seed", dataset_cfg.get("seed", 0)))
    random.seed(random_seed)

    # Shuffle at the video level while preserving chronological ordering per video.
    weight_estimation_videos = 3
    video_to_indices: Dict[int, List[int]] = {}
    for idx, (_, vid, _) in enumerate(dataset.samples):
        video_to_indices.setdefault(int(vid), []).append(idx)
    weight_temperature = float(train_cfg.get("button_weight_temperature", 0.5))
    button_weight_tensor = _estimate_button_class_weights(
        dataset,
        video_to_indices,
        max_videos=weight_estimation_videos,
        weight_temperature=weight_temperature,
    )
    print(f"Button weight tensor: {button_weight_tensor}")
    video_ids = list(video_to_indices.keys())
    if len(video_ids) < 2:
        raise ValueError(
            "Need at least two trajectories to reserve one for validation while keeping the rest for training."
        )
    random.shuffle(video_ids)

    # Reserve exactly one entire trajectory for validation; keep remaining
    # whole trajectories for training while respecting the fraction budget at
    # the granularity of complete videos.
    val_video_id = video_ids[0]
    train_candidate_videos = video_ids[1:]
    if not train_candidate_videos:
        raise ValueError(
            "Need at least two trajectories to create a train/val split that preserves full videos."
        )

    val_indices = list(video_to_indices[val_video_id])
    target_train_samples = max(0, subset_len - len(val_indices))
    limit_train_subset = fraction < 1.0
    train_indices: List[int] = []
    selected_train_videos: List[int] = []
    for vid in train_candidate_videos:
        if limit_train_subset and target_train_samples <= 0 and selected_train_videos:
            break
        train_indices.extend(video_to_indices[vid])
        selected_train_videos.append(vid)
        target_train_samples -= len(video_to_indices[vid])
    if not train_indices:
        # Fall back to the first remaining trajectory so training is non-empty.
        first_vid = train_candidate_videos[0]
        train_indices = list(video_to_indices[first_vid])
        selected_train_videos = [first_vid]

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    effective_total = max(1, len(train_indices) + len(val_indices))
    val_fraction = len(val_indices) / effective_total

    train_video_to_positions: Dict[int, List[int]] = {}
    for subset_idx, dataset_idx in enumerate(train_indices):
        _, vid, _ = dataset.samples[dataset_idx]
        train_video_to_positions.setdefault(int(vid), []).append(subset_idx)


    # load in training parameters
    train_batch_size = batch_size
    train_epochs = epochs
    train_workers = int(train_cfg.get("num_workers", 4))
    val_batch_size = int(train_cfg.get("val_batch_size", train_batch_size))
    val_workers = int(train_cfg.get("val_num_workers", max(1, train_workers // 2)))
    checkpoint_interval = int(train_cfg.get("checkpoint_interval", 1))
    camera_gate_weight = float(train_cfg.get("camera_gate_weight", 1.0))
    entropy_weight = float(train_cfg.get("entropy_weight", 0.0))
    grad_clip_norm = float(train_cfg.get("grad_clip_norm", 1.0))
    esc_loss_weight = float(train_cfg.get("esc_loss_weight", 1.0))

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
            "val_fraction": val_fraction,
            "seed": random_seed,
            "train_samples": len(train_subset),
            "val_samples": len(val_subset),
            "grad_clip_norm": grad_clip_norm if grad_clip_norm is not None else 0.0,
            "label_smoothing": label_smoothing,
            "camera_gate_weight": camera_gate_weight,
            "esc_loss_weight": esc_loss_weight,
            "entropy_weight": entropy_weight,
            "button_weight_videos": weight_estimation_videos,
            "button_weight_temperature": weight_temperature,
        },
    )

    # initialize data loader
    prefetch_factor = int(train_cfg.get("prefetch_factor", 2))
    train_batch_sampler = PerVideoSequentialBatchSampler(
        video_to_positions=train_video_to_positions,
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
    model = FoundationBCPolicy(policy_cfg)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    model = model.to(device)
    button_class_weights = button_weight_tensor.to(device)
    channel_mean = IMAGENET_MEAN.view(1, 1, 3, 1, 1).to(device)
    channel_std = IMAGENET_STD.view(1, 1, 3, 1, 1).to(device)
    camera_null_idx = dataset.action_mapper.camera_null_idx
    smoothing = float(label_smoothing if label_smoothing is not None else 0.0)

    def _prepare_observations(frames: torch.Tensor) -> torch.Tensor:
        frames = frames.permute(0, 1, 4, 2, 3).contiguous()
        frames = frames.div(255.0)
        frames = frames.to(device, non_blocking=True)
        return (frames - channel_mean) / channel_std

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
        running_action_correct = 0
        running_action_total = 0
        model.train(mode=train)
        sample_outputs: list[tuple[dict[str, list[int]], dict[str, list[int]]]] = []
        iterator = tqdm(data_loader, desc="train" if train else "val", leave=False)
        for batch in iterator:
            observations, _, video_ids, frame_indices = batch
            frames = _prepare_observations(observations)
            video_ids_list = video_ids.cpu().tolist()
            frame_indices_list = frame_indices.cpu().tolist()
            agent_action_list = [
                dataset.get_agent_action(vid, frame_idx)
                for vid, frame_idx in zip(video_ids_list, frame_indices_list)
            ]
            esc_targets = torch.tensor(
                [dataset.get_esc_flag(vid, frame_idx) for vid, frame_idx in zip(video_ids_list, frame_indices_list)],
                dtype=torch.long,
                device=device,
            )
            buttons_targets = (
                torch.stack([action["buttons"] for action in agent_action_list], dim=0)
                .to(device, non_blocking=True)
                .squeeze(-1)
                .long()
            )
            camera_targets = (
                torch.stack([action["camera"] for action in agent_action_list], dim=0)
                .to(device, non_blocking=True)
                .squeeze(-1)
                .long()
            )
            camera_gate_targets = (camera_targets != camera_null_idx).long()

            with torch.set_grad_enabled(train):
                logits = model(frames)
                last_logits = {key: _squeeze_action_dim(value[:, -1]) for key, value in logits.items()}
                buttons_loss = _logprob_cross_entropy(
                    last_logits["buttons"],
                    buttons_targets,
                    smoothing,
                    weights=button_class_weights,
                )
                camera_loss = _logprob_cross_entropy(last_logits["camera"], camera_targets, smoothing)
                entropy_term = _entropy_from_logprobs(last_logits["buttons"]) + _entropy_from_logprobs(last_logits["camera"])
                loss = buttons_loss + camera_loss - entropy_weight * entropy_term
                if "camera_gate" in last_logits:
                    gate_loss = F.cross_entropy(
                        last_logits["camera_gate"], camera_gate_targets, label_smoothing=smoothing
                    )
                    loss = loss + camera_gate_weight * gate_loss
                if "esc" in last_logits:
                    esc_loss = _logprob_cross_entropy(last_logits["esc"], esc_targets, smoothing)
                    loss = loss + esc_loss_weight * esc_loss
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    optimizer.step()

            batch_size = frames.size(0)
            running_loss += loss.detach().item() * batch_size
            running_examples += batch_size

            preds: dict[str, torch.Tensor] = {
                "buttons": last_logits["buttons"].argmax(dim=-1),
                "camera": last_logits["camera"].argmax(dim=-1),
            }
            targets: dict[str, torch.Tensor] = {
                "buttons": buttons_targets,
                "camera": camera_targets,
            }
            if "esc" in last_logits:
                preds["esc"] = last_logits["esc"].argmax(dim=-1)
                targets["esc"] = esc_targets
            if "camera_gate" in last_logits:
                preds["camera_gate"] = last_logits["camera_gate"].argmax(dim=-1)
                targets["camera_gate"] = camera_gate_targets

            for key, target_tensor in targets.items():
                pred_tensor = preds.get(key)
                if pred_tensor is None:
                    continue
                matches = pred_tensor == target_tensor
                running_action_correct += matches.sum().item()
                running_action_total += target_tensor.numel()

            if len(sample_outputs) < 3:
                take = min(batch_size, 3 - len(sample_outputs))
                for sample_idx in range(take):
                    target_snapshot = {
                        key: [int(targets[key][sample_idx].detach().cpu())]
                        for key in targets
                        if key in preds
                    }
                    pred_snapshot = {
                        key: [int(preds[key][sample_idx].detach().cpu())] for key in target_snapshot
                    }
                    sample_outputs.append((target_snapshot, pred_snapshot))
                    if len(sample_outputs) >= 3:
                        break

            if running_examples > 0:
                postfix = {"loss": f"{running_loss / running_examples:.3f}"}
                if running_action_total > 0:
                    postfix["acc"] = f"{running_action_correct / running_action_total:.3f}"
                iterator.set_postfix(postfix)

        if running_examples == 0:
            return 0.0, None, sample_outputs
        action_acc = None
        if running_action_total > 0:
            action_acc = running_action_correct / running_action_total
        return running_loss / running_examples, action_acc, sample_outputs

    for epoch in range(train_epochs):
        train_loss, train_acc, train_samples = run_epoch(loader, train=True)
        val_loss, val_acc, val_samples = run_epoch(val_loader, train=False)

        metrics = {
            "train/loss": train_loss,
        }
        if train_acc is not None:
            metrics["train/acc"] = train_acc
        if val_loss is not None:
            metrics["val/loss"] = val_loss
            if val_acc is not None:
                metrics["val/acc"] = val_acc
        wandb.log(metrics, step=epoch + 1)
        train_acc_str = f"{train_acc:.4f}" if train_acc is not None else "n/a"
        val_acc_str = f"{val_acc:.4f}" if val_acc is not None else "n/a"
        if val_loss is not None:
            print(
                f"Epoch {epoch + 1}/{epochs} - loss: {train_loss:.4f} - acc: {train_acc_str} "
                f"- val_loss: {val_loss:.4f} - val_acc: {val_acc_str} "
            )
            if train_samples:
                print("Training samples (target vs. prediction):")
                for idx, (target_tokens, pred_tokens) in enumerate(train_samples, start=1):
                    print(f"  Sample {idx}:")
                    for key in sorted(target_tokens.keys()):
                        target_vals = target_tokens[key]
                        pred_vals = pred_tokens.get(key, [])
                        print(f"    {key}: target={target_vals} pred={pred_vals}")
            if val_samples:
                print("Validation samples (target vs. prediction):")
                for idx, (target_tokens, pred_tokens) in enumerate(val_samples, start=1):
                    print(f"  Sample {idx}:")
                    for key in sorted(target_tokens.keys()):
                        target_vals = target_tokens[key]
                        pred_vals = pred_tokens.get(key, [])
                        print(f"    {key}: target={target_vals} pred={pred_vals}")
        else:
            print(
                f"Epoch {epoch + 1}/{epochs} - loss: {train_loss:.4f} - acc: {train_acc_str} "
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
        default="policy_bc",
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
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.1,
        help="Label smoothing coefficient for BC loss (overrides config).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset_cfg, policy_cfg_dict, exp_cfg = _load_hydra_configs(
        dataset_name=args.dataset, algorithm_name='policy_bc', experiment_name='mineworld_bc_train'
    )
    algorithm_name = args.algorithm or policy_cfg_dict.get("name", "policy_bc")
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

    if args.output is not None:
        output = args.output
    else:
        output = dataset_cfg.get("output", "checkpoints/bc_policy.ckpt")

    recursive = dataset_cfg.get("recursive", True)
    if args.no_recursive:
        recursive = False

    train_initial_bc(
        data_root=data_root,
        fraction=fraction,
        epochs=epochs,
        batch_size=batch_size,
        context_frames=context_frames,
        recursive=recursive,
        output=output,
        dataset_cfg=dataset_cfg,
        policy_cfg_dict=policy_cfg_dict,
        algorithm_name=algorithm_name,
        experiment_name=experiment_name,
        dataset_name=dataset_name,
        exp_cfg=exp_cfg,
        label_smoothing=args.label_smoothing,
    )
