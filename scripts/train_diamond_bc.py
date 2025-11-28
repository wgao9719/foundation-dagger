"""
Train BC policy on MineRL ObtainDiamond/ObtainIronPickaxe data.

This script trains a policy on the MineRL task_diamond format data which has:
- 64x64 video frames
- Explicit inventory observations
- Factored action space with binary buttons + categorical craft/smelt/equip actions

Example:
    python scripts/train_minerl_bc.py \
        --data-root data/task_diamond \
        --epochs 10 \
        --batch-size 64 \
        --output checkpoints/minerl_bc.ckpt
"""

from __future__ import annotations

import argparse
import random
import sys
import math
from pathlib import Path
from typing import Dict, List, Optional, Union
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torch.multiprocessing

try:
    torch.multiprocessing.set_sharing_strategy('file_system')
except RuntimeError:
    pass

from tqdm.auto import tqdm
import wandb

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from datasets.mineworld_data.diamond_dataset import (
    MineRLDiamondDataset,
    MineRLDiamondDecodedDataset,
    collate_minerl_batch,
)
from utils.VideoGroupedSampler import VideoGroupedSampler
from algorithms.foundation_dagger.diamond_policy import (
    MineRLPolicy,
    MineRLPolicyConfig,
)


# ImageNet normalization (optional, for pretrained backbones)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)


def _nll_loss_with_smoothing(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    smoothing: float = 0.0,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """NLL loss with optional label smoothing."""
    if smoothing > 0.0:
        n_classes = log_probs.size(-1)
        confidence = 1.0 - smoothing
        smooth_val = smoothing / max(n_classes - 1, 1)
        true_dist = torch.full_like(log_probs, smooth_val)
        true_dist.scatter_(-1, targets.unsqueeze(-1), confidence)
        loss = (-true_dist * log_probs).sum(dim=-1)
        if weight is not None:
            loss = loss * weight.gather(0, targets)
        return loss.mean()
    return F.nll_loss(log_probs, targets, weight=weight)


def compute_loss(
    logits: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    button_weight: float = 1.0,
    camera_weight: float = 1.0,
    categorical_weight: float = 1.0,
    label_smoothing: float = 0.0,
    class_weights: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """Compute multi-head BC loss with class weighting for all heads."""
    device = next(iter(logits.values())).device
    losses = {}
    
    action_buttons = batch["action_buttons"].to(device)
    action_camera = batch["action_camera"].to(device)
    
    def get_weight(name: str) -> Optional[torch.Tensor]:
        if class_weights is None:
            return None
        w = class_weights.get(name)
        return w.to(device) if w is not None else None
    
    # Binary button losses (with class weights)
    button_names = ["fwd", "left", "back", "right", "jump", "sneak", "sprint", "attack"]
    button_loss = 0.0
    for i, name in enumerate(button_names):
        log_probs = logits[f"button_{name}"][:, -1]
        targets = action_buttons[:, i]
        weight = get_weight(f"button_{name}")
        loss = _nll_loss_with_smoothing(log_probs, targets, label_smoothing, weight)
        losses[f"button_{name}"] = loss
        button_loss += loss
    losses["buttons_total"] = button_loss
    
    # Camera loss (with class weights)
    pitch_weight = get_weight("camera_pitch")
    yaw_weight = get_weight("camera_yaw")
    pitch_loss = _nll_loss_with_smoothing(
        logits["camera_pitch"][:, -1], action_camera[:, 0], label_smoothing, pitch_weight
    )
    yaw_loss = _nll_loss_with_smoothing(
        logits["camera_yaw"][:, -1], action_camera[:, 1], label_smoothing, yaw_weight
    )
    losses["camera_pitch"] = pitch_loss
    losses["camera_yaw"] = yaw_loss
    losses["camera_total"] = pitch_loss + yaw_loss
    
    # Categorical action losses (with class weights)
    categorical_names = ["place", "equip", "craft", "nearby_craft", "nearby_smelt"]
    categorical_loss = 0.0
    for name in categorical_names:
        log_probs = logits[name][:, -1]
        targets = batch[f"action_{name}"].to(device)
        weight = get_weight(name)
        loss = _nll_loss_with_smoothing(log_probs, targets, label_smoothing, weight)
        losses[name] = loss
        categorical_loss += loss
    losses["categorical_total"] = categorical_loss
    
    losses["total"] = (
        button_weight * button_loss +
        camera_weight * losses["camera_total"] +
        categorical_weight * categorical_loss
    )
    return losses


def compute_accuracy(
    logits: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """Compute accuracy metrics for each action head."""
    device = next(iter(logits.values())).device
    accuracies = {}
    
    action_buttons = batch["action_buttons"].to(device)
    action_camera = batch["action_camera"].to(device)
    
    # Button accuracy
    # Note: "fwd" is used instead of "forward" to avoid nn.Module conflict
    button_names = ["fwd", "left", "back", "right", "jump", "sneak", "sprint", "attack"]
    button_correct = 0
    button_total = 0
    for i, name in enumerate(button_names):
        preds = logits[f"button_{name}"][:, -1].argmax(dim=-1)
        targets = action_buttons[:, i]
        correct = (preds == targets).sum().item()
        total = targets.numel()
        accuracies[f"button_{name}"] = correct / total
        button_correct += correct
        button_total += total
    accuracies["buttons_avg"] = button_correct / button_total
    
    # Camera accuracy
    pitch_preds = logits["camera_pitch"][:, -1].argmax(dim=-1)
    yaw_preds = logits["camera_yaw"][:, -1].argmax(dim=-1)
    pitch_correct = (pitch_preds == action_camera[:, 0]).float().mean().item()
    yaw_correct = (yaw_preds == action_camera[:, 1]).float().mean().item()
    accuracies["camera_pitch"] = pitch_correct
    accuracies["camera_yaw"] = yaw_correct
    accuracies["camera_avg"] = (pitch_correct + yaw_correct) / 2
    
    # Categorical accuracy
    categorical_names = ["place", "equip", "craft", "nearby_craft", "nearby_smelt"]
    for name in categorical_names:
        preds = logits[name][:, -1].argmax(dim=-1)
        targets = batch[f"action_{name}"].to(device)
        accuracies[name] = (preds == targets).float().mean().item()
    
    return accuracies


def estimate_class_weights_fast(
    dataset,
    sample_indices: List[int],
    temperature: float = 0.5,
    n_camera_bins: int = 11,
) -> Dict[str, torch.Tensor]:
    """
    Fast class weight estimation for ALL action heads.
    
    Includes buttons (8), camera (2), and categorical (5) heads.
    """
    action_space = dataset.get_action_space_info()
    traj_data = getattr(dataset, '_trajectory_data', dataset._trajectory_data)
    
    # Button names
    button_names = ["fwd", "left", "back", "right", "jump", "sneak", "sprint", "attack"]
    
    # Initialize counters
    button_counts = {name: Counter() for name in button_names}
    camera_pitch_counts = Counter()
    camera_yaw_counts = Counter()
    place_counts = Counter()
    equip_counts = Counter()
    craft_counts = Counter()
    nearby_craft_counts = Counter()
    nearby_smelt_counts = Counter()
    
    # Sample from pre-loaded actions
    sampled = random.sample(sample_indices, min(10000, len(sample_indices)))
    for idx in sampled:
        video_id, frame_idx = dataset.samples[idx]
        actions = traj_data[video_id]["actions"]
        
        # Buttons (binary_buttons is [T, 8])
        buttons = actions["binary_buttons"][frame_idx]
        for i, name in enumerate(button_names):
            button_counts[name][int(buttons[i])] += 1
        
        # Camera
        camera = actions["camera"][frame_idx]
        camera_pitch_counts[int(camera[0])] += 1
        camera_yaw_counts[int(camera[1])] += 1
        
        # Categorical
        place_counts[int(actions["place"][frame_idx])] += 1
        equip_counts[int(actions["equip"][frame_idx])] += 1
        craft_counts[int(actions["craft"][frame_idx])] += 1
        nearby_craft_counts[int(actions["nearby_craft"][frame_idx])] += 1
        nearby_smelt_counts[int(actions["nearby_smelt"][frame_idx])] += 1
    
    def counts_to_weights(counts: Counter, num_classes: int) -> torch.Tensor:
        total = sum(counts.values())
        weights = torch.ones(num_classes)
        if total == 0:
            return weights
        mean_count = total / num_classes
        for cls, count in counts.items():
            if count > 0:
                raw = (mean_count / count) ** temperature
                # Asymmetric scaling:
                def squash_tail(raw: float, k: float = 33.0) -> float:
                    return raw / (1.0 + raw / k)  # 2→~2, 5→~4.3, 100→~24.8

                if raw >= 1.0:
                    weights[cls] = squash_tail(raw, k=33.0)
                else:
                    weights[cls] = max(0.15, 1.0 + 0.5 * math.log(raw))
        weights = weights * (num_classes / weights.sum())
        return weights
    
    weights = {}
    
    # Button weights (each is binary: 2 classes)
    for name in button_names:
        weights[f"button_{name}"] = counts_to_weights(button_counts[name], 2)
    
    # Camera weights
    weights["camera_pitch"] = counts_to_weights(camera_pitch_counts, n_camera_bins)
    weights["camera_yaw"] = counts_to_weights(camera_yaw_counts, n_camera_bins)
    # weights["camera_pitch"] = torch.ones(n_camera_bins)
    # weights["camera_yaw"] = torch.ones(n_camera_bins)
    
    # Categorical weights
    weights["place"] = counts_to_weights(place_counts, action_space["num_place_classes"])
    weights["equip"] = counts_to_weights(equip_counts, action_space["num_equip_classes"])
    weights["craft"] = counts_to_weights(craft_counts, action_space["num_craft_classes"])
    weights["nearby_craft"] = counts_to_weights(nearby_craft_counts, action_space["num_nearby_craft_classes"])
    weights["nearby_smelt"] = counts_to_weights(nearby_smelt_counts, action_space["num_nearby_smelt_classes"])
    
    return weights


def train(
    data_root: Path,
    output: Path,
    epochs: int = 10,
    batch_size: int = 64,
    context_frames: int = 8,
    n_camera_bins: int = 11,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    grad_clip_norm: float = 1.0,
    label_smoothing: float = 0.0,
    train_fraction: float = 1.0,
    val_fraction: float = 0.1,
    num_workers: int = 4,
    warmup_steps: int = 1000,
    backbone: str = "small_cnn",
    use_temporal_encoder: bool = False,
    button_weight: float = 1.0,
    camera_weight: float = 1.0,
    categorical_weight: float = 2.0,
    class_weight_temperature: float = 0.5,
    seed: int = 42,
    checkpoint_interval: int = 1,
    wandb_project: str = "diamond-bc",
    use_decoded: bool = True,
    sampler_chunk_size: int = 512,
    prefetch_factor: int = 4,
    checkpoint_path: Optional[str] = None,
    cifar_stem: Union[bool, str] = False,
) -> None:
    """Main training function."""
    
    # Set seeds
    random.seed(seed)
    torch.manual_seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Build dataset
    print(f"Loading dataset from {data_root}...")
    if use_decoded:
        dataset = MineRLDiamondDecodedDataset(
            decoded_root=data_root,
            context_frames=context_frames,
            n_camera_bins=n_camera_bins,
            skip_null_actions=True,
        )
    else:
        dataset = MineRLDiamondDataset(
            data_root=data_root,
            context_frames=context_frames,
            n_camera_bins=n_camera_bins,
            skip_null_actions=True,
        )
    print(f"Total samples: {len(dataset)}")
    
    # Train/val split by trajectory
    # Group samples by video_id
    video_to_indices: Dict[int, List[int]] = {}
    train_samples = dataset.samples[:int(len(dataset) * train_fraction)]
    for idx, (video_id, _) in enumerate(train_samples):
        video_to_indices.setdefault(video_id, []).append(idx)
    
    video_ids = list(video_to_indices.keys())
    random.shuffle(video_ids)
    
    # Split videos
    n_val_videos = max(1, int(len(video_ids) * val_fraction))
    val_video_ids = set(video_ids[:n_val_videos])
    train_video_ids = set(video_ids[n_val_videos:])
    
    train_indices = []
    val_indices = []
    for vid, indices in video_to_indices.items():
        if vid in val_video_ids:
            val_indices.extend(indices)
        else:
            train_indices.extend(indices)
    
    print(f"Train samples: {len(train_indices)} ({len(train_video_ids)} videos)")
    print(f"Val samples: {len(val_indices)} ({len(val_video_ids)} videos)")
    
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    
    # Estimate class weights for ALL action heads (fast - no video decoding)
    print("Estimating class weights for all action heads...")
    class_weights = estimate_class_weights_fast(
        dataset,
        sample_indices=train_indices,
        temperature=class_weight_temperature,
        n_camera_bins=n_camera_bins,
    )
    print("Class weights:")
    for name, weights in sorted(class_weights.items()):
        print(f"  {name}: {[f'{w:.2f}' for w in weights.tolist()]}")
    
    # Build VideoGroupedSampler for cache-friendly loading
    # Map train_indices to (None, video_id, subset_idx) format
    subset_samples = [
        (None, dataset.samples[train_indices[i]][0], i)
        for i in range(len(train_indices))
    ]
    train_sampler = VideoGroupedSampler(
        samples=subset_samples,
        chunk_size=sampler_chunk_size,
        seed=seed,
        shuffle_videos=True,
        shuffle_within_video=True,
    )
    print(f"Using VideoGroupedSampler with chunk_size={sampler_chunk_size}")
    
    # Data loaders with locality-aware sampling
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_minerl_batch,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        collate_fn=collate_minerl_batch,
        pin_memory=True,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    
    # Build model
    action_space_info = dataset.get_action_space_info()
    cfg = MineRLPolicyConfig(
        backbone=backbone,
        vision_dim=256,  # Could make configurable
        n_camera_bins=n_camera_bins,
        use_temporal_encoder=use_temporal_encoder,
        cifar_stem=cifar_stem,
    )
    
    # Checkpoint loading
    start_epoch = 0
    model_state = None
    optimizer_state = None
    
    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_state = checkpoint.get("model_state_dict", checkpoint)
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"]
            print(f"Resuming from epoch {start_epoch}")
        
        # If loading full checkpoint with config, verify consistency
        if "config" in checkpoint:
            ckpt_cfg = checkpoint["config"]
            if isinstance(ckpt_cfg, dict):
                # Basic check - warn if major mismatch
                if ckpt_cfg.get("n_camera_bins") != n_camera_bins:
                    print(f"WARNING: Checkpoint n_camera_bins ({ckpt_cfg.get('n_camera_bins')}) != current ({n_camera_bins})")
    
    model = MineRLPolicy(
        cfg=cfg,
        action_space_info=action_space_info,
        num_inventory_items=dataset.get_inventory_dim(),
        num_equipped_types=dataset.get_num_equipped_types(),
    )
    
    if model_state:
        # Allow loading partial state (e.g. if resizing vocab)
        model.load_state_dict(model_state, strict=False)
        print("Model weights loaded.")
    
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    
    if optimizer_state:
        optimizer.load_state_dict(optimizer_state)
        print("Optimizer state loaded.")
    
    total_steps = len(train_loader) * epochs
    warmup_steps = min(warmup_steps, total_steps//10)

    base_lr = optimizer.param_groups[0]["lr"]
    min_ratio = 1e-5 / base_lr if base_lr > 0 else 0.0

    def _cosine_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if total_steps <= warmup_steps:
            return 1.0
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_cosine_lambda)
    
    # If resuming, step scheduler to current epoch
    if start_epoch > 0:
        print(f"Fast-forwarding scheduler for {start_epoch} epochs...")
        steps_to_skip = start_epoch * len(train_loader)
        for _ in range(steps_to_skip):
            scheduler.step()
    
    # Initialize wandb
    wandb.init(
        project=wandb_project,
        config={
            "data_root": str(data_root),
            "epochs": epochs,
            "batch_size": batch_size,
            "context_frames": context_frames,
            "n_camera_bins": n_camera_bins,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "warmup_steps": warmup_steps,
            "backbone": backbone,
            "use_temporal_encoder": use_temporal_encoder,
            "button_weight": button_weight,
            "camera_weight": camera_weight,
            "categorical_weight": categorical_weight,
            "train_samples": len(train_indices),
            "val_samples": len(val_indices),
            "num_params": num_params,
            "cifar_stem": cifar_stem,
            "resumed_from": checkpoint_path if checkpoint_path else "scratch",
        },
    )
    
    # Training loop
    global_step = start_epoch * len(train_loader)
    best_val_loss = float("inf")
    
    for epoch in range(start_epoch, epochs):
        # Update sampler epoch for reproducible shuffling
        train_sampler.set_epoch(epoch)
        
        # Training
        model.train()
        train_losses = []
        train_accuracies = []
        train_samples = []  # Collect sample predictions
        # Distribution tracking (lightweight - just counts)
        # Model uses "fwd" instead of "forward" to avoid nn.Module.forward conflict
        btn_names_full = ["fwd", "left", "back", "right", "jump", "sneak", "sprint", "attack"]
        train_dist = {"tgt_btn": {n: [0,0] for n in btn_names_full}, "pred_btn": {n: [0,0] for n in btn_names_full},
                      "tgt_cam_p": [0]*n_camera_bins, "pred_cam_p": [0]*n_camera_bins,
                      "tgt_cam_y": [0]*n_camera_bins, "pred_cam_y": [0]*n_camera_bins, "count": 0}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]")
        for batch in pbar:
            # Forward pass
            frames = batch["frames"].to(device)
            inventory = batch["inventory"].to(device)
            equipped_type = batch["equipped_type"].to(device)
            
            logits = model(frames, inventory, equipped_type)
            
            # Compute loss
            losses = compute_loss(
                logits, batch,
                button_weight=button_weight,
                camera_weight=camera_weight,
                categorical_weight=categorical_weight,
                label_smoothing=label_smoothing,
                class_weights=class_weights,
            )
            
            # Backward pass
            optimizer.zero_grad()
            losses["total"].backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            scheduler.step()
            
            # Logging
            train_losses.append(losses["total"].item())
            accuracies = compute_accuracy(logits, batch)
            train_accuracies.append(accuracies)
            
            # Accumulate distribution (up to 10000 samples)
            if train_dist["count"] < 10000:
                bs = min(batch["frames"].shape[0], 10000 - train_dist["count"])
                with torch.no_grad():
                    for i, n in enumerate(btn_names_full):
                        for v in batch["action_buttons"][:bs, i].tolist():
                            train_dist["tgt_btn"][n][v] += 1
                        for v in logits[f"button_{n}"][:bs, -1].argmax(-1).cpu().tolist():
                            train_dist["pred_btn"][n][v] += 1
                    for v in batch["action_camera"][:bs, 0].tolist():
                        train_dist["tgt_cam_p"][v] += 1
                    for v in batch["action_camera"][:bs, 1].tolist():
                        train_dist["tgt_cam_y"][v] += 1
                    for v in logits["camera_pitch"][:bs, -1].argmax(-1).cpu().tolist():
                        train_dist["pred_cam_p"][v] += 1
                    for v in logits["camera_yaw"][:bs, -1].argmax(-1).cpu().tolist():
                        train_dist["pred_cam_y"][v] += 1
                train_dist["count"] += bs
            
            # Collect sample predictions (first 5)
            if len(train_samples) < 5:
                with torch.no_grad():
                    btn_names = ["fwd", "left", "back", "right", "jump", "sneak", "sprint", "attack"]
                    preds = {f"btn_{n}": logits[f"button_{n}"][:, -1].argmax(-1)[0].item() for n in btn_names}
                    preds["cam_p"] = logits["camera_pitch"][:, -1].argmax(-1)[0].item()
                    preds["cam_y"] = logits["camera_yaw"][:, -1].argmax(-1)[0].item()
                    preds["craft"] = logits["craft"][:, -1].argmax(-1)[0].item()
                    
                    tgts = {f"btn_{n}": batch["action_buttons"][0, i].item() for i, n in enumerate(btn_names)}
                    tgts["cam_p"] = batch["action_camera"][0, 0].item()
                    tgts["cam_y"] = batch["action_camera"][0, 1].item()
                    tgts["craft"] = batch["action_craft"][0].item()
                    train_samples.append((tgts, preds))
            
            running_loss = sum(train_losses) / len(train_losses)
            pbar.set_postfix({
                "loss": f"{losses['total'].item():.3f}",
                "run_loss": f"{running_loss:.3f}",
                "btn_acc": f"{accuracies['buttons_avg']:.3f}",
                "cam_acc": f"{accuracies['camera_avg']:.3f}",
            })
            
            if global_step % 100 == 0:
                # Compute running averages
                running_btn_acc = sum(a["buttons_avg"] for a in train_accuracies) / len(train_accuracies)
                running_cam_acc = sum(a["camera_avg"] for a in train_accuracies) / len(train_accuracies)
                wandb.log({
                    "train/loss": losses["total"].item(),
                    "train/run_loss": running_loss,
                    "train/loss_buttons": losses["buttons_total"].item(),
                    "train/loss_camera": losses["camera_total"].item(),
                    "train/loss_categorical": losses["categorical_total"].item(),
                    "train/acc_buttons": accuracies["buttons_avg"],
                    "train/acc_camera": accuracies["camera_avg"],
                    "train/run_acc_buttons": running_btn_acc,
                    "train/run_acc_camera": running_cam_acc,
                    "train/lr": scheduler.get_last_lr()[0],
                    "step": global_step,
                })
            
            global_step += 1
        
        # Validation
        model.eval()
        val_losses = []
        val_accuracies = []
        val_samples = []
        val_dist = {"tgt_btn": {n: [0,0] for n in btn_names_full}, "pred_btn": {n: [0,0] for n in btn_names_full},
                    "tgt_cam_p": [0]*n_camera_bins, "pred_cam_p": [0]*n_camera_bins,
                    "tgt_cam_y": [0]*n_camera_bins, "pred_cam_y": [0]*n_camera_bins, "count": 0}
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [val]", leave=False):
                frames = batch["frames"].to(device)
                inventory = batch["inventory"].to(device)
                equipped_type = batch["equipped_type"].to(device)
                
                logits = model(frames, inventory, equipped_type)
                losses = compute_loss(
                    logits, batch,
                    button_weight=button_weight,
                    camera_weight=camera_weight,
                    categorical_weight=categorical_weight,
                    class_weights=class_weights,
                )
                
                val_losses.append(losses["total"].item())
                val_accuracies.append(compute_accuracy(logits, batch))
                
                # Accumulate distribution
                if val_dist["count"] < 10000:
                    bs = min(batch["frames"].shape[0], 10000 - val_dist["count"])
                    for i, n in enumerate(btn_names_full):
                        for v in batch["action_buttons"][:bs, i].tolist():
                            val_dist["tgt_btn"][n][v] += 1
                        for v in logits[f"button_{n}"][:bs, -1].argmax(-1).cpu().tolist():
                            val_dist["pred_btn"][n][v] += 1
                    for v in batch["action_camera"][:bs, 0].tolist():
                        val_dist["tgt_cam_p"][v] += 1
                    for v in batch["action_camera"][:bs, 1].tolist():
                        val_dist["tgt_cam_y"][v] += 1
                    for v in logits["camera_pitch"][:bs, -1].argmax(-1).cpu().tolist():
                        val_dist["pred_cam_p"][v] += 1
                    for v in logits["camera_yaw"][:bs, -1].argmax(-1).cpu().tolist():
                        val_dist["pred_cam_y"][v] += 1
                    val_dist["count"] += bs
                
                # Collect sample predictions (first 5)
                if len(val_samples) < 5:
                    btn_names = ["fwd", "left", "back", "right", "jump", "sneak", "sprint", "attack"]
                    preds = {f"btn_{n}": logits[f"button_{n}"][:, -1].argmax(-1)[0].item() for n in btn_names}
                    preds["cam_p"] = logits["camera_pitch"][:, -1].argmax(-1)[0].item()
                    preds["cam_y"] = logits["camera_yaw"][:, -1].argmax(-1)[0].item()
                    preds["craft"] = logits["craft"][:, -1].argmax(-1)[0].item()
                    
                    tgts = {f"btn_{n}": batch["action_buttons"][0, i].item() for i, n in enumerate(btn_names)}
                    tgts["cam_p"] = batch["action_camera"][0, 0].item()
                    tgts["cam_y"] = batch["action_camera"][0, 1].item()
                    tgts["craft"] = batch["action_craft"][0].item()
                    val_samples.append((tgts, preds))
        
        # Aggregate metrics
        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_val_loss = sum(val_losses) / len(val_losses)
        
        avg_train_acc = {
            key: sum(a[key] for a in train_accuracies) / len(train_accuracies)
            for key in train_accuracies[0]
        }
        avg_val_acc = {
            key: sum(a[key] for a in val_accuracies) / len(val_accuracies)
            for key in val_accuracies[0]
        }
        
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train loss: {avg_train_loss:.4f}, buttons_acc: {avg_train_acc['buttons_avg']:.4f}, camera_acc: {avg_train_acc['camera_avg']:.4f}")
        print(f"  Val loss: {avg_val_loss:.4f}, buttons_acc: {avg_val_acc['buttons_avg']:.4f}, camera_acc: {avg_val_acc['camera_avg']:.4f}")
        
        # Print accumulated distributions
        def print_dist(dist, name):
            total = dist["count"]
            if total == 0:
                return
            print(f"  {name} dist (n={total}): Buttons(tgt/pred): ", end="")
            for n in btn_names_full:
                t1 = dist["tgt_btn"][n][1] / total * 100
                p1 = dist["pred_btn"][n][1] / total * 100
                print(f"{n[:3]}:{t1:.0f}/{p1:.0f} ", end="")
            print()
            print(f"    CamP: ", end="")
            for b in range(n_camera_bins):
                tp = dist["tgt_cam_p"][b] / total * 100
                pp = dist["pred_cam_p"][b] / total * 100
                print(f"b{b}:{tp:.0f}/{pp:.0f} ", end="")
            print(f" | CamY: ", end="")
            for b in range(n_camera_bins):
                ty = dist["tgt_cam_y"][b] / total * 100
                py = dist["pred_cam_y"][b] / total * 100
                print(f"b{b}:{ty:.0f}/{py:.0f} ", end="")
            print()
        print_dist(train_dist, "Train")
        print_dist(val_dist, "Val")
        
        wandb.log({
            "epoch": epoch + 1,
            "train/epoch_loss": avg_train_loss,
            "val/epoch_loss": avg_val_loss,
            "val/acc_buttons": avg_val_acc["buttons_avg"],
            "val/acc_camera": avg_val_acc["camera_avg"],
            "val/acc_place": avg_val_acc["place"],
            "val/acc_equip": avg_val_acc["equip"],
            "val/acc_craft": avg_val_acc["craft"],
        })
        
        # Save checkpoint
        if checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0:
            ckpt_path = output.with_name(f"{output.stem}_epoch{epoch+1:03d}{output.suffix}")
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg,
                "action_space_info": action_space_info,
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")
            wandb.save(str(ckpt_path))
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = output.with_name(f"{output.stem}_best{output.suffix}")
            best_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "config": cfg,
                "action_space_info": action_space_info,
                "val_loss": best_val_loss,
            }, best_path)
            print(f"  New best model saved: {best_path}")
    
    # Save final model
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "action_space_info": action_space_info,
    }, output)
    print(f"\nFinal model saved: {output}")
    
    wandb.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/task_diamond"),
        help="Root directory containing MineRL trajectory folders",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints/minerl_bc.ckpt"),
        help="Output checkpoint path",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--context-frames", type=int, default=8)
    parser.add_argument("--n-camera-bins", type=int, default=11)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument(
        "--backbone",
        type=str,
        default="small_cnn",
        choices=["small_cnn", "resnet18", "resnet34", "resnet50"],
    )
    parser.add_argument("--use-temporal-encoder", action="store_true")
    parser.add_argument("--button-weight", type=float, default=1.0)
    parser.add_argument("--camera-weight", type=float, default=1.0)
    parser.add_argument("--categorical-weight", type=float, default=2.0)
    parser.add_argument("--class-weight-temperature", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--wandb-project", type=str, default="diamond-bc")
    parser.add_argument("--use-decoded", action="store_true",
                        help="Use pre-decoded tensor bundles from decode_diamond_frames.py")
    parser.add_argument("--sampler-chunk-size", type=int, default=512,
                        help="Chunk size for VideoGroupedSampler (larger = more locality)")
    parser.add_argument("--prefetch-factor", type=int, default=4,
                        help="DataLoader prefetch factor")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint to resume from")
    parser.add_argument("--cifar-stem", type=str, default="full",
                        choices=["full", "half", False],
                        help="CIFAR stem for ResNet backbone")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        data_root=args.data_root,
        output=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        context_frames=args.context_frames,
        n_camera_bins=args.n_camera_bins,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        label_smoothing=args.label_smoothing,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
        warmup_steps=args.warmup_steps,
        backbone=args.backbone,
        use_temporal_encoder=args.use_temporal_encoder,
        button_weight=args.button_weight,
        camera_weight=args.camera_weight,
        categorical_weight=args.categorical_weight,
        class_weight_temperature=args.class_weight_temperature,
        seed=args.seed,
        checkpoint_interval=args.checkpoint_interval,
        wandb_project=args.wandb_project,
        use_decoded=args.use_decoded,
        sampler_chunk_size=args.sampler_chunk_size,
        prefetch_factor=args.prefetch_factor,
        checkpoint_path=args.checkpoint,
        cifar_stem=args.cifar_stem,
    )

