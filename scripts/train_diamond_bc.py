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
from typing import Dict, List, Optional
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
    collate_minerl_batch,
)
from algorithms.foundation_dagger.diamond_policy import (
    MineRLPolicy,
    MineRLPolicyConfig,
)


# ImageNet normalization (optional, for pretrained backbones)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)


def compute_loss(
    logits: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    button_weight: float = 1.0,
    camera_weight: float = 1.0,
    categorical_weight: float = 1.0,
    label_smoothing: float = 0.0,
    rare_action_weights: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute multi-head BC loss.
    
    Returns dict with individual losses and total loss.
    """
    device = next(iter(logits.values())).device
    losses = {}
    
    # Move targets to device
    action_buttons = batch["action_buttons"].to(device)  # [B, 8]
    action_camera = batch["action_camera"].to(device)    # [B, 2]
    
    # Binary button losses (8 heads)
    # Note: "fwd" is used instead of "forward" to avoid nn.Module conflict
    button_names = ["fwd", "left", "back", "right", "jump", "sneak", "sprint", "attack"]
    button_loss = 0.0
    for i, name in enumerate(button_names):
        log_probs = logits[f"button_{name}"][:, -1]  # [B, 2] - last timestep
        targets = action_buttons[:, i]  # [B]
        loss = F.nll_loss(log_probs, targets, label_smoothing=label_smoothing)
        losses[f"button_{name}"] = loss
        button_loss += loss
    losses["buttons_total"] = button_loss
    
    # Camera loss (2 axes)
    camera_pitch_probs = logits["camera_pitch"][:, -1]  # [B, n_bins]
    camera_yaw_probs = logits["camera_yaw"][:, -1]      # [B, n_bins]
    camera_pitch_target = action_camera[:, 0]
    camera_yaw_target = action_camera[:, 1]
    
    pitch_loss = F.nll_loss(camera_pitch_probs, camera_pitch_target)
    yaw_loss = F.nll_loss(camera_yaw_probs, camera_yaw_target)
    losses["camera_pitch"] = pitch_loss
    losses["camera_yaw"] = yaw_loss
    losses["camera_total"] = pitch_loss + yaw_loss
    
    # Categorical action losses
    categorical_names = ["place", "equip", "craft", "nearby_craft", "nearby_smelt"]
    categorical_loss = 0.0
    for name in categorical_names:
        log_probs = logits[name][:, -1]  # [B, num_classes]
        targets = batch[f"action_{name}"].to(device)  # [B]
        
        # Apply class weighting for rare actions
        weight = None
        if rare_action_weights is not None and name in rare_action_weights:
            weight = rare_action_weights[name].to(device)
        
        loss = F.nll_loss(log_probs, targets, weight=weight)
        losses[name] = loss
        categorical_loss += loss
    losses["categorical_total"] = categorical_loss
    
    # Total loss
    total = (
        button_weight * button_loss +
        camera_weight * losses["camera_total"] +
        categorical_weight * categorical_loss
    )
    losses["total"] = total
    
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


def estimate_class_weights(
    dataset: MineRLDiamondDataset,
    num_samples: int = 10000,
    temperature: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """
    Estimate class weights for rare categorical actions.
    
    Uses inverse frequency weighting with temperature smoothing.
    """
    action_space = dataset.get_action_space_info()
    
    # Sample subset for frequency estimation
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    # Count frequencies
    place_counts = Counter()
    equip_counts = Counter()
    craft_counts = Counter()
    nearby_craft_counts = Counter()
    nearby_smelt_counts = Counter()
    
    for idx in tqdm(indices, desc="Estimating class weights", leave=False):
        sample = dataset[idx]
        place_counts[sample["action_place"].item()] += 1
        equip_counts[sample["action_equip"].item()] += 1
        craft_counts[sample["action_craft"].item()] += 1
        nearby_craft_counts[sample["action_nearby_craft"].item()] += 1
        nearby_smelt_counts[sample["action_nearby_smelt"].item()] += 1
    
    def counts_to_weights(counts: Counter, num_classes: int) -> torch.Tensor:
        total = sum(counts.values())
        weights = torch.ones(num_classes)
        mean_count = total / num_classes
        for cls, count in counts.items():
            if count > 0:
                weights[cls] = (mean_count / count) ** temperature
        return weights
    
    return {
        "place": counts_to_weights(place_counts, action_space["num_place_classes"]),
        "equip": counts_to_weights(equip_counts, action_space["num_equip_classes"]),
        "craft": counts_to_weights(craft_counts, action_space["num_craft_classes"]),
        "nearby_craft": counts_to_weights(nearby_craft_counts, action_space["num_nearby_craft_classes"]),
        "nearby_smelt": counts_to_weights(nearby_smelt_counts, action_space["num_nearby_smelt_classes"]),
    }


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
    val_fraction: float = 0.1,
    num_workers: int = 4,
    backbone: str = "small_cnn",
    use_temporal_encoder: bool = False,
    button_weight: float = 1.0,
    camera_weight: float = 1.0,
    categorical_weight: float = 2.0,  # Higher weight for rare actions
    class_weight_temperature: float = 0.5,
    seed: int = 42,
    checkpoint_interval: int = 1,
    wandb_project: str = "diamond-bc",
) -> None:
    """Main training function."""
    
    # Set seeds
    random.seed(seed)
    torch.manual_seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Build dataset
    print(f"Loading dataset from {data_root}...")
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
    for idx, (video_id, _) in enumerate(dataset.samples):
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
    
    # Estimate class weights for rare actions
    print("Estimating class weights for rare actions...")
    rare_action_weights = estimate_class_weights(
        dataset,
        num_samples=min(10000, len(train_indices)),
        temperature=class_weight_temperature,
    )
    print("Class weights:")
    for name, weights in rare_action_weights.items():
        print(f"  {name}: {weights.tolist()}")
    
    # Data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_minerl_batch,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        collate_fn=collate_minerl_batch,
        pin_memory=True,
    )
    
    # Build model
    action_space_info = dataset.get_action_space_info()
    cfg = MineRLPolicyConfig(
        backbone=backbone,
        vision_dim=256,
        n_camera_bins=n_camera_bins,
        use_temporal_encoder=use_temporal_encoder,
    )
    
    model = MineRLPolicy(
        cfg=cfg,
        action_space_info=action_space_info,
        num_inventory_items=dataset.get_inventory_dim(),
        num_equipped_types=dataset.get_num_equipped_types(),
    )
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    
    total_steps = len(train_loader) * epochs
    warmup_steps = min(1000, total_steps // 10)
    
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
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
            "backbone": backbone,
            "use_temporal_encoder": use_temporal_encoder,
            "button_weight": button_weight,
            "camera_weight": camera_weight,
            "categorical_weight": categorical_weight,
            "train_samples": len(train_indices),
            "val_samples": len(val_indices),
            "num_params": num_params,
        },
    )
    
    # Training loop
    global_step = 0
    best_val_loss = float("inf")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        train_accuracies = []
        
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
                rare_action_weights=rare_action_weights,
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
            
            pbar.set_postfix({
                "loss": f"{losses['total'].item():.3f}",
                "btn_acc": f"{accuracies['buttons_avg']:.3f}",
                "cam_acc": f"{accuracies['camera_avg']:.3f}",
            })
            
            if global_step % 100 == 0:
                wandb.log({
                    "train/loss": losses["total"].item(),
                    "train/loss_buttons": losses["buttons_total"].item(),
                    "train/loss_camera": losses["camera_total"].item(),
                    "train/loss_categorical": losses["categorical_total"].item(),
                    "train/acc_buttons": accuracies["buttons_avg"],
                    "train/acc_camera": accuracies["camera_avg"],
                    "train/lr": scheduler.get_last_lr()[0],
                    "step": global_step,
                })
            
            global_step += 1
        
        # Validation
        model.eval()
        val_losses = []
        val_accuracies = []
        
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
                )
                
                val_losses.append(losses["total"].item())
                val_accuracies.append(compute_accuracy(logits, batch))
        
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
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
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
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
        backbone=args.backbone,
        use_temporal_encoder=args.use_temporal_encoder,
        button_weight=args.button_weight,
        camera_weight=args.camera_weight,
        categorical_weight=args.categorical_weight,
        class_weight_temperature=args.class_weight_temperature,
        seed=args.seed,
        checkpoint_interval=args.checkpoint_interval,
        wandb_project=args.wandb_project,
    )

