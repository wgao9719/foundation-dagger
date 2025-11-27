"""Analyze action distribution in MineRL Diamond dataset."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from datasets.mineworld_data.diamond_dataset import (
    MineRLDiamondDataset,
    MineRLDiamondDecodedDataset,
)


def analyze_raw_camera(data_root: Path):
    """Analyze raw camera angles before binning."""
    print("\n" + "="*60)
    print("RAW CAMERA ANGLE ANALYSIS")
    print("="*60)
    
    all_pitch = []
    all_yaw = []
    
    # Find and load npz files
    for npz_path in data_root.rglob("rendered.npz"):
        try:
            npz_data = dict(np.load(npz_path, allow_pickle=True))
            camera = npz_data["action$camera"].astype(np.float32)
            all_pitch.extend(camera[:, 0].tolist())
            all_yaw.extend(camera[:, 1].tolist())
        except Exception as e:
            continue
    
    if not all_pitch:
        print("  No raw camera data found (use --data-root with raw data, not decoded)")
        return
    
    all_pitch = np.array(all_pitch)
    all_yaw = np.array(all_yaw)
    
    print(f"\n  Total frames: {len(all_pitch):,}")
    
    print(f"\n  PITCH (vertical):")
    print(f"    min: {all_pitch.min():.2f}°, max: {all_pitch.max():.2f}°")
    print(f"    mean: {all_pitch.mean():.2f}°, std: {all_pitch.std():.2f}°")
    print(f"    median: {np.median(all_pitch):.2f}°")
    for thresh in [1, 5, 10, 20, 45, 90]:
        pct = (np.abs(all_pitch) <= thresh).mean() * 100
        print(f"    |pitch| <= {thresh:2d}°: {pct:.1f}%")
    
    print(f"\n  YAW (horizontal):")
    print(f"    min: {all_yaw.min():.2f}°, max: {all_yaw.max():.2f}°")
    print(f"    mean: {all_yaw.mean():.2f}°, std: {all_yaw.std():.2f}°")
    print(f"    median: {np.median(all_yaw):.2f}°")
    for thresh in [1, 5, 10, 20, 45, 90]:
        pct = (np.abs(all_yaw) <= thresh).mean() * 100
        print(f"    |yaw| <= {thresh:2d}°: {pct:.1f}%")
    
    # Show what % would be clipped at various ranges
    print(f"\n  Clipping analysis (% of data outside range):")
    for max_angle in [10, 15, 20, 30, 45, 90, 180]:
        pitch_clipped = (np.abs(all_pitch) > max_angle).mean() * 100
        yaw_clipped = (np.abs(all_yaw) > max_angle).mean() * 100
        print(f"    ±{max_angle:3d}°: pitch={pitch_clipped:.2f}% clipped, yaw={yaw_clipped:.2f}% clipped")


def analyze(data_root: Path, use_decoded: bool, n_camera_bins: int):
    print(f"Loading dataset from {data_root}...")
    
    if use_decoded:
        dataset = MineRLDiamondDecodedDataset(
            decoded_root=data_root,
            context_frames=1,
            n_camera_bins=n_camera_bins,
            skip_null_actions=False,  # Include all actions for analysis
        )
    else:
        dataset = MineRLDiamondDataset(
            data_root=data_root,
            context_frames=1,
            n_camera_bins=n_camera_bins,
            skip_null_actions=False,
        )
    
    traj_data = dataset._trajectory_data
    
    # Initialize counters
    button_names = ["forward", "left", "back", "right", "jump", "sneak", "sprint", "attack"]
    button_counts = {name: Counter() for name in button_names}
    camera_pitch_counts = Counter()
    camera_yaw_counts = Counter()
    place_counts = Counter()
    equip_counts = Counter()
    craft_counts = Counter()
    nearby_craft_counts = Counter()
    nearby_smelt_counts = Counter()
    
    total_frames = 0
    
    # Count from all trajectory data
    for traj_id, data in traj_data.items():
        actions = data["actions"]
        n_frames = len(actions["binary_buttons"])
        total_frames += n_frames
        
        # Buttons
        for i, name in enumerate(button_names):
            for frame_idx in range(n_frames):
                button_counts[name][int(actions["binary_buttons"][frame_idx, i])] += 1
        
        # Camera
        for frame_idx in range(n_frames):
            camera_pitch_counts[int(actions["camera"][frame_idx, 0])] += 1
            camera_yaw_counts[int(actions["camera"][frame_idx, 1])] += 1
        
        # Categorical
        for frame_idx in range(n_frames):
            place_counts[int(actions["place"][frame_idx])] += 1
            equip_counts[int(actions["equip"][frame_idx])] += 1
            craft_counts[int(actions["craft"][frame_idx])] += 1
            nearby_craft_counts[int(actions["nearby_craft"][frame_idx])] += 1
            nearby_smelt_counts[int(actions["nearby_smelt"][frame_idx])] += 1
    
    print(f"\nTotal frames: {total_frames:,}")
    print(f"Trajectories: {len(traj_data)}")
    
    # Print button distributions
    print("\n" + "="*60)
    print("BUTTON DISTRIBUTIONS")
    print("="*60)
    for name in button_names:
        counts = button_counts[name]
        total = sum(counts.values())
        pct_0 = counts[0] / total * 100 if total > 0 else 0
        pct_1 = counts[1] / total * 100 if total > 0 else 0
        bar_1 = "█" * int(pct_1 / 2)
        print(f"  {name:10s}: 0={pct_0:5.1f}%  1={pct_1:5.1f}% {bar_1}")
    
    # Print camera distributions
    print("\n" + "="*60)
    print("CAMERA DISTRIBUTIONS")
    print("="*60)
    center_bin = n_camera_bins // 2
    
    print(f"\n  Pitch (bin {center_bin} = center):")
    total = sum(camera_pitch_counts.values())
    for bin_idx in range(n_camera_bins):
        count = camera_pitch_counts[bin_idx]
        pct = count / total * 100 if total > 0 else 0
        bar = "█" * int(pct / 2)
        marker = " <-- center" if bin_idx == center_bin else ""
        print(f"    bin {bin_idx:2d}: {pct:5.1f}% {bar}{marker}")
    
    print(f"\n  Yaw (bin {center_bin} = center):")
    total = sum(camera_yaw_counts.values())
    for bin_idx in range(n_camera_bins):
        count = camera_yaw_counts[bin_idx]
        pct = count / total * 100 if total > 0 else 0
        bar = "█" * int(pct / 2)
        marker = " <-- center" if bin_idx == center_bin else ""
        print(f"    bin {bin_idx:2d}: {pct:5.1f}% {bar}{marker}")
    
    # Print categorical distributions
    vocab = dataset.action_mapper.vocab
    
    print("\n" + "="*60)
    print("CATEGORICAL DISTRIBUTIONS")
    print("="*60)
    
    def print_categorical(name: str, counts: Counter, vocab_list: list):
        print(f"\n  {name}:")
        total = sum(counts.values())
        for idx, item in enumerate(vocab_list):
            count = counts[idx]
            pct = count / total * 100 if total > 0 else 0
            bar = "█" * int(pct / 2) if pct >= 1 else ("▏" if pct > 0 else "")
            print(f"    {idx} ({item:20s}): {pct:6.2f}% ({count:,}) {bar}")
    
    print_categorical("place", place_counts, vocab.PLACE_VOCAB)
    print_categorical("equip", equip_counts, vocab.EQUIP_VOCAB)
    print_categorical("craft", craft_counts, vocab.CRAFT_VOCAB)
    print_categorical("nearby_craft", nearby_craft_counts, vocab.NEARBY_CRAFT_VOCAB)
    print_categorical("nearby_smelt", nearby_smelt_counts, vocab.NEARBY_SMELT_VOCAB)
    
    # Summary stats
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    # Calculate null action percentage
    null_count = 0
    for traj_id, data in traj_data.items():
        actions = data["actions"]
        n_frames = len(actions["binary_buttons"])
        for frame_idx in range(n_frames):
            is_null = True
            # Check buttons
            if actions["binary_buttons"][frame_idx].sum() > 0:
                is_null = False
            # Check camera
            if actions["camera"][frame_idx, 0] != center_bin or actions["camera"][frame_idx, 1] != center_bin:
                is_null = False
            # Check categorical
            if actions["place"][frame_idx] != 0 or actions["equip"][frame_idx] != 0 or \
               actions["craft"][frame_idx] != 0 or actions["nearby_craft"][frame_idx] != 0 or \
               actions["nearby_smelt"][frame_idx] != 0:
                is_null = False
            if is_null:
                null_count += 1
    
    print(f"  Null actions: {null_count:,} ({null_count/total_frames*100:.1f}%)")
    print(f"  Non-null actions: {total_frames - null_count:,} ({(total_frames-null_count)/total_frames*100:.1f}%)")
    
    # Most common button combo
    forward_1 = button_counts["forward"][1] / total_frames * 100
    attack_1 = button_counts["attack"][1] / total_frames * 100
    print(f"  forward=1: {forward_1:.1f}%")
    print(f"  attack=1: {attack_1:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Analyze Diamond action distributions")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--use-decoded", action="store_true")
    parser.add_argument("--n-camera-bins", type=int, default=11)
    parser.add_argument("--raw-camera-only", action="store_true", help="Only analyze raw camera angles")
    args = parser.parse_args()
    
    if args.raw_camera_only:
        analyze_raw_camera(args.data_root)
    else:
        analyze(args.data_root, args.use_decoded, args.n_camera_bins)
        if not args.use_decoded:
            analyze_raw_camera(args.data_root)


if __name__ == "__main__":
    main()

