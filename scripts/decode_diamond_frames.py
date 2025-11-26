"""Decode MineRL Diamond trajectories into tensor bundles for fast training."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm


def _decode_trajectory(job: Dict) -> Dict:
    """Decode a single trajectory's video into a tensor bundle."""
    try:
        cv2.setNumThreads(0)
    except AttributeError:
        pass
    
    traj_dir = Path(job["traj_dir"])
    output_file = Path(job["output_file"])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    video_path = traj_dir / "recording.mp4"
    npz_path = traj_dir / "rendered.npz"
    
    # Load npz data
    npz_data = dict(np.load(npz_path, allow_pickle=True))
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {video_path}")
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_actions = len(npz_data["action$forward"])
    usable_frames = min(frame_count, num_actions)
    
    # Decode all frames
    frames_list = []
    for _ in range(usable_frames):
        success, frame = cap.read()
        if not success:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_list.append(frame_rgb)
    cap.release()
    
    if not frames_list:
        raise RuntimeError(f"No frames decoded from {video_path}")
    
    frames = np.stack(frames_list, axis=0)  # [T, H, W, 3]
    
    # Save everything
    payload = {
        "frames": torch.from_numpy(frames),
        "npz_keys": list(npz_data.keys()),
    }
    # Add all npz arrays
    for key, value in npz_data.items():
        payload[key] = value[:usable_frames] if len(value) > usable_frames else value
    
    torch.save(payload, output_file)
    
    return {
        "traj_id": job["traj_id"],
        "decoded_file": job["relative_decoded"],
        "num_frames": len(frames_list),
    }


def main():
    parser = argparse.ArgumentParser(description="Decode MineRL Diamond trajectories.")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    
    data_root = Path(args.data_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Find trajectories
    traj_dirs = []
    for root, dirs, files in data_root.rglob("*"):
        pass
    for path in data_root.rglob("rendered.npz"):
        if (path.parent / "recording.mp4").exists():
            traj_dirs.append(path.parent)
    traj_dirs = sorted(set(traj_dirs))
    
    print(f"Found {len(traj_dirs)} trajectories")
    
    # Build jobs
    jobs = []
    manifest_entries = []
    for idx, traj_dir in enumerate(traj_dirs):
        rel_path = traj_dir.relative_to(data_root)
        output_file = output_root / f"{rel_path.as_posix().replace('/', '_')}.pt"
        
        if args.skip_existing and output_file.exists():
            manifest_entries.append({
                "traj_id": idx,
                "decoded_file": output_file.name,
                "traj_dir": rel_path.as_posix(),
            })
            continue
        
        jobs.append({
            "traj_id": idx,
            "traj_dir": str(traj_dir),
            "output_file": str(output_file),
            "relative_decoded": output_file.name,
            "relative_traj": rel_path.as_posix(),
        })
    
    if args.skip_existing:
        print(f"Skipping {len(manifest_entries)} existing, decoding {len(jobs)}")
    
    # Decode
    if jobs:
        if args.num_workers <= 1:
            for job in tqdm(jobs, desc="Decoding"):
                entry = _decode_trajectory(job)
                entry["traj_dir"] = job["relative_traj"]
                manifest_entries.append(entry)
        else:
            with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
                futures = {executor.submit(_decode_trajectory, j): j for j in jobs}
                for future in tqdm(as_completed(futures), total=len(futures), desc="Decoding"):
                    job = futures[future]
                    entry = future.result()
                    entry["traj_dir"] = job["relative_traj"]
                    manifest_entries.append(entry)
    
    # Write manifest
    manifest = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_data_root": str(data_root),
        "trajectories": sorted(manifest_entries, key=lambda x: x["traj_id"]),
    }
    manifest_path = output_root / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Decoded {len(manifest_entries)} trajectories to {output_root}")


if __name__ == "__main__":
    main()

