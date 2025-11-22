"""Decode MineWorld trajectories into reusable tensor bundles.

This utility loads MineWorld videos and action logs, but unlike the original implementation,
it parallelizes the parsing of action logs and video alignment to avoid the high startup cost
and IPC overhead of passing large data structures to workers.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from datasets.mineworld_data.mineworld_frame_dataset import (
    MineWorldDecodedFrameDataset,
)
from datasets.mineworld_data.parsing import process_video_actions
from evaluation.agent import AGENT_RESOLUTION, resize_image
from evaluation.data_loader import (
    CURSOR_FILE,
    MINEREC_ORIGINAL_HEIGHT_PX,
    composite_images_with_alpha,
)


_WORKER_CURSOR_RGB = None
_WORKER_CURSOR_ALPHA = None


def _init_worker_cursor() -> None:
    global _WORKER_CURSOR_RGB, _WORKER_CURSOR_ALPHA
    if _WORKER_CURSOR_RGB is not None:
        return
    cursor_image = cv2.imread(CURSOR_FILE, cv2.IMREAD_UNCHANGED)
    if cursor_image is None:
        raise FileNotFoundError(f"Cursor image missing at {CURSOR_FILE}")
    cursor_image = cursor_image[:16, :16, :]
    _WORKER_CURSOR_ALPHA = cursor_image[:, :, 3:] / 255.0
    _WORKER_CURSOR_RGB = cursor_image[:, :, :3]


def _frame_to_tensor_worker(
    frame: np.ndarray,
    is_gui_open: bool,
    cursor_x: float,
    cursor_y: float,
) -> torch.Tensor:
    _init_worker_cursor()
    if is_gui_open:
        camera_scaling_factor = frame.shape[0] / MINEREC_ORIGINAL_HEIGHT_PX
        x_pos = int(cursor_x * camera_scaling_factor)
        y_pos = int(cursor_y * camera_scaling_factor)
        composite_images_with_alpha(frame, _WORKER_CURSOR_RGB, _WORKER_CURSOR_ALPHA, x_pos, y_pos)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.asarray(np.clip(frame, 0, 255), dtype=np.uint8)
    frame = resize_image(frame, AGENT_RESOLUTION)
    return torch.from_numpy(frame)


def _read_frame_at(cap: cv2.VideoCapture, frame_idx: int, last_idx: int) -> Tuple[np.ndarray, int]:
    if last_idx + 1 != frame_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    success, frame = cap.read()
    if not success:
        raise RuntimeError(f"Could not read frame {frame_idx}")
    return frame, frame_idx


def _decode_video_job(job: Dict[str, object]) -> Dict[str, object]:
    video_path = Path(job["video_path"])
    action_path = Path(job["action_path"])
    output_file = Path(job["output_file"])
    output_file.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {video_path}")
    
    # Get frame count to limit parsing
    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Parse actions locally in the worker
    # We pass max_frames to avoid re-opening the video inside parsing
    step_infos, agent_actions, esc_flags, _ = process_video_actions(
        video_path, 
        action_path, 
        context_frames=1,  # Not used for filtering here
        check_video_frame_count=False,
        max_frames=video_frame_count
    )

    if not step_infos:
        cap.release()
        return {
            "video_id": job["video_id"],
            "decoded_file": None,
            "original_video": job["original_video"],
            "num_steps": 0,
            "skipped": True,
        }

    last_idx = -1
    frames: List[torch.Tensor] = []
    frame_indices = torch.empty(len(step_infos), dtype=torch.long)
    
    buttons_list: List[int] = []
    camera_list: List[int] = []

    for idx, step in enumerate(step_infos):
        frame_idx = int(step["frame_idx"])
        raw_frame, last_idx = _read_frame_at(cap, frame_idx, last_idx)
        tensor = _frame_to_tensor_worker(
            raw_frame,
            is_gui_open=bool(step.get("is_gui_open", False)),
            cursor_x=float(step.get("cursor_x", 0.0)),
            cursor_y=float(step.get("cursor_y", 0.0)),
        )
        frames.append(tensor)
        frame_indices[idx] = frame_idx
        
        # Extract action values
        agent_action = agent_actions[idx]
        buttons_list.append(int(torch.as_tensor(agent_action["buttons"]).view(-1)[0].item()))
        camera_list.append(int(torch.as_tensor(agent_action["camera"]).view(-1)[0].item()))

    cap.release()

    frames_tensor = torch.stack(frames, dim=0).contiguous()
    payload = {
        "frames": frames_tensor.to(torch.uint8),
        "frame_indices": frame_indices,
        "buttons": torch.tensor(buttons_list, dtype=torch.long),
        "camera": torch.tensor(camera_list, dtype=torch.long),
        "esc": torch.tensor(esc_flags, dtype=torch.long),
    }
    torch.save(payload, output_file)
    return {
        "video_id": job["video_id"],
        "decoded_file": job["relative_decoded"],
        "original_video": job["original_video"],
        "num_steps": len(step_infos),
    }


def _prepare_jobs_lightweight(
    data_root: Path,
    output_root: Path,
    recursive: bool
) -> List[Dict[str, object]]:
    jobs: List[Dict[str, object]] = []
    pattern = "**/*.mp4" if recursive else "*.mp4"
    video_paths = sorted(data_root.glob(pattern))
    
    for i, video_path in enumerate(video_paths):
        action_path = video_path.with_suffix(".jsonl")
        if not action_path.exists():
            continue
            
        relative_video = video_path.relative_to(data_root)
        output_file = output_root / relative_video.with_suffix(".pt")
        
        jobs.append({
            "video_id": i,
            "video_path": str(video_path),
            "action_path": str(action_path),
            "output_file": str(output_file),
            "relative_decoded": output_file.relative_to(output_root).as_posix(),
            "original_video": relative_video.as_posix(),
        })
    return jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode MineWorld rollouts into tensor files.")
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Directory containing MineWorld videos (.mp4) and action logs (.jsonl).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Destination directory for decoded tensors and manifest.",
    )
    parser.add_argument(
        "--context-frames",
        type=int,
        default=8,
        help="Context window size; used only for dataset construction consistency.",
    )
    parser.add_argument(
        "--max-open-captures",
        type=int,
        default=12,
        help="Ignored in this optimized version, but kept for compatibility.",
    )
    parser.add_argument(
        "--manifest-name",
        type=str,
        default="manifest.json",
        help="Name of the manifest file recorded inside the output directory.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Disable recursive search for videos under the MineWorld data root.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow reusing an existing output directory instead of requiring it to be empty.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel workers to use while decoding videos.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"MineWorld data root {data_root} does not exist.")
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise RuntimeError(
            f"Output directory {output_root} already contains files. "
            "Pass --overwrite to rebuild the decoded dataset."
        )
    output_root.mkdir(parents=True, exist_ok=True)
    recursive = not args.no_recursive

    # Use lightweight job preparation
    jobs = _prepare_jobs_lightweight(data_root, output_root, recursive)
    
    manifest_entries: List[Dict[str, object]] = []
    num_workers = max(1, int(args.num_workers))
    
    if not jobs:
        manifest_entries = []
    elif num_workers == 1:
        iterator = tqdm(jobs, desc="Decoding videos")
        for job in iterator:
            entry = _decode_video_job(job)
            if not entry.get("skipped"):
                manifest_entries.append(entry)
                iterator.set_postfix({"frames": entry["num_steps"]})
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_decode_video_job, job): job for job in jobs}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Decoding videos"):
                try:
                    entry = future.result()
                    if not entry.get("skipped"):
                        manifest_entries.append(entry)
                except Exception as e:
                    job = futures[future]
                    print(f"Job failed for video {job['video_path']}: {e}")

    if not manifest_entries:
        print("Warning: No videos were successfully decoded.")
    
    manifest = {
        "version": MineWorldDecodedFrameDataset.MANIFEST_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "context_frames": int(args.context_frames),
        "source_data_root": data_root.as_posix(),
        "videos": manifest_entries,
    }
    manifest_path = output_root / args.manifest_name
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Decoded {len(manifest_entries)} videos to {output_root}. Manifest written to {manifest_path}.")


if __name__ == "__main__":
    main()
