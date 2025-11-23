"""Decode MineWorld trajectories into reusable tensor bundles.

This utility loads MineWorld videos and action logs using the existing
MineWorldFrameDataset implementation, but instead of decoding frames on the fly
it writes the resized RGB frames plus the aligned action labels to disk. The
resulting dataset can be consumed by MineWorldDecodedFrameDataset, enabling BC
training to avoid repeatedly decoding the same MP4 files.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from datasets.mineworld_data.mineworld_frame_dataset import (
    MineWorldFrameDataset,
    MineWorldDecodedFrameDataset,
)
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
    working_frame = frame.copy() if is_gui_open else frame
    if is_gui_open:
        camera_scaling_factor = working_frame.shape[0] / MINEREC_ORIGINAL_HEIGHT_PX
        x_pos = int(cursor_x * camera_scaling_factor)
        y_pos = int(cursor_y * camera_scaling_factor)
        composite_images_with_alpha(
            working_frame,
            _WORKER_CURSOR_RGB,
            _WORKER_CURSOR_ALPHA,
            x_pos,
            y_pos,
        )
    working_frame = cv2.cvtColor(working_frame, cv2.COLOR_BGR2RGB)
    working_frame = np.asarray(np.clip(working_frame, 0, 255), dtype=np.uint8)
    working_frame = resize_image(working_frame, AGENT_RESOLUTION)
    return torch.from_numpy(working_frame)


def _decode_video_job(job: Dict[str, object]) -> Dict[str, object]:
    video_path = Path(job["video_path"])
    output_file = Path(job["output_file"])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    step_infos: List[Dict[str, float]] = job["step_infos"]  # type: ignore[assignment]
    if not step_infos:
        return {
            "video_id": job["video_id"],
            "decoded_file": job["relative_decoded"],
            "original_video": job["original_video"],
            "num_steps": 0,
        }
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {video_path}")

    frames: List[torch.Tensor] = []
    frame_indices = torch.empty(len(step_infos), dtype=torch.long)
    target_indices = [int(step["frame_idx"]) for step in step_infos]
    for earlier, later in zip(target_indices, target_indices[1:]):
        if later < earlier:
            raise ValueError(
                f"Frame indices for {video_path} must be non-decreasing; "
                f"got {earlier} followed by {later}."
            )

    current_frame_idx = -1
    target_pos = 0
    total_targets = len(target_indices)
    next_target = target_indices[0]

    while target_pos < total_targets:
        success, frame = cap.read()
        if not success:
            raise RuntimeError(
                f"Could not read frame {next_target} from {video_path} "
                f"(stopped at {current_frame_idx})."
            )
        current_frame_idx += 1
        if current_frame_idx < next_target:
            continue
        if current_frame_idx > next_target:
            continue

        while target_pos < total_targets and target_indices[target_pos] == current_frame_idx:
            step = step_infos[target_pos]
            tensor = _frame_to_tensor_worker(
                frame,
                is_gui_open=bool(step.get("is_gui_open", False)),
                cursor_x=float(step.get("cursor_x", 0.0)),
                cursor_y=float(step.get("cursor_y", 0.0)),
            )
            frames.append(tensor)
            frame_indices[target_pos] = current_frame_idx
            target_pos += 1
            if target_pos < total_targets:
                next_target = target_indices[target_pos]
    cap.release()

    frames_tensor = torch.stack(frames, dim=0).contiguous()
    payload = {
        "frames": frames_tensor.to(torch.uint8),
        "frame_indices": frame_indices,
        "buttons": torch.tensor(job["buttons"], dtype=torch.long),  # type: ignore[arg-type]
        "camera": torch.tensor(job["camera"], dtype=torch.long),  # type: ignore[arg-type]
        "esc": torch.tensor(job["esc"], dtype=torch.long),  # type: ignore[arg-type]
    }
    torch.save(payload, output_file)
    return {
        "video_id": job["video_id"],
        "decoded_file": job["relative_decoded"],
        "original_video": job["original_video"],
        "num_steps": len(step_infos),
    }


def _prepare_jobs(
    dataset: MineWorldFrameDataset,
    data_root: Path,
    output_root: Path,
) -> List[Dict[str, object]]:
    jobs: List[Dict[str, object]] = []
    video_items = sorted(dataset._video_ids.items(), key=lambda item: item[1])
    for video_path, video_id in video_items:
        step_infos = dataset.video_step_infos.get(int(video_id), [])
        if not step_infos:
            continue
        simplified_steps: List[Dict[str, float]] = []
        buttons: List[int] = []
        camera: List[int] = []
        esc_flags: List[int] = []
        for step in step_infos:
            frame_idx = int(step["frame_idx"])
            simplified_steps.append(
                dict(
                    frame_idx=frame_idx,
                    is_gui_open=bool(step.get("is_gui_open", False)),
                    cursor_x=float(step.get("cursor_x", 0.0)),
                    cursor_y=float(step.get("cursor_y", 0.0)),
                )
            )
            agent_action = dataset.agent_actions.get((int(video_id), frame_idx))
            if agent_action is None:
                raise KeyError(f"Missing agent action for video_id={video_id}, frame_idx={frame_idx}")
            buttons.append(int(torch.as_tensor(agent_action["buttons"]).view(-1)[0].item()))
            camera.append(int(torch.as_tensor(agent_action["camera"]).view(-1)[0].item()))
            esc_flags.append(int(dataset.get_esc_flag(int(video_id), frame_idx)))
        relative_video = Path(video_path).relative_to(data_root)
        output_file = output_root / relative_video.with_suffix(".pt")
        jobs.append(
            {
                "video_id": int(video_id),
                "video_path": str(video_path),
                "output_file": str(output_file),
                "relative_decoded": output_file.relative_to(output_root).as_posix(),
                "original_video": relative_video.as_posix(),
                "step_infos": simplified_steps,
                "buttons": buttons,
                "camera": camera,
                "esc": esc_flags,
            }
        )
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
        help="Maximum number of concurrent cv2.VideoCapture handles.",
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

    dataset = MineWorldFrameDataset(
        data_root=data_root,
        context_frames=int(args.context_frames),
        recursive=recursive,
        max_open_captures=int(args.max_open_captures),
    )

    jobs = _prepare_jobs(dataset, data_root, output_root)
    manifest_entries: List[Dict[str, object]] = []
    num_workers = max(1, int(args.num_workers))
    if not jobs:
        manifest_entries = []
    elif num_workers == 1:
        iterator = tqdm(jobs, desc="Decoding videos")
        for job in iterator:
            entry = _decode_video_job(job)
            manifest_entries.append(entry)
            iterator.set_postfix({"frames": entry["num_steps"]})
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_decode_video_job, job): job for job in jobs}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Decoding videos"):
                entry = future.result()
                manifest_entries.append(entry)

    if not manifest_entries:
        raise RuntimeError("No videos were decoded; ensure the MineWorld data root is correct.")

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
