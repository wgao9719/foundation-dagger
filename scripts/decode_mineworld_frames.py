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
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

try:
    import av  # type: ignore
except ImportError:  # pragma: no cover
    av = None

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from datasets.mineworld_data.mineworld_decode_index import MineWorldDecodeIndex
from datasets.mineworld_data.mineworld_frame_dataset import MineWorldDecodedFrameDataset
from evaluation.agent import AGENT_RESOLUTION, resize_image
from evaluation.data_loader import (
    CURSOR_FILE,
    MINEREC_ORIGINAL_HEIGHT_PX,
    composite_images_with_alpha,
)

_WORKER_CURSOR_RGB = None
_WORKER_CURSOR_ALPHA = None
_WORKER_CURSOR_TENSORS: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
_WORKER_CV2_THREADS_DISABLED = False


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


def _ensure_single_threaded_cv2() -> None:
    global _WORKER_CV2_THREADS_DISABLED
    if _WORKER_CV2_THREADS_DISABLED:
        return
    try:
        cv2.setNumThreads(0)
    except AttributeError:
        pass
    _WORKER_CV2_THREADS_DISABLED = True


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


def _decode_video_job_opencv(job: Dict[str, object]) -> Dict[str, object]:
    _ensure_single_threaded_cv2()
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

    frames_tensor: torch.Tensor | None = None
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
        next_target = target_indices[target_pos]
        if current_frame_idx + 1 < next_target:
            cap.set(cv2.CAP_PROP_POS_FRAMES, next_target)
            current_frame_idx = next_target - 1
        success, frame = cap.read()
        if not success:
            raise RuntimeError(
                f"Could not read frame {next_target} from {video_path} "
                f"(stopped at {current_frame_idx})."
            )
        current_frame_idx += 1
        if current_frame_idx < next_target:
            continue

        while target_pos < total_targets and target_indices[target_pos] == current_frame_idx:
            step = step_infos[target_pos]
            tensor = _frame_to_tensor_worker(
                frame,
                is_gui_open=bool(step.get("is_gui_open", False)),
                cursor_x=float(step.get("cursor_x", 0.0)),
                cursor_y=float(step.get("cursor_y", 0.0)),
            )
            if frames_tensor is None:
                frames_tensor = torch.empty(
                    (total_targets, *tensor.shape),
                    dtype=torch.uint8,
                    device=tensor.device,
                )
            frames_tensor[target_pos] = tensor
            frame_indices[target_pos] = current_frame_idx
            target_pos += 1
            if target_pos < total_targets:
                next_target = target_indices[target_pos]
    cap.release()

    if frames_tensor is None:
        raise RuntimeError(f"No frames decoded for {video_path}")
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


def _decode_video_job_pyav(job: Dict[str, object]) -> Dict[str, object]:
    if av is None:
        raise RuntimeError("PyAV backend requested but the 'av' package is not installed.")
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

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    thread_count = job.get("pyav_thread_count")
    if isinstance(thread_count, int) and thread_count > 0:
        stream.thread_type = "AUTO"
        stream.thread_count = thread_count

    frames_tensor: torch.Tensor | None = None
    frame_indices = torch.empty(len(step_infos), dtype=torch.long)
    target_indices = [int(step["frame_idx"]) for step in step_infos]
    for earlier, later in zip(target_indices, target_indices[1:]):
        if later < earlier:
            raise ValueError(
                f"Frame indices for {video_path} must be non-decreasing; got {earlier} followed by {later}."
            )

    current_frame_idx = -1
    target_pos = 0
    total_targets = len(target_indices)
    next_target = target_indices[0]

    try:
        for frame in container.decode(stream):
            current_frame_idx += 1
            if current_frame_idx < next_target:
                continue
            if current_frame_idx > next_target:
                continue

            numpy_frame = frame.to_ndarray(format="bgr24")
            while target_pos < total_targets and target_indices[target_pos] == current_frame_idx:
                step = step_infos[target_pos]
                tensor = _frame_to_tensor_worker(
                    numpy_frame,
                    is_gui_open=bool(step.get("is_gui_open", False)),
                    cursor_x=float(step.get("cursor_x", 0.0)),
                    cursor_y=float(step.get("cursor_y", 0.0)),
                )
                if frames_tensor is None:
                    frames_tensor = torch.empty(
                        (total_targets, *tensor.shape),
                        dtype=torch.uint8,
                        device=tensor.device,
                    )
                frames_tensor[target_pos] = tensor
                frame_indices[target_pos] = current_frame_idx
                target_pos += 1
                if target_pos < total_targets:
                    next_target = target_indices[target_pos]
            if target_pos >= total_targets:
                break
    finally:
        container.close()

    if target_pos < total_targets or frames_tensor is None:
        raise RuntimeError(f"Could not decode all target frames from {video_path} using PyAV.")

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


def _decode_video_job(job: Dict[str, object]) -> Dict[str, object]:
    backend = job.get("video_backend", "opencv")
    if backend == "pyav":
        return _decode_video_job_pyav(job)
    return _decode_video_job_opencv(job)


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
    parser.add_argument(
        "--video-backend",
        type=str,
        choices=["opencv", "pyav"],
        default="opencv",
        help="Video decoding backend to use. PyAV allows multi-threaded decoding when available.",
    )
    parser.add_argument(
        "--pyav-thread-count",
        type=int,
        default=None,
        help="Optional per-stream thread count for the PyAV backend.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Resume decoding by skipping videos whose decoded tensors already exist.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of available videos to decode (0 < fraction <= 1).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.skip_existing and args.overwrite:
        raise ValueError("--skip-existing cannot be combined with --overwrite.")
    data_root = Path(args.data_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"MineWorld data root {data_root} does not exist.")
    if output_root.exists() and any(output_root.iterdir()) and not (args.overwrite or args.skip_existing):
        raise RuntimeError(
            f"Output directory {output_root} already contains files. "
            "Pass --overwrite to rebuild the decoded dataset."
        )
    output_root.mkdir(parents=True, exist_ok=True)
    recursive = not args.no_recursive

    fraction = float(args.fraction)
    if fraction <= 0.0 or fraction > 1.0:
        raise ValueError("--fraction must be in the range (0, 1].")
    print("Building decode index...")
    decode_index = MineWorldDecodeIndex(
        data_root=data_root,
        recursive=recursive,
        max_fraction=fraction,
    )
    print("Decode index built")

    jobs: List[Dict[str, object]] = []
    backend = args.video_backend.lower()
    if backend not in {"opencv", "pyav"}:
        raise ValueError(f"Unsupported video backend '{args.video_backend}'.")
    if backend == "pyav" and av is None:
        raise RuntimeError(
            "PyAV backend requested via --video-backend=pyav, but the 'av' package is not installed."
        )
    records = decode_index.iter_records()
    if fraction < 1.0:
        total = decode_index.total_candidates or len(records)
        frac_pct = (len(records) / total) if total else 0.0
        print(f"Decoding {len(records)} of {total} videos (~{frac_pct:.2%}).")
    for record in records:
        relative_video = Path(record["relative_video"])
        output_file = output_root / relative_video.with_suffix(".pt")
        jobs.append(
            {
                "video_id": record["video_id"],
                "video_path": str(record["video_path"]),
                "output_file": str(output_file),
                "relative_decoded": output_file.relative_to(output_root).as_posix(),
                "original_video": relative_video.as_posix(),
                "step_infos": record["step_infos"],
                "buttons": record["buttons"],
                "camera": record["camera"],
                "esc": record["esc"],
                "video_backend": backend,
                "pyav_thread_count": args.pyav_thread_count,
            }
        )
    manifest_entries: List[Dict[str, object]] = []
    if args.skip_existing:
        remaining_jobs: List[Dict[str, object]] = []
        skipped = 0
        for job in jobs:
            if Path(job["output_file"]).exists():
                skipped += 1
                manifest_entries.append(
                    {
                        "video_id": job["video_id"],
                        "decoded_file": job["relative_decoded"],
                        "original_video": job["original_video"],
                        "num_steps": len(job["step_infos"]),
                    }
                )
                continue
            remaining_jobs.append(job)
        jobs = remaining_jobs
        if skipped:
            print(f"Skipping {skipped} existing decoded videos.")
    num_workers = max(1, int(args.num_workers))
    if jobs:
        if num_workers == 1:
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
