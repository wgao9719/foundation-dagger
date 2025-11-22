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
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from tqdm.auto import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from datasets.mineworld_data.mineworld_frame_dataset import (
    MineWorldFrameDataset,
    MineWorldDecodedFrameDataset,
)


def _decode_video(
    dataset: MineWorldFrameDataset,
    video_path: Path,
    video_id: int,
) -> Tuple[Dict[str, torch.Tensor], int]:
    step_infos = dataset.video_step_infos.get(int(video_id), [])
    if not step_infos:
        return {}, 0
    frames: List[torch.Tensor] = []
    frame_indices = torch.empty(len(step_infos), dtype=torch.long)
    buttons = torch.empty(len(step_infos), dtype=torch.long)
    camera = torch.empty(len(step_infos), dtype=torch.long)
    esc_flags = torch.empty(len(step_infos), dtype=torch.uint8)

    for idx, step in enumerate(step_infos):
        frame_idx = int(step["frame_idx"])
        raw_frame = dataset._read_frame(video_path, frame_idx)
        tensor = dataset._frame_to_tensor(
            raw_frame,
            is_gui_open=bool(step.get("is_gui_open", False)),
            cursor_x=float(step.get("cursor_x", 0.0)),
            cursor_y=float(step.get("cursor_y", 0.0)),
        )
        frames.append(tensor)
        frame_indices[idx] = frame_idx
        agent_action = dataset.agent_actions.get((int(video_id), frame_idx))
        if agent_action is None:
            raise KeyError(f"Missing agent action for video_id={video_id}, frame_idx={frame_idx}")
        buttons[idx] = int(torch.as_tensor(agent_action["buttons"]).view(-1)[0].item())
        camera[idx] = int(torch.as_tensor(agent_action["camera"]).view(-1)[0].item())
        esc_flags[idx] = dataset.get_esc_flag(video_id, frame_idx)

    frames_tensor = torch.stack(frames, dim=0).contiguous()
    payload = {
        "frames": frames_tensor.to(torch.uint8),
        "frame_indices": frame_indices,
        "buttons": buttons,
        "camera": camera,
        "esc": esc_flags,
    }
    return payload, len(step_infos)


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

    video_items = sorted(dataset._video_ids.items(), key=lambda item: item[1])
    manifest_entries: List[Dict[str, object]] = []
    iterator = tqdm(video_items, desc="Decoding videos")
    for video_path, video_id in iterator:
        video_path = Path(video_path)
        relative_video = video_path.relative_to(data_root)
        output_file = output_root / relative_video.with_suffix(".pt")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        payload, num_steps = _decode_video(dataset, video_path, video_id)
        if not payload:
            continue
        torch.save(payload, output_file)
        manifest_entries.append(
            {
                "video_id": int(video_id),
                "decoded_file": output_file.relative_to(output_root).as_posix(),
                "original_video": relative_video.as_posix(),
                "num_steps": num_steps,
            }
        )
        iterator.set_postfix({"frames": num_steps})

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
