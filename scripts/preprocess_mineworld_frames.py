"""Decode MineWorld rollouts into preprocessed tensors for fast BC training.

This preprocessing stage converts MineWorld videos + action logs into sharded
PyTorch tensors so the BC trainer can stream batches without hitting OpenCV.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from torchvision import transforms

from datasets.mineworld_data.mcdataset import MCDataset


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT_DIR / "configurations"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _load_hydra_configs(
    dataset_name: str,
    algorithm_name: str,
    experiment_name: str,
) -> Tuple[Dict, Dict, Dict]:
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


class MineWorldFrameSummary:
    """Utility class to collect MineWorld metadata and length estimates."""

    def __init__(
        self,
        data_root: Path,
        context_frames: int,
        resize: int,
        recursive: bool,
    ) -> None:
        self.data_root = Path(data_root)
        self.context_frames = context_frames
        self.resize = resize
        self.recursive = recursive

        dataset = MCDataset()
        dataset.make_action_vocab(action_vocab_offset=0)
        self.action_length = dataset.action_length
        self.action_vocab_size = len(dataset.action_vocab)

        self._video_paths = self._discover_videos()

    def _discover_videos(self) -> List[Path]:
        pattern = "**/*.mp4" if self.recursive else "*.mp4"
        video_paths = sorted(self.data_root.glob(pattern))
        if not video_paths:
            raise FileNotFoundError(f"No .mp4 files found under {self.data_root}")
        return video_paths

    def __len__(self) -> int:
        total = 0
        for video_path in self._video_paths:
            action_path = video_path.with_suffix(".jsonl")
            if not action_path.exists():
                continue
            frame_total = self._count_video_frames(video_path)
            action_total = self._count_actions(action_path)
            usable = min(frame_total, action_total)
            if usable >= self.context_frames:
                total += usable - (self.context_frames - 1)
        return total

    @staticmethod
    def _count_video_frames(video_path: Path) -> int:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            return 0
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return count

    @staticmethod
    def _count_actions(action_path: Path) -> int:
        with action_path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())

    @property
    def video_paths(self) -> List[Path]:
        return list(self._video_paths)


# Worker globals
CHUNK_COUNTER = None
CONTEXT_FRAMES = None
RESIZE = None
CHUNK_SIZE = None
OUTPUT_ROOT = None
FRAME_DTYPE = None
NORMALIZE = None
WORKER_DATASET = None


def _worker_init(
    counter,
    context_frames: int,
    resize: int,
    chunk_size: int,
    output_root: str,
    frame_dtype: str,
) -> None:
    global CHUNK_COUNTER, CONTEXT_FRAMES, RESIZE, CHUNK_SIZE, OUTPUT_ROOT, FRAME_DTYPE
    global NORMALIZE, WORKER_DATASET
    CHUNK_COUNTER = counter
    CONTEXT_FRAMES = context_frames
    RESIZE = resize
    CHUNK_SIZE = chunk_size
    OUTPUT_ROOT = Path(output_root)
    FRAME_DTYPE = frame_dtype
    NORMALIZE = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    cv2.setNumThreads(1)
    WORKER_DATASET = MCDataset()
    WORKER_DATASET.make_action_vocab(action_vocab_offset=0)


def _allocate_chunk_index() -> int:
    with CHUNK_COUNTER.get_lock():  # type: ignore[attr-defined]
        idx = CHUNK_COUNTER.value  # type: ignore[attr-defined]
        CHUNK_COUNTER.value += 1  # type: ignore[attr-defined]
    return idx


def _frame_to_tensor(frame: np.ndarray) -> torch.Tensor:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (RESIZE, RESIZE), interpolation=cv2.INTER_AREA)  # type: ignore[arg-type]
    tensor = torch.from_numpy(frame).permute(2, 0, 1).float().div_(255.0)
    tensor = NORMALIZE(tensor)  # type: ignore[arg-type]
    return tensor


def _process_video(video_path: Path) -> Dict:
    action_path = video_path.with_suffix(".jsonl")
    if not action_path.exists():
        return {"num_samples": 0, "chunks": []}

    try:
        with action_path.open("r", encoding="utf-8") as handle:
            actions = [json.loads(line) for line in handle if line.strip()]
    except Exception:
        return {"num_samples": 0, "chunks": []}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return {"num_samples": 0, "chunks": []}

    frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    usable = min(frame_total, len(actions))
    if usable < CONTEXT_FRAMES:  # type: ignore[operator]
        cap.release()
        return {"num_samples": 0, "chunks": []}

    chunk_frames: List[torch.Tensor] = []
    chunk_labels: List[torch.Tensor] = []
    chunk_entries: List[Dict] = []
    total_samples = 0
    frame_idx = 0

    while frame_idx < usable:
        success, frame = cap.read()
        if not success:
            break
        if frame_idx >= CONTEXT_FRAMES - 1:  # type: ignore[operator]
            tensor = _frame_to_tensor(frame)
            json_action = actions[frame_idx]
            env_action, _ = WORKER_DATASET.json_action_to_env_action(json_action)  # type: ignore[attr-defined]
            indices = WORKER_DATASET.get_action_index_from_actiondict(  # type: ignore[attr-defined]
                env_action, action_vocab_offset=0
            )
            label = torch.tensor(indices, dtype=torch.long)
            chunk_frames.append(tensor)
            chunk_labels.append(label)
            total_samples += 1
            if len(chunk_frames) >= CHUNK_SIZE:  # type: ignore[operator]
                chunk_entries.append(_flush_chunk(chunk_frames, chunk_labels))
        frame_idx += 1

    cap.release()

    if chunk_frames:
        chunk_entries.append(_flush_chunk(chunk_frames, chunk_labels))

    return {"num_samples": total_samples, "chunks": chunk_entries}


def _flush_chunk(
    frames: List[torch.Tensor],
    labels: List[torch.Tensor],
) -> Dict:
    chunk_idx = _allocate_chunk_index()
    chunk_path = OUTPUT_ROOT / f"chunk_{chunk_idx:06d}.pt"  # type: ignore[arg-type]
    frames_tensor = torch.stack(frames, dim=0).to(
        torch.float16 if FRAME_DTYPE == "float16" else torch.float32
    )
    labels_tensor = torch.stack(labels, dim=0)
    torch.save({"frames": frames_tensor, "labels": labels_tensor}, chunk_path)
    frames.clear()
    labels.clear()
    return {"file": chunk_path.name, "num_samples": frames_tensor.size(0)}


def _ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"{path} already exists. Pass --overwrite to replace existing artifacts."
            )
        for existing in path.glob("chunk_*.pt"):
            existing.unlink()
        meta_path = path / "metadata.json"
        if meta_path.exists():
            meta_path.unlink()
    path.mkdir(parents=True, exist_ok=True)


def preprocess_dataset(args: argparse.Namespace) -> None:
    dataset_cfg, _, exp_cfg = _load_hydra_configs(
        dataset_name=args.dataset,
        algorithm_name=args.algorithm,
        experiment_name=args.experiment,
    )

    context_frames = args.context_frames or int(dataset_cfg.get("context_frames", 8))
    resize = args.resize or int(dataset_cfg.get("resize", dataset_cfg.get("resolution", 256)))
    recursive = not args.no_recursive

    data_root_value = args.data_root or dataset_cfg.get("data_root")
    if data_root_value is None:
        raise ValueError("dataset.data_root must be set or provided via --data-root.")
    data_root = (
        Path(data_root_value)
        if Path(data_root_value).is_absolute()
        else (ROOT_DIR / data_root_value).resolve()
    )

    output_root_value = args.output_root or dataset_cfg.get("processed_root")
    if output_root_value is None:
        output_root = ROOT_DIR / "data" / "processed_mineworld_frames"
    else:
        output_root = (
            Path(output_root_value)
            if Path(output_root_value).is_absolute()
            else (ROOT_DIR / output_root_value).resolve()
        )

    _ensure_output_dir(output_root, overwrite=args.overwrite)

    summary = MineWorldFrameSummary(
        data_root=data_root,
        context_frames=context_frames,
        resize=resize,
        recursive=recursive,
    )

    video_paths = summary.video_paths
    if not video_paths:
        raise RuntimeError(f"No videos found under {data_root}")

    chunk_size = int(args.chunk_size)
    if chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")

    frame_dtype = args.frame_dtype
    total_estimate = None if args.no_length else len(summary)

    ctx = mp.get_context("spawn")
    chunk_counter = ctx.Value("i", 0)
    num_workers = args.num_workers or ctx.cpu_count()

    sample_progress = tqdm(
        total=total_estimate,
        desc="processing",
        ncols=80,
        unit="sample",
    )

    chunk_entries: List[Dict] = []
    total_samples = 0

    with ctx.Pool(
        processes=num_workers,
        initializer=_worker_init,
        initargs=(
            chunk_counter,
            context_frames,
            resize,
            chunk_size,
            str(output_root),
            frame_dtype,
        ),
    ) as pool:
        for stats in pool.imap_unordered(_process_video, video_paths):
            total_samples += stats["num_samples"]
            chunk_entries.extend(stats["chunks"])
            sample_progress.update(stats["num_samples"])

    sample_progress.close()

    chunk_entries.sort(key=lambda item: item["file"])

    metadata = {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "source_data_root": str(data_root),
        "context_frames": context_frames,
        "resize": resize,
        "recursive": recursive,
        "chunk_size": chunk_size,
        "frame_dtype": frame_dtype,
        "num_samples": total_samples,
        "action_length": summary.action_length,
        "action_vocab_size": summary.action_vocab_size,
        "training_config": exp_cfg.get("training", {}),
        "chunks": chunk_entries,
        "num_videos": len(video_paths),
        "num_workers": num_workers,
    }

    with (output_root / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(
        f"Processed {total_samples} samples into {len(chunk_entries)} chunks under {output_root}."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=str,
        default="mineworld_frames",
        help="Hydra dataset config name.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="mineworld_bc",
        help="Hydra algorithm config name.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="mineworld_bc_train",
        help="Hydra experiment config name.",
    )
    parser.add_argument("--data-root", type=str, default=None, help="Raw MineWorld data root.")
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Directory to store processed chunks. Defaults to dataset.processed_root.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Number of samples written per shard.",
    )
    parser.add_argument(
        "--frame-dtype",
        type=str,
        choices=["float16", "float32"],
        default="float16",
        help="Floating point precision to store frames on disk.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of parallel decoding workers (default: all cores).",
    )
    parser.add_argument(
        "--context-frames",
        type=int,
        default=None,
        help="Override the context window length.",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=None,
        help="Override the resize dimension.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Disable recursive video discovery under the data root.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing processed chunks before writing new ones.",
    )
    parser.add_argument(
        "--no-length",
        action="store_true",
        help="Skip sample-count estimation for tqdm.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    preprocess_dataset(parse_args())
