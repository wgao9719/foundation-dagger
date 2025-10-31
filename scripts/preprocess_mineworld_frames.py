"""Decode MineWorld rollouts into preprocessed tensors for fast BC training.

This script walks the raw MineWorld dataset, converts each usable frame to a
normalised tensor, computes the corresponding action token targets, and writes
sharded `.pt` chunks alongside metadata. The resulting directory can be fed
directly into `train_initial_bc.py` to avoid per-epoch video decoding.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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


class MineWorldFrameExtractor:
    """Sequential iterator over MineWorld trajectories with preprocessing."""

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

        self.mc_dataset = MCDataset()
        self.mc_dataset.make_action_vocab(action_vocab_offset=0)
        self.action_length = self.mc_dataset.action_length
        self.action_vocab_size = len(self.mc_dataset.action_vocab)

        self.normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        self.to_tensor = transforms.ToTensor()
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
            actions = self._load_actions(action_path)
            frame_total = self._count_video_frames(video_path)
            usable = min(frame_total, len(actions))
            if usable >= self.context_frames:
                total += usable - (self.context_frames - 1)
        return total

    @staticmethod
    def _load_actions(action_path: Path) -> List[Dict]:
        actions: List[Dict] = []
        with action_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    actions.append(json.loads(line))
        return actions

    @staticmethod
    def _count_video_frames(video_path: Path) -> int:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return 0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return frame_count

    def _frame_to_tensor(self, frame: np.ndarray) -> torch.Tensor:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (self.resize, self.resize), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(frame).permute(2, 0, 1).float().div_(255.0)
        tensor = self.normalize(tensor)
        return tensor

    def iter_samples(self) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
        for video_path in self._video_paths:
            action_path = video_path.with_suffix(".jsonl")
            if not action_path.exists():
                continue
            actions = self._load_actions(action_path)
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                cap.release()
                continue

            frame_idx = 0
            usable = min(self._count_video_frames(video_path), len(actions))
            if usable < self.context_frames:
                cap.release()
                continue

            while frame_idx < usable:
                success, frame = cap.read()
                if not success:
                    break
                if frame_idx >= self.context_frames - 1:
                    tensor = self._frame_to_tensor(frame)
                    json_action = actions[frame_idx]
                    env_action, _ = self.mc_dataset.json_action_to_env_action(json_action)
                    indices = self.mc_dataset.get_action_index_from_actiondict(
                        env_action, action_vocab_offset=0
                    )
                    label = torch.tensor(indices, dtype=torch.long)
                    yield tensor, label
                frame_idx += 1
            cap.release()


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


def _save_chunk(
    output_root: Path,
    chunk_index: int,
    frames: List[torch.Tensor],
    labels: List[torch.Tensor],
) -> Dict:
    chunk_path = output_root / f"chunk_{chunk_index:05d}.pt"
    frames_tensor = torch.stack(frames, dim=0)
    labels_tensor = torch.stack(labels, dim=0)
    torch.save({"frames": frames_tensor, "labels": labels_tensor}, chunk_path)
    return {"file": chunk_path.name, "num_samples": len(frames)}


def preprocess_dataset(args: argparse.Namespace) -> None:
    dataset_cfg, policy_cfg_dict, exp_cfg = _load_hydra_configs(
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

    extractor = MineWorldFrameExtractor(
        data_root=data_root,
        context_frames=context_frames,
        resize=resize,
        recursive=recursive,
    )

    chunk_size = int(args.chunk_size)
    if chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")

    chunk_frames: List[torch.Tensor] = []
    chunk_labels: List[torch.Tensor] = []
    chunk_entries: List[Dict] = []
    chunk_index = 0
    total_samples = 0

    progress = tqdm(
        extractor.iter_samples(),
        desc="processing",
        total=None if args.no_length else len(extractor),
        ncols=80,
    )

    for frame_tensor, label_tensor in progress:
        chunk_frames.append(frame_tensor)
        chunk_labels.append(label_tensor)

        if len(chunk_frames) >= chunk_size:
            entry = _save_chunk(output_root, chunk_index, chunk_frames, chunk_labels)
            chunk_entries.append(entry)
            total_samples += entry["num_samples"]
            chunk_frames.clear()
            chunk_labels.clear()
            chunk_index += 1
            progress.set_postfix(samples=total_samples)

    if chunk_frames:
        entry = _save_chunk(output_root, chunk_index, chunk_frames, chunk_labels)
        chunk_entries.append(entry)
        total_samples += entry["num_samples"]

    metadata = {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "source_data_root": str(data_root),
        "context_frames": context_frames,
        "resize": resize,
        "recursive": recursive,
        "chunk_size": chunk_size,
        "num_samples": total_samples,
        "action_length": extractor.action_length,
        "action_vocab_size": extractor.action_vocab_size,
        "policy_config": policy_cfg_dict,
        "training_config": exp_cfg.get("training", {}),
        "chunks": chunk_entries,
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
        default=512,
        help="Number of samples per shard written to disk.",
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
        help="Disable length estimation for tqdm (useful when frame counts are unreliable).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    preprocess_dataset(parse_args())
