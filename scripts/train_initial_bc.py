"""Train an initial BC policy on MineWorld rollouts.

The script ingests MineWorld-style trajectories (paired ``.mp4`` videos and
``.jsonl`` action logs) and produces a supervised policy checkpoint compatible
with the ``foundation_dagger`` module. Each training sample uses the final
frame of an ``N``-frame context window as the observation and maps the
associated action dictionary to one of four movement classes (forward, back,
left, right). Only a user-specified fraction of the available samples is used
to keep the bootstrap stage quick.

Example
-------
python scripts/train_initial_bc.py \
    --dataset mineworld_frames \
    --data-root /Users/willi1/foundation-dagger/diffusion-forcing-transformer/data/mineworld \
    --fraction 0.1 \
    --epochs 10 \
    --batch-size 32 \
    --output checkpoints/bc_policy.ckpt
"""


from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from bisect import bisect_right

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm
from torchvision import transforms
import wandb

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from datasets.mineworld_data.mcdataset import MCDataset

import cv2 
import numpy as np

CONFIG_DIR = ROOT_DIR / "configurations"

from algorithms.foundation_dagger.policy import FoundationBCPolicy, PolicyConfig

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class ProcessedMineWorldDataset(Dataset):
    """
    Dataset wrapper for preprocessed MineWorld chunks stored under a directory.
    """

    def __init__(self, processed_root: Path, cache_size: int = 2) -> None:
        self.processed_root = Path(processed_root)
        meta_path = self.processed_root / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Processed metadata not found at {meta_path}. "
                "Run preprocess_mineworld_frames.py before consuming the dataset."
            )
        with meta_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        self.metadata = metadata
        self.context_frames = int(metadata["context_frames"])
        self.resize = int(metadata.get("resize", 256))
        self.action_length = int(metadata["action_length"])
        self.action_vocab_size = int(metadata["action_vocab_size"])
        self.total_samples = int(metadata["num_samples"])

        self.chunks = metadata.get("chunks", [])
        if not self.chunks:
            raise ValueError("Processed dataset metadata does not list any chunks.")

        cumulative: list[int] = []
        total = 0
        for entry in self.chunks:
            count = int(entry["num_samples"])
            if count <= 0:
                continue
            total += count
            cumulative.append(total)
        if total != self.total_samples:
            self.total_samples = total
        self.cumulative_sizes = cumulative

        self.cache_size = max(1, cache_size)
        self._chunk_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._chunk_order: list[int] = []

    def __len__(self) -> int:
        return self.total_samples

    def _prune_cache(self) -> None:
        while len(self._chunk_order) > self.cache_size:
            oldest = self._chunk_order.pop(0)
            self._chunk_cache.pop(oldest, None)

    def _get_chunk(self, chunk_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        cached = self._chunk_cache.get(chunk_idx)
        if cached is not None:
            if chunk_idx in self._chunk_order:
                self._chunk_order.remove(chunk_idx)
            self._chunk_order.append(chunk_idx)
            return cached

        entry = self.chunks[chunk_idx]
        chunk_path = self.processed_root / entry["file"]
        if not chunk_path.exists():
            raise FileNotFoundError(f"Missing processed chunk file at {chunk_path}")
        data = torch.load(chunk_path, map_location="cpu")
        frames = data["frames"].float()
        labels = data["labels"].long()
        self._chunk_cache[chunk_idx] = (frames, labels)
        self._chunk_order.append(chunk_idx)
        self._prune_cache()
        return frames, labels

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self)}")
        chunk_idx = bisect_right(self.cumulative_sizes, index)
        chunk_start = 0 if chunk_idx == 0 else self.cumulative_sizes[chunk_idx - 1]
        local_idx = index - chunk_start
        frames, labels = self._get_chunk(chunk_idx)
        return frames[local_idx], labels[local_idx]

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_chunk_cache"] = {}
        state["_chunk_order"] = []
        return state


def _load_hydra_configs(
    dataset_name: str = "mineworld_frames",
    algorithm_name: str = "mineworld_bc",
    experiment_name: str = "mineworld_bc_train"
) -> tuple[dict, dict, dict]:
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

def _load_actions(json_path: Path) -> List[Dict]:
    if not json_path.exists():
        raise FileNotFoundError(f"Missing action log for {json_path}")
    actions: List[Dict] = []
    with json_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            actions.append(json.loads(line))
    return actions


class MineWorldFrameDataset(Dataset):
    """
    Sliding-window dataset over MineWorld trajectories.
    """

    def __init__(
        self,
        data_root: Path,
        context_frames: int = 8,
        resize: int = 256,
        recursive: bool = True,
    ) -> None:
        self.data_root = Path(data_root)
        self.context_frames = context_frames
        self.resize = resize
        self.samples: list[tuple[Path, Path, int]] = []
        self.actions_cache: dict[Path, list[dict]] = {}
        self.mc_dataset = MCDataset()
        self.mc_dataset.make_action_vocab(action_vocab_offset=0)
        self.action_vocab = self.mc_dataset.action_vocab
        self.action_vocab_size = len(self.action_vocab)
        self.action_length = self.mc_dataset.action_length
        self.normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        self._capture_cache: dict[Path, cv2.VideoCapture] = {}
        self._last_frame_index: dict[Path, int] = {}
        self._build_index(recursive=recursive)
        self._max_open_captures = 8
        self._capture_order: list[Path] = []

    def _build_index(self, recursive: bool) -> None:
        pattern = "**/*.mp4" if recursive else "*.mp4"
        video_paths = sorted(self.data_root.glob(pattern))
        if not video_paths:
            raise FileNotFoundError(
                f"No .mp4 files found under {self.data_root}. "
                "Pass the MineWorld data directory via --data-root."
            )

        for video_path in video_paths:
            action_path = video_path.with_suffix(".jsonl")
            try:
                actions = self.actions_cache.setdefault(
                    action_path, _load_actions(action_path)
                )
            except FileNotFoundError:
                continue

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                cap.release()
                continue
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            usable = min(frame_count, len(actions))
            for frame_idx in range(self.context_frames - 1, usable):
                self.samples.append((video_path, action_path, frame_idx))

        if not self.samples:
            raise RuntimeError(
                "Found videos but no usable context windows. Ensure .jsonl files align with videos."
            )

    def _get_capture(self, video_path: Path) -> cv2.VideoCapture:
        cap = self._capture_cache.get(video_path)
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open {video_path}")
            self._capture_cache[video_path] = cap
            self._last_frame_index[video_path] = -1
            self._capture_order.append(video_path)
            if len(self._capture_order) > self._max_open_captures:
                oldest = self._capture_order.pop(0)
                old_cap = self._capture_cache.pop(oldest, None)
                if old_cap is not None and old_cap.isOpened():
                    old_cap.release()
                self._last_frame_index.pop(oldest, None)
        else:
            # refresh LRU order
            if video_path in self._capture_order:
                self._capture_order.remove(video_path)
            self._capture_order.append(video_path)
        return cap

    def _read_frame(self, video_path: Path, frame_idx: int) -> np.ndarray:
        cap = self._get_capture(video_path)
        last_idx = self._last_frame_index.get(video_path, -1)
        if last_idx + 1 == frame_idx:
            success, frame = cap.read()
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = cap.read()
        if not success:
            raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
        self._last_frame_index[video_path] = frame_idx
        return frame

    def _frame_to_tensor(self, frame: np.ndarray) -> torch.Tensor:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (self.resize, self.resize), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        tensor = self.normalize(tensor)
        return tensor

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_capture_cache"] = {}
        state["_last_frame_index"] = {}
        state["_capture_order"] = []
        return state

    def __del__(self) -> None:
        for cap in self._capture_cache.values():
            cap.release()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        video_path, action_path, frame_idx = self.samples[index]
        frame = self._read_frame(video_path, frame_idx)
        tensor = self._frame_to_tensor(frame)

        json_action = self.actions_cache[action_path][frame_idx]
        env_action, _ = self.mc_dataset.json_action_to_env_action(json_action)
        action_indices = self.mc_dataset.get_action_index_from_actiondict(env_action, action_vocab_offset=0)
        label = torch.tensor(action_indices, dtype=torch.long)
        return tensor, label


def train_initial_bc(
    data_root: Path,
    fraction: float,
    epochs: int,
    batch_size: int,
    context_frames: int,
    recursive: bool,
    resize: int,
    output: Path,
    dataset_cfg: Dict,
    policy_cfg_dict: Dict,
    dataset_name: str,
    exp_cfg: Dict,
    processed_root: Path | None,
    chunk_cache_size: int,
    require_processed: bool,
) -> None:
    policy_cfg = PolicyConfig(**policy_cfg_dict)
    action_keys_cfg = dataset_cfg.get("action_keys")
    if action_keys_cfg:
        policy_cfg.action_dim = len(action_keys_cfg)
    else:
        action_keys_cfg = None
    action_classes = dataset_cfg.get("action_classes")
    if action_classes is not None:
        policy_cfg.action_dim = int(action_classes)

    dataset: Dataset
    processed_used = False
    if processed_root is not None:
        try:
            dataset = ProcessedMineWorldDataset(
                processed_root=processed_root,
                cache_size=max(1, chunk_cache_size),
            )
            processed_used = True
            context_frames = dataset.context_frames  # type: ignore[attr-defined]
            resize = dataset.resize  # type: ignore[attr-defined]
        except (FileNotFoundError, ValueError) as exc:
            if require_processed:
                raise
            print(
                f"[train_initial_bc] Processed dataset unavailable ({exc}). "
                "Falling back to raw MineWorld videos."
            )
            processed_used = False

    if not processed_used:
        dataset = MineWorldFrameDataset(
            data_root=data_root,
            context_frames=context_frames,
            resize=resize,
            recursive=recursive,
        )

    action_length = dataset.action_length
    action_vocab_size = dataset.action_vocab_size
    policy_cfg.action_dim = action_vocab_size * action_length

    train_cfg = exp_cfg.get("training", {})
    opt_cfg = exp_cfg.get("optimizer", {})

    subset_len = max(1, int(len(dataset) * fraction))
    indices = list(range(len(dataset)))
    random_seed = int(train_cfg.get("seed", dataset_cfg.get("seed", 0)))
    random.seed(random_seed)
    random.shuffle(indices)

    val_fraction = float(train_cfg.get("val_fraction", dataset_cfg.get("val_fraction", 0.1)))
    if subset_len <= 1 or val_fraction <= 0:
        val_fraction = 0.0
        val_split = 0
    else:
        val_split = max(1, int(round(subset_len * val_fraction)))
        if val_split >= subset_len:
            val_split = max(1, subset_len - 1)
    val_indices = indices[:val_split]
    train_indices = indices[val_split:subset_len]
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    train_batch_size = batch_size
    train_epochs = epochs
    train_workers = int(train_cfg.get("num_workers", 4))
    val_batch_size = int(train_cfg.get("val_batch_size", train_batch_size))
    val_workers = int(train_cfg.get("val_num_workers", max(1, train_workers // 2)))
    checkpoint_interval = int(train_cfg.get("checkpoint_interval", 0))

    wandb.init(
        project="foundation-dagger",
        job_type="bc-bootstrap",
        config={
            "dataset": dataset_name,
            "data_root": str(data_root),
            "fraction": fraction,
            "epochs": train_epochs,
            "batch_size": train_batch_size,
            "context_frames": context_frames,
            "resize": resize,
            "val_fraction": val_fraction,
            "seed": random_seed,
            "action_vocab_size": action_vocab_size,
            "action_length": action_length,
            "policy_output_dim": policy_cfg.action_dim,
            "train_samples": len(train_subset),
            "val_samples": len(val_subset),
            "processed": processed_used,
            "processed_root": str(processed_root) if processed_root is not None else None,
            "chunk_cache_size": chunk_cache_size if processed_used else None,
        },
    )

    prefetch_factor = int(train_cfg.get("prefetch_factor", 2))
    loader_kwargs = dict(
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=train_workers,
        pin_memory=True,
        persistent_workers=train_workers > 0,
    )
    if train_workers > 0:
        loader_kwargs["prefetch_factor"] = max(prefetch_factor, 1)
    loader = DataLoader(train_subset, **loader_kwargs)

    model = FoundationBCPolicy(policy_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    def save_checkpoint(suffix: str | None = None) -> Path:
        if suffix:
            checkpoint_path = output.with_name(f"{output.stem}_{suffix}{output.suffix}")
        else:
            checkpoint_path = output
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        wandb.save(str(checkpoint_path))
        return checkpoint_path

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(opt_cfg.get("lr", 3e-4)),
        betas=tuple(opt_cfg.get("betas", [0.9, 0.99])),
        weight_decay=float(opt_cfg.get("weight_decay", 1e-4)),
    )
    criterion = nn.CrossEntropyLoss()

    val_prefetch_factor = int(train_cfg.get("val_prefetch_factor", prefetch_factor))
    val_loader_kwargs = dict(
        batch_size=val_batch_size,
        num_workers=val_workers,
        pin_memory=True,
        shuffle=False,
        persistent_workers=val_workers > 0,
    )
    if val_workers > 0:
        val_loader_kwargs["prefetch_factor"] = max(val_prefetch_factor, 1)
    val_loader = DataLoader(val_subset, **val_loader_kwargs)

    def run_epoch(data_loader, train: bool):
        running_loss = 0.0
        running_acc = 0.0
        count = 0
        if train:
            model.train()
        else:
            model.eval()
        iterator = tqdm(
            data_loader,
            desc="train" if train else "val",
            leave=False,
            ncols=80,
        )
        for observations, labels in iterator:
            observations = observations.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.set_grad_enabled(train):
                logits = model(observations)
                logits = logits.view(observations.size(0), action_length, action_vocab_size)
                loss = criterion(logits.view(-1, action_vocab_size), labels.view(-1))
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            running_loss += loss.item() * observations.size(0)
            preds = logits.argmax(dim=-1)
            token_acc = (preds == labels).float().mean(dim=1)
            running_acc += token_acc.sum().item()
            count += observations.size(0)
            if count > 0:
                iterator.set_postfix(
                    loss=f"{running_loss / count:.4f}",
                    acc=f"{running_acc / count:.4f}",
                )
        if count == 0:
            return 0.0, 0.0
        return running_loss / count, running_acc / count

    for epoch in range(train_epochs):
        train_loss, train_acc = run_epoch(loader, train=True)
        if len(val_subset) > 0:
            val_loss, val_acc = run_epoch(val_loader, train=False)
        else:
            val_loss, val_acc = None, None
        metrics = {"train/loss": train_loss, "train/acc": train_acc}
        if val_loss is not None:
            metrics["val/loss"] = val_loss
            metrics["val/acc"] = val_acc
        wandb.log(metrics, step=epoch + 1)
        if val_loss is not None:
            print(
                f"Epoch {epoch + 1}/{epochs} - loss: {train_loss:.4f} - acc: {train_acc:.4f} "
                f"- val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}"
            )
        else:
            print(
                f"Epoch {epoch + 1}/{epochs} - loss: {train_loss:.4f} - acc: {train_acc:.4f}"
            )
        if checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0:
            ckpt_path = save_checkpoint(f"epoch{epoch + 1:03d}")
            print(f"Saved checkpoint to {ckpt_path}")

    final_path = save_checkpoint(None)
    wandb.finish()
    print(f"Saved checkpoint to {final_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=str,
        default="mineworld_frames",
        help="Hydra dataset config name.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Directory containing MineWorld .mp4 videos and matching .jsonl logs. Overrides config if provided.",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=None,
        help="Directory containing preprocessed MineWorld chunks.",
    )
    parser.add_argument(
        "--fraction", 
        type=float, 
        default=1.0 / 6, 
        help="Dataset fraction to use"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=5, 
        help="Number of training epochs"
    )
    parser.add_argument(
        "--chunk-cache-size",
        type=int,
        default=2,
        help="Number of processed chunks to keep in memory per worker.",
    )
    parser.add_argument(
        "--require-processed",
        action="store_true",
        help="Fail if the processed dataset is unavailable.",
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=32, 
        help="Training batch size"
    )
    parser.add_argument(
        "--context-frames",
        type=int,
        default=None,
        help="Number of frames in the conditioning window. Overrides config if provided.",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=None,
        help="Spatial size to resize frames to. Overrides config if provided.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Disable recursive search for videos under the data root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs") / "bootstrap_bc.ckpt",
        help="Path to save the trained checkpoint.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset_cfg, policy_cfg_dict, exp_cfg = _load_hydra_configs(
        dataset_name=args.dataset, algorithm_name='mineworld_bc', experiment_name='mineworld_bc_train'
    )
    dataset_name = args.dataset or dataset_cfg.get("name", "mineworld_frames")

    data_root_value = args.data_root or dataset_cfg.get("data_root")
    if data_root_value is None:
        raise ValueError("Dataset config must define data_root; alternatively pass --data-root.")
    data_root = (Path(data_root_value)
                 if Path(data_root_value).is_absolute()
                 else (ROOT_DIR / data_root_value).resolve())

    if args.fraction is not None:
        fraction = args.fraction
    else:
        fraction = float(dataset_cfg.get("fraction", 0.1))

    if args.epochs is not None:
        epochs = args.epochs
    else:
        epochs = int(dataset_cfg.get("epochs", 5))

    if args.batch_size is not None:
        batch_size = args.batch_size
    else:
        batch_size = int(dataset_cfg.get("batch_size", 32))

    if args.context_frames is not None:
        context_frames = args.context_frames
    else:
        context_frames = int(dataset_cfg.get("context_frames", 8))

    if args.resize is not None:
        resize = args.resize
    else:
        resize = int(dataset_cfg.get("resize", dataset_cfg.get("resolution", 256)))

    if args.output is not None:
        output = args.output
    else:
        output = Path(dataset_cfg.get("output", "checkpoints/bc_policy.ckpt"))
    if not isinstance(output, Path):
        output = Path(output)
    if not output.is_absolute():
        output = (ROOT_DIR / output).resolve()

    recursive = dataset_cfg.get("recursive", True)
    if args.no_recursive:
        recursive = False

    processed_root_value = args.processed_root or dataset_cfg.get("processed_root")
    processed_root = None
    if processed_root_value is not None:
        processed_root = Path(processed_root_value)
        if not processed_root.is_absolute():
            processed_root = (ROOT_DIR / processed_root).resolve()

    training_cfg = exp_cfg.get("training", {})
    chunk_cache_size = int(training_cfg.get("chunk_cache_size", args.chunk_cache_size))

    train_initial_bc(
        data_root=data_root,
        fraction=fraction,
        epochs=epochs,
        batch_size=batch_size,
        context_frames=context_frames,
        recursive=recursive,
        resize=resize,
        output=output,
        dataset_cfg=dataset_cfg,
        policy_cfg_dict=policy_cfg_dict,
        dataset_name=dataset_name,
        exp_cfg=exp_cfg,
        processed_root=processed_root,
        chunk_cache_size=max(1, chunk_cache_size),
        require_processed=args.require_processed,
    )
