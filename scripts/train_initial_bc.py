"""Train an initial BC policy on MineWorld rollouts.

The script ingests MineWorld-style trajectories (paired ``.mp4`` videos and
``.jsonl`` action logs) and produces a supervised policy checkpoint compatible
with the ``foundation_dagger`` module. Each training sample stacks the full
``N``-frame context window and predicts the action token sequence associated
with the final frame, enabling Transformer-style policies to leverage temporal
attention while still supporting single-frame heads. Only a user-specified
fraction of the available samples is used to keep the bootstrap stage quick.

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

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from typing import Dict, List, Optional, Sequence, Tuple

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

from algorithms.foundation_dagger.policy import (
    BasePolicyConfig,
    VPTCausalPolicy,
    VPTPolicyConfig,
    build_policy,
    parse_policy_config,
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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
        self.samples: list[tuple[Path, Path, int, int]] = []
        self.actions_cache: dict[Path, list[dict]] = {}
        self.mc_dataset = MCDataset()
        self.mc_dataset.make_action_vocab(action_vocab_offset=0)
        self.action_vocab = self.mc_dataset.action_vocab
        self.action_vocab_size = len(self.action_vocab)
        self.action_length = self.mc_dataset.action_length
        self.normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        self._capture_cache: dict[Path, cv2.VideoCapture] = {}
        self._last_frame_index: dict[Path, int] = {}
        self._video_ids: dict[Path, int] = {}
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
            video_id = self._video_ids.setdefault(video_path, len(self._video_ids))
            for frame_idx in range(self.context_frames - 1, usable):
                json_action = actions[frame_idx]
                _, is_null_action = self.mc_dataset.json_action_to_env_action(json_action)
                if is_null_action:
                    continue
                self.samples.append((video_path, action_path, frame_idx, video_id))

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

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        video_path, action_path, frame_idx, video_id = self.samples[index]
        start_idx = frame_idx - (self.context_frames - 1)
        frame_tensors: list[torch.Tensor] = []
        for idx in range(start_idx, frame_idx + 1):
            frame = self._read_frame(video_path, idx)
            tensor = self._frame_to_tensor(frame)
            frame_tensors.append(tensor)
        frames_tensor = torch.stack(frame_tensors, dim=0)

        json_action = self.actions_cache[action_path][frame_idx]
        env_action, _ = self.mc_dataset.json_action_to_env_action(json_action)
        action_indices = self.mc_dataset.get_action_index_from_actiondict(env_action, action_vocab_offset=0)
        label = torch.tensor(action_indices, dtype=torch.long)
        return frames_tensor, label, torch.tensor(video_id, dtype=torch.long), torch.tensor(frame_idx, dtype=torch.long)


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
) -> None:
    policy_cfg: BasePolicyConfig = parse_policy_config(policy_cfg_dict)
    action_keys_cfg = dataset_cfg.get("action_keys")
    if action_keys_cfg:
        policy_cfg.action_dim = len(action_keys_cfg)
    else:
        action_keys_cfg = None
    action_classes = dataset_cfg.get("action_classes")
    if action_classes is not None:
        policy_cfg.action_dim = int(action_classes)

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
    if isinstance(policy_cfg, VPTPolicyConfig):
        # Shuffle at the video level while preserving chronological ordering inside each video.
        video_to_indices: Dict[int, List[int]] = {}
        for idx, (_, _, _, vid) in enumerate(dataset.samples):
            video_to_indices.setdefault(int(vid), []).append(idx)
        video_ids = list(video_to_indices.keys())
        random.shuffle(video_ids)
        indices = []
        for vid in video_ids:
            indices.extend(video_to_indices[vid])
    else:
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
    if isinstance(policy_cfg, VPTPolicyConfig):
        val_indices = sorted(val_indices)
        train_indices = sorted(train_indices)
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
        },
    )

    use_memory = isinstance(policy_cfg, VPTPolicyConfig)
    prefetch_factor = int(train_cfg.get("prefetch_factor", 2))
    loader_kwargs = dict(
        batch_size=train_batch_size,
        shuffle=not use_memory,
        num_workers=train_workers,
        pin_memory=True,
        persistent_workers=False,  # Disabled to avoid stale VideoCapture objects across epochs
    )
    if train_workers > 0:
        loader_kwargs["prefetch_factor"] = max(prefetch_factor, 1)
    loader = DataLoader(train_subset, **loader_kwargs)

    model = build_policy(policy_cfg)
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
        persistent_workers=False,  # Disabled to avoid stale VideoCapture objects across epochs
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
        use_transformer = isinstance(model, VPTCausalPolicy)
        mem_bank: dict[int, Optional[Sequence[torch.Tensor]]] = {} if use_transformer else {}
        last_frame_index: dict[int, int] = {} if use_transformer else {}
        sample_outputs: list[tuple[list[int], list[int]]] = []
        iterator = tqdm(
            data_loader,
            desc="train" if train else "val",
            leave=False,
            ncols=80,
        )
        for batch in iterator:
            if isinstance(batch, (tuple, list)) and len(batch) == 4:
                observations, labels, video_ids, frame_indices = batch
            else:
                observations, labels = batch
                video_ids = None
                frame_indices = None
            observations = observations.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if video_ids is not None:
                video_ids_list = video_ids.cpu().tolist()
                frame_indices_list = frame_indices.cpu().tolist()
            else:
                video_ids_list = None
                frame_indices_list = None
            with torch.set_grad_enabled(train):
                if use_transformer:
                    batch_logits: list[torch.Tensor] = []
                    for sample_idx in range(observations.size(0)):
                        vid = int(video_ids_list[sample_idx]) if video_ids_list is not None else sample_idx
                        frame_idx = int(frame_indices_list[sample_idx]) if frame_indices_list is not None else -1
                        mem = mem_bank.get(vid)
                        last_idx = last_frame_index.get(vid)
                        if last_idx is None or frame_idx != last_idx + 1:
                            mem = None
                        logits_i, new_mem = model(
                            observations[sample_idx : sample_idx + 1],
                            mems=mem,
                            return_mems=True,
                        )
                        mem_bank[vid] = new_mem
                        last_frame_index[vid] = frame_idx
                        batch_logits.append(logits_i)
                    logits = torch.cat(batch_logits, dim=0)
                else:
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
            if not train and len(sample_outputs) < 3:
                max_samples = min(observations.size(0), 3 - len(sample_outputs))
                for sample_idx in range(max_samples):
                    sample_outputs.append(
                        (
                            labels[sample_idx].detach().cpu().view(-1).tolist(),
                            preds[sample_idx].detach().cpu().view(-1).tolist(),
                        )
                    )
            if count > 0:
                iterator.set_postfix(
                    loss=f"{running_loss / count:.4f}",
                    acc=f"{running_acc / count:.4f}",
                )
        if count == 0:
            return 0.0, 0.0, sample_outputs
        return running_loss / count, running_acc / count, sample_outputs

    for epoch in range(train_epochs):
        train_loss, train_acc, _ = run_epoch(loader, train=True)
        if len(val_subset) > 0:
            val_loss, val_acc, val_samples = run_epoch(val_loader, train=False)
        else:
            val_loss, val_acc, val_samples = None, None, []
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
            if val_samples:
                print("Validation samples (target vs. prediction):")
                for idx, (target_tokens, pred_tokens) in enumerate(val_samples, start=1):
                    print(f"  Sample {idx}: target={target_tokens} pred={pred_tokens}")
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
        output = dataset_cfg.get("output", "checkpoints/bc_policy.ckpt")

    recursive = dataset_cfg.get("recursive", True)
    if args.no_recursive:
        recursive = False

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
    )
