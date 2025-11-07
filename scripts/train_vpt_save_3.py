"""Train an initial BC policy on MineWorld rollouts.

The script ingests MineWorld-style trajectories (paired ``.mp4`` videos and
``.jsonl`` action logs) and produces a supervised policy checkpoint compatible
with the ``foundation_dagger`` module. Each training sample stacks the full
``N``-frame context window and maximises the joint log-likelihood of the final
frame's hierarchical action under the VPT policy head, enabling
Transformer-style policies to leverage temporal attention while still
supporting single-frame heads. Only a user-specified fraction of the available
samples is used to keep the bootstrap stage quick.

Example
-------
python -m scripts.train_vpt \
  --algorithm vpt_bc
  --dataset mineworld_frames \
  --experiment mineworld_bc_train \
  --data-root /workspace/foundation-dagger/foundation-dagger/data/mineworld \
  --fraction 0.3 \
  --epochs 20 \
  --batch-size 32 \
  --output checkpoints/bc_checkpoints/bc_policy.ckpt
"""


from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from typing import Dict, List, Optional
from gym3.types import DictType

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
import wandb

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from datasets.mineworld_data.mineworld_frame_dataset import MineWorldFrameDataset

CONFIG_DIR = ROOT_DIR / "configurations"

from algorithms.foundation_dagger.vpt_model.policy import MinecraftAgentPolicy
from algorithms.foundation_dagger.vpt_model.action_mapping import CameraHierarchicalMapping
from algorithms.foundation_dagger.vpt_model.load_vpt_config import VPTConfig, parse_policy_config
from algorithms.foundation_dagger.vpt_model.tree_util import tree_map
from algorithms.foundation_dagger.vpt_model.actions import ActionTransformer, Buttons


def _load_hydra_configs(
    dataset_name: str,
    algorithm_name: str,
    experiment_name: str,
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
    policy_cfg = OmegaConf.to_container(cfg.algorithm, resolve=True)
    exp_cfg = OmegaConf.to_container(cfg.experiment, resolve=True)
    return dataset_cfg, policy_cfg, exp_cfg

def train_vpt_bc(
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
    algorithm_name: str,
    experiment_name: str,
    dataset_name: str,
    exp_cfg: Dict,
) -> None:
    # get configs
    policy_cfg: VPTConfig = parse_policy_config(policy_cfg_dict)

    #initialize dataset
    dataset = MineWorldFrameDataset(
        data_root=data_root,
        context_frames=context_frames,
        recursive=recursive,
        max_open_captures=int(dataset_cfg.get("max_open_captures", 2)),
    )
    print("MineWorldFrameDataset action transformer:", dataset.action_transformer.camera_binsize,
           dataset.action_transformer.camera_maxval,
           dataset.action_transformer.camera_mu,
           dataset.action_transformer.camera_quantization_scheme)

    train_cfg = exp_cfg.get("training", {})
    opt_cfg = exp_cfg.get("optimizer", {})

    # get subset length and indices
    subset_len = max(1, int(len(dataset) * fraction))
    indices = list(range(len(dataset)))
    random_seed = int(train_cfg.get("seed", dataset_cfg.get("seed", 0)))
    random.seed(random_seed)

    # Shuffle at the video level while preserving chronological ordering per video.
    video_to_indices: Dict[int, List[int]] = {}
    for idx, (_, vid, _) in enumerate(dataset.samples):
        video_to_indices.setdefault(int(vid), []).append(idx)
    video_ids = list(video_to_indices.keys())
    random.shuffle(video_ids)
    indices = []
    for vid in video_ids:
        indices.extend(video_to_indices[vid])

    # split dataset into training and validation
    val_fraction = float(train_cfg.get("val_fraction", dataset_cfg.get("val_fraction", 0.1)))
    if subset_len <= 1 or val_fraction <= 0:
        val_fraction = 0.0
        val_split = 0
    else:
        val_split = max(1, int(round(subset_len * val_fraction)))
        if val_split >= subset_len:
            val_split = max(1, subset_len - 1)
    val_indices = sorted(indices[:val_split])
    train_indices = sorted(indices[val_split:subset_len])
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    # load in training hparameters
    train_batch_size = batch_size
    train_epochs = epochs
    train_workers = int(train_cfg.get("num_workers", 4))
    val_batch_size = int(train_cfg.get("val_batch_size", train_batch_size))
    val_workers = int(train_cfg.get("val_num_workers", max(1, train_workers // 2)))
    checkpoint_interval = int(train_cfg.get("checkpoint_interval", 0))
    grad_clip_norm = float(train_cfg.get("grad_clip_norm", 1.0))
    
    # initialize wandb
    wandb.init(
        project="foundation-dagger",
        job_type="bc-bootstrap",
        config={
            "dataset": dataset_name,
            "algorithm": algorithm_name,
            "experiment": experiment_name,
            "data_root": str(data_root),
            "fraction": fraction,
            "epochs": train_epochs,
            "batch_size": train_batch_size,
            "context_frames": context_frames,
            "resize": resize,
            "val_fraction": val_fraction,
            "seed": random_seed,
            "train_samples": len(train_subset),
            "val_samples": len(val_subset),
            "grad_clip_norm": grad_clip_norm if grad_clip_norm is not None else 0.0,
        },
    )

    # initialize data loader
    prefetch_factor = int(train_cfg.get("prefetch_factor", 2))
    loader_kwargs = dict(
        batch_size=train_batch_size,
        shuffle=False,
        num_workers=train_workers,
        pin_memory=True,
        persistent_workers=False,  # Disabled to avoid stale VideoCapture objects across epochs
    )
    if train_workers > 0:
        loader_kwargs["prefetch_factor"] = max(prefetch_factor, 1)
    loader = DataLoader(train_subset, **loader_kwargs)

    # initialize model
    action_mapper = CameraHierarchicalMapping(n_camera_bins=11)
    action_space = action_mapper.get_action_space_update()
    action_space = DictType(**action_space)

    model_args = policy_cfg.model.get("args", {})
    policy_kwargs = model_args.get("net", {}).get("args", {})
    pi_head_kwargs = model_args.get("pi_head_opts", {})

    model = MinecraftAgentPolicy(action_space, policy_kwargs, pi_head_kwargs)

    # initialize weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = "/Users/willi1/foundation-dagger/diffusion-forcing-transformer/checkpoints/vpt/foundation-model-1x.weights"
    # path = "/workspace/foundation-dagger/foundation-dagger/checkpoints/vpt/foundation-model-1x.weights"
    # model.load_state_dict(torch.load(path, map_location=device), strict=False)
    state_dict = torch.load(path, map_location=device)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        warn_lines = ["VPT checkpoint load was non-strict:"]
        if missing_keys:
            warn_lines.append(f"  missing keys ({len(missing_keys)}): {sorted(missing_keys)[:10]}{' ...' if len(missing_keys) > 10 else ''}")
        if unexpected_keys:
            warn_lines.append(f"  unexpected keys ({len(unexpected_keys)}): {sorted(unexpected_keys)[:10]}{' ...' if len(unexpected_keys) > 10 else ''}")
        print("\n".join(warn_lines))
    model = model.to(device)

    def _move_state_to_device(state):
        if state is None:
            return None
        if isinstance(state, (list, tuple)):
            return type(state)(_move_state_to_device(s) for s in state)
        return state.to(device, non_blocking=True)

    def _detach_state(state):
        if state is None:
            return None
        return tree_map(lambda x: x.detach() if isinstance(x, torch.Tensor) else x, state)

    # Diagnostics to verify checkpoint alignment with dataset labels.
    print(f"[diagnostics] checkpoint load -> missing: {len(missing_keys)} | unexpected: {len(unexpected_keys)}")
    diag_loader = DataLoader(
        train_subset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    if len(train_subset) == 0:
        print("[diagnostics] dataset empty; skipping log-prob check.")
    else:
        diag_limit = min(64, len(train_subset))
        total_logprobs: List[float] = []
        buttons_logprobs: List[float] = []
        camera_logprobs: List[float] = []
        raw_cameras: List[np.ndarray] = []
        env_actions: List[Dict[str, object]] = []
        diag_state_cache: Dict[int, object] = {}
        diag_last_frame: Dict[int, int] = {}
        buttons_targets: List[torch.Tensor] = []
        camera_targets: List[torch.Tensor] = []
        final_pd_buffers: List[Dict[str, torch.Tensor]] = []
        with torch.no_grad():
            for step_i, (obs_seq, _, video_id_tensor, frame_idx_tensor) in enumerate(diag_loader):
                if step_i >= diag_limit:
                    break
                obs_seq = obs_seq.to(device, non_blocking=True)
                vid = int(video_id_tensor.item())
                frame_idx = int(frame_idx_tensor.item())

                agent_action_np = dataset.get_agent_action(vid, frame_idx)
                agent_action = {
                    key: agent_action_np[key].unsqueeze(0).to(device, non_blocking=True)
                    for key in agent_action_np.keys()
                }

                cached_state = diag_state_cache.get(vid)
                last_seen = diag_last_frame.get(vid)
                if cached_state is None or last_seen is None or frame_idx <= last_seen:
                    state_in = _move_state_to_device(model.initial_state(1))
                    first_diag = torch.zeros(1, obs_seq.size(1), dtype=torch.bool, device=device)
                    first_diag[:, 0] = True
                else:
                    state_in = _move_state_to_device(cached_state)
                    first_diag = torch.zeros(1, obs_seq.size(1), dtype=torch.bool, device=device)

                policy_tuple_diag, state_out = model({"img": obs_seq}, first_diag, state_in)
                final_pd_diag = tree_map(lambda x: x[:, -1], policy_tuple_diag[0])
                final_pd_buffers.append(tree_map(lambda t: t.detach().cpu(), final_pd_diag))
                total_lp = model.pi_head.logprob(agent_action, final_pd_diag)
                buttons_lp = model.pi_head._modules["buttons"].logprob(
                    agent_action["buttons"], final_pd_diag["buttons"]
                )
                camera_lp = model.pi_head._modules["camera"].logprob(
                    agent_action["camera"], final_pd_diag["camera"]
                )

                total_logprobs.append(total_lp.item())
                buttons_logprobs.append(buttons_lp.item())
                camera_logprobs.append(camera_lp.item())
                buttons_targets.append(agent_action["buttons"].detach().cpu())
                camera_targets.append(agent_action["camera"].detach().cpu())

                diag_state_cache[vid] = _detach_state(state_out)
                diag_last_frame[vid] = frame_idx

                # Retrieve underlying continuous camera deltas for context
                step_entries = dataset.video_step_infos[int(vid)]
                step_info = next(
                    (entry for entry in step_entries if entry["frame_idx"] == frame_idx),
                    None,
                )
                if step_info is not None:
                    raw_cameras.append(np.asarray(step_info["action"]["camera"], dtype=np.float32))
                    env_actions.append(step_info["action"])

        if not total_logprobs:
            print("[diagnostics] warning: unable to compute log-probs; skipping report.")
        else:
            print(
                "[diagnostics] log-prob means ->"
                f" total {np.mean(total_logprobs):.3f}"
                f" | buttons {np.mean(buttons_logprobs):.3f}"
                f" | camera {np.mean(camera_logprobs):.3f}"
            )

            buttons_stack = torch.cat(buttons_targets, dim=0) if buttons_targets else None
            camera_stack = torch.cat(camera_targets, dim=0) if camera_targets else None
            if buttons_stack is not None and camera_stack is not None:
                print(
                    "[diagnostics] label ranges -> buttons",
                    (int(buttons_stack.min().item()), int(buttons_stack.max().item())),
                    "| camera indices",
                    (int(camera_stack.min().item()), int(camera_stack.max().item())),
                )
            else:
                print("[diagnostics] warning: missing action targets; skipping range report.")

            raw_cameras_np = np.stack(raw_cameras, axis=0) if raw_cameras else np.zeros((0, 2), dtype=np.float32)
            if raw_cameras_np.size == 0:
                print("[diagnostics] warning: unable to fetch raw camera deltas; skipping quantiser check.")
                return

            if buttons_stack is None or camera_stack is None or not final_pd_buffers:
                print("[diagnostics] insufficient data for detailed action diagnostics; skipping quantiser check.")
            else:
                with torch.no_grad():
                    agent_actions_diag = {
                        "buttons": buttons_stack.to(device, non_blocking=True),
                        "camera": camera_stack.to(device, non_blocking=True),
                    }
                    final_pd_diag = {
                        "buttons": torch.cat([buf["buttons"] for buf in final_pd_buffers], dim=0).to(
                            device, non_blocking=True
                        ),
                        "camera": torch.cat([buf["camera"] for buf in final_pd_buffers], dim=0).to(
                            device, non_blocking=True
                        ),
                    }

                    mu_bins = dataset.action_transformer.quantizer.discretize(raw_cameras_np)
                    linear_transformer = ActionTransformer(
                        camera_binsize=2,
                        camera_maxval=10,
                        camera_mu=5,
                        camera_quantization_scheme="linear",
                    )
                    linear_bins = linear_transformer.quantizer.discretize(raw_cameras_np)
                    mu5_transformer = ActionTransformer(
                        camera_binsize=2,
                        camera_maxval=10,
                        camera_mu=5,
                        camera_quantization_scheme="mu_law",
                    )

                    alt_agent_actions = []
                    for env_action in env_actions[: len(raw_cameras_np)]:
                        env_format_alt = {
                            "camera": np.asarray(env_action["camera"], dtype=np.float32)[None],
                        }
                        for button_name in Buttons.ALL:
                            env_format_alt[button_name] = np.asarray([env_action.get(button_name, 0)], dtype=np.int64)
                        alt_factored = mu5_transformer.env2policy(env_format_alt)
                        if alt_factored["camera"].ndim == 1:
                            alt_factored = {k: v[None] for k, v in alt_factored.items()}
                        alt_agent_actions.append(dataset.action_mapper.from_factored(alt_factored))

                    if len(alt_agent_actions) == agent_actions_diag["buttons"].size(0):
                        alt_camera = torch.stack(
                            [torch.from_numpy(a["camera"][0].copy()) for a in alt_agent_actions],
                            dim=0,
                        ).to(device, non_blocking=True)
                        alt_actions_diag = {
                            "buttons": agent_actions_diag["buttons"],
                            "camera": alt_camera,
                        }
                        alt_logprob = model.pi_head.logprob(alt_actions_diag, final_pd_diag)
                        alt_camera_logprob = model.pi_head._modules["camera"].logprob(
                            alt_actions_diag["camera"], final_pd_diag["camera"]
                        )
                    else:
                        alt_logprob = None
                        alt_camera_logprob = None

                    print(
                        "[diagnostics] raw camera deltas ->",
                        "min",
                        raw_cameras_np.min(axis=0),
                        "max",
                        raw_cameras_np.max(axis=0),
                        "mu-law bins sample",
                        mu_bins[:5],
                        "linear bins sample",
                        linear_bins[:5],
                        "mu=5 mu-law bins sample",
                        mu5_transformer.quantizer.discretize(raw_cameras_np)[:5],
                    )
                    if alt_logprob is not None:
                        print(
                            "[diagnostics] alt mu=5 mu-law log-probs ->"
                            f" total {alt_logprob.mean().item():.3f}"
                            f" | camera {alt_camera_logprob.mean().item():.3f}"
                        )
                    else:
                        print("[diagnostics] alt mu=5 mu-law log-probs skipped (mismatched sample count).")

                    det_actions = model.pi_head.sample(final_pd_diag, deterministic=True)
                    combos = dataset.action_mapper.BUTTONS_IDX_TO_COMBINATION
                    cam_idx_to_combo = dataset.action_mapper.camera_idx_to_combination
                    decoded_targets = [
                        (combos[int(b.item())], cam_idx_to_combo[int(c.item())])
                        for b, c in zip(agent_actions_diag["buttons"].cpu(), agent_actions_diag["camera"].cpu())
                    ]
                    decoded_preds = [
                        (combos[int(b.item())], cam_idx_to_combo[int(c.item())])
                        for b, c in zip(det_actions["buttons"].cpu(), det_actions["camera"].cpu())
                    ]
                    print("[diagnostics] first target button/camera combos:", decoded_targets[:5])
                    print("[diagnostics] first predicted button/camera combos:", decoded_preds[:5])


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
        running_examples = 0
        state_cache: Dict[int, object] = {}
        if train:
            model.train()
        else:
            model.eval()

        iterator = tqdm(data_loader, desc="train" if train else "val", leave=False, ncols=80)

        for batch in iterator:
            observations, _, video_ids, frame_indices = batch
            observations = observations.to(device, non_blocking=True).contiguous()
            video_ids_list = video_ids.cpu().tolist()
            frame_indices_list = frame_indices.cpu().tolist()
            agent_action_list = [
                dataset.get_agent_action(int(vid), int(frame_idx))
                for vid, frame_idx in zip(video_ids_list, frame_indices_list)
            ]
            seq_len = observations.size(1) if observations.dim() >= 2 else 1
            sample_losses: List[torch.Tensor] = []
            if train:
                optimizer.zero_grad()
            with torch.set_grad_enabled(train):
                for sample_idx, (vid, frame_idx) in enumerate(zip(video_ids_list, frame_indices_list)):
                    obs_seq = observations[sample_idx : sample_idx + 1]
                    agent_action_cpu = agent_action_list[sample_idx]
                    agent_action = {
                        key: agent_action_cpu[key].unsqueeze(0).to(device, non_blocking=True)
                        for key in agent_action_cpu.keys()
                    }
                    cached_state = state_cache.get(int(vid))
                    new_episode = cached_state is None
                    if new_episode:
                        state_in = _move_state_to_device(model.initial_state(1))
                    else:
                        state_in = _move_state_to_device(cached_state)
                    first = torch.zeros(1, seq_len, dtype=torch.bool, device=device)
                    if new_episode:
                        first[:, 0] = True
                    policy_tuple, state_out = model({"img": obs_seq}, first, state_in)
                    pd_params = policy_tuple[0]
                    final_pd_params = tree_map(lambda x: x[:, -1], pd_params)
                    log_prob = model.pi_head.logprob(agent_action, final_pd_params)
                    running_loss += (-log_prob.detach()).sum().item()
                    running_examples += log_prob.numel()
                    if train:
                        sample_losses.append(-log_prob.mean())
                    detached_state = _detach_state(state_out)
                    state_cache[int(vid)] = _move_state_to_device(detached_state)
            if train and sample_losses:
                batch_loss = torch.stack(sample_losses).mean()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
            if running_examples > 0:
                iterator.set_postfix(
                    loss=f"{running_loss / running_examples:.4f}",
                )
        if running_examples == 0:
            return 0.0, {}
        return running_loss / running_examples, {}

    for epoch in range(train_epochs):
        train_loss, _ = run_epoch(loader, train=True)
        if len(val_subset) > 0:
            val_loss, _ = run_epoch(val_loader, train=False)
        else:
            val_loss, _ = None, {}
        metrics = {
            "train/loss": train_loss,
        }
        if val_loss is not None:
            metrics["val/loss"] = val_loss
        wandb.log(metrics, step=epoch + 1)
        if val_loss is not None:
            print(
                f"Epoch {epoch + 1}/{epochs} - loss: {train_loss:.4f} "
                f"- val_loss: {val_loss:.4f}"
            )
        else:
            print(
                f"Epoch {epoch + 1}/{epochs} - loss: {train_loss:.4f}"
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
        "--algorithm",
        type=str,
        default="vpt_bc",
        help="Hydra algorithm config name.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mineworld_frames",
        help="Hydra dataset config name.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="mineworld_bc_train",
        help="Hydra experiment config name.",
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
        dataset_name=args.dataset, algorithm_name=args.algorithm, experiment_name=args.experiment
    )
    algorithm_name = args.algorithm or policy_cfg_dict.get("name", "vpt_bc")
    dataset_name = args.dataset or dataset_cfg.get("name", "mineworld_frames")
    experiment_name = args.experiment or exp_cfg.get("name", "mineworld_bc_train")

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

    train_vpt_bc(
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
        algorithm_name=algorithm_name,
        experiment_name=experiment_name,
        dataset_name=dataset_name,
        exp_cfg=exp_cfg,
    )
