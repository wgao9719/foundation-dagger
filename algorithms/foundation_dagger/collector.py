from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from datasets.video import MinecraftAdvancedVideoDataset
from .dagger_module import FoundationDaggerModule
from .planner import CEMPlanner, MPCConfig
from .vlm import VisionLanguageScorer
from algorithms.mineworld.model import MineWorldModel

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "configurations"
ALGO_BASE_PATH = CONFIG_ROOT / "algorithm" / "dfot_video.yaml"
# Fully-resolved defaults to avoid Hydra placeholders in downstream code.
DEFAULT_MINECRAFT_DATASET = {
    "name": "minecraft",
    "save_dir": "data/minecraft",
    "latent": {
        "enable": False,
        "type": "pre_sample",
        "suffix": "1cd9pgpb",
        "downsampling_factor": [1, 8],
        "num_channels": 4,
    },
    "resolution": 256,
    "observation_shape": [3, 256, 256],
    "max_frames": 50,
    "n_frames": 50,
    "context_length": 25,
    "frame_skip": 2,
    "filter_min_len": None,
    "preload": False,
    "subdataset_size": 1920000,
    "num_eval_videos": 768,
    "external_cond_dim": 4,
    "external_cond_stack": True,
    "external_cond_processing": "mask_first",
    "data_mean": [[[0.557]], [[0.222]], [[0.416]], [[-0.847]]],
    "data_std": [[[4.268]], [[4.953]], [[5.722]], [[7.336]]],
}


def _resolve_dataset_placeholders(config: Dict, root: Dict) -> Dict:
    def _resolve(value):
        if isinstance(value, str) and value.startswith("${dataset.") and value.endswith("}"):
            key = value[len("${dataset.") : -1]
            if key not in root:
                raise KeyError(f"Missing dataset field '{key}' required for interpolation")
            return root[key]
        if isinstance(value, dict):
            return {k: _resolve(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_resolve(v) for v in value]
        return value

    return {k: _resolve(v) for k, v in config.items()}


def _resolve_config_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (REPO_ROOT / path_str).resolve()
    return path


def _deep_merge(base, addition):
    if not isinstance(base, dict) or not isinstance(addition, dict):
        return addition
    merged = dict(base)
    for key, value in addition.items():
        if key in merged:
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config_tree(path: Path) -> dict:
    cfg = OmegaConf.load(str(path))
    data = OmegaConf.to_container(cfg, resolve=False)
    defaults = data.pop("defaults", [])
    merged: Dict = {}
    for entry in defaults:
        if entry == "_self_":
            continue
        if isinstance(entry, str):
            entry_path = entry.lstrip("/")
            if entry.startswith("/"):
                sub_path = CONFIG_ROOT / f"{entry_path}.yaml"
            else:
                sub_path = path.parent / f"{entry}.yaml"
            merged = _deep_merge(merged, _load_config_tree(sub_path))
        elif isinstance(entry, dict):
            for key, value in entry.items():
                value_path = str(value).lstrip("/")
                if str(value).startswith("/"):
                    sub_path = CONFIG_ROOT / key / f"{value_path}.yaml"
                else:
                    sub_path = path.parent / key / f"{value}.yaml"
                group_dict = merged.get(key, {})
                merged[key] = _deep_merge(group_dict, _load_config_tree(sub_path))
    merged = _deep_merge(merged, data)
    return merged


@dataclass
class CollectorConfig:
    output_dir: str = "data/dagger"
    batch_size: int = 4
    max_episodes: int = 32
    context_frames: int = 8
    horizon: int = 8
    policy_checkpoint: Optional[str] = None
    overwrite: bool = False


class FoundationDaggerCollector:
    """Runs policy/world-model rollouts and aggregates the DAgger buffer."""

    def __init__(
        self,
        policy_cfg: DictConfig,
        dataset_cfg: DictConfig,
        world_model_cfg: DictConfig,
        planner_cfg: DictConfig,
        vlm_cfg: DictConfig,
        collector_cfg: DictConfig,
    ) -> None:
        if isinstance(policy_cfg, str):
            policy_cfg = OmegaConf.load(policy_cfg)
        algo_cfg = DictConfig(OmegaConf.to_container(policy_cfg, resolve=True))
        collector = CollectorConfig(**OmegaConf.to_container(collector_cfg, resolve=True))
        self.collector_cfg = collector
        if collector.policy_checkpoint:
            self.policy = FoundationDaggerModule.load_from_checkpoint(
                collector.policy_checkpoint, cfg=algo_cfg
            )
        else:
            self.policy = FoundationDaggerModule(algo_cfg)
        self.policy.eval()

        if isinstance(dataset_cfg, str):
            dataset_cfg = _load_config_tree(_resolve_config_path(dataset_cfg))
        else:
            dataset_cfg = OmegaConf.to_container(dataset_cfg, resolve=True)
        dataset_cfg = self._finalize_dataset_cfg(dataset_cfg or {})
        dataset_conf = OmegaConf.create(dataset_cfg)
        OmegaConf.resolve(dataset_conf)
        dataset = MinecraftAdvancedVideoDataset(dataset_conf, split="validation")
        self.loader = DataLoader(
            dataset,
            batch_size=collector.batch_size,
            shuffle=True,
            num_workers=dataset_cfg.get("num_workers", 2),
        )

        dataset_values = OmegaConf.to_container(dataset_conf, resolve=True)
        dataset_for_algo = deepcopy(dataset_values)
        latent_cfg = dataset_for_algo.setdefault("latent", {})
        latent_cfg.setdefault("type", "pre_sample")
        latent_cfg.setdefault("suffix", "1cd9pgpb")
        latent_cfg.setdefault("downsampling_factor", [1, 8])
        latent_cfg.setdefault("num_channels", 4)
        latent_cfg["enable"] = True

        if isinstance(world_model_cfg, str):
            world_model_cfg = _load_config_tree(_resolve_config_path(world_model_cfg))
        else:
            world_model_cfg = OmegaConf.to_container(world_model_cfg, resolve=True)
        algorithm_overrides = world_model_cfg.pop("algorithm", {})
        if isinstance(algorithm_overrides, str):
            algorithm_overrides = _load_config_tree(_resolve_config_path(algorithm_overrides))

        algo_base = _load_config_tree(ALGO_BASE_PATH)
        algo_dict = _deep_merge(algo_base, algorithm_overrides or {})
        root_cfg = OmegaConf.create({"dataset": dataset_for_algo, "algorithm": algo_dict, "debug": False})
        OmegaConf.resolve(root_cfg)
        algo_conf = root_cfg.algorithm
        wm_cfg = WorldModelConfig(
            checkpoint=world_model_cfg.get("checkpoint", "pretrained:DFoT_MCRAFT.ckpt"),
            algorithm=algo_conf,
            device=world_model_cfg.get("device", "cuda"),
            context_frames=world_model_cfg.get("context_frames", 8),
        )
        self.world_model = MineWorldModel(wm_cfg)

        if isinstance(vlm_cfg, str):
            vlm_cfg = OmegaConf.load(vlm_cfg)
        self.vlm = VisionLanguageScorer(OmegaConf.to_object(vlm_cfg))

        if isinstance(planner_cfg, str):
            planner_cfg = OmegaConf.load(planner_cfg)
        mpc_cfg = MPCConfig(**OmegaConf.to_container(planner_cfg, resolve=True))
        self.planner = CEMPlanner(
            self.world_model,
            self.vlm,
            action_dim=self.world_model.model.external_cond_dim,
            cfg=mpc_cfg,
        )

        self.output_dir = Path(collector.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.buffer: List[Dict[str, torch.Tensor]] = []

    def _policy_action(self, frame: torch.Tensor) -> torch.Tensor:
        logits = self.policy(frame)
        return torch.argmax(logits, dim=-1)

    def _one_hot(self, actions: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.one_hot(
            actions, self.world_model.model.external_cond_dim
        ).float()

    def _finalize_dataset_cfg(self, cfg: Dict) -> Dict:
        cfg_dict = deepcopy(DEFAULT_MINECRAFT_DATASET)
        cfg_dict = _deep_merge(cfg_dict, cfg)
        latent = cfg_dict.setdefault("latent", {})
        latent.setdefault("enable", False)
        latent.setdefault("type", "pre_sample")
        latent.setdefault("suffix", None)
        latent.setdefault("downsampling_factor", [1, 8])
        latent.setdefault("num_channels", 4)
        max_frames = cfg_dict.get("max_frames", 32)
        n_frames = cfg_dict.get("n_frames")
        if not isinstance(n_frames, int):
            cfg_dict["n_frames"] = max_frames
        resolution = cfg_dict.get("resolution", 256)
        observation = cfg_dict.get("observation_shape")
        if not isinstance(observation, list) or any(
            isinstance(item, str) and item.startswith("${dataset.") for item in observation
        ):
            cfg_dict["observation_shape"] = [3, resolution, resolution]
        cfg_dict = _resolve_dataset_placeholders(cfg_dict, cfg_dict)
        return cfg_dict

    def run(self) -> Path:
        total = 0
        for batch in self.loader:
            videos = batch["videos"][:, : self.collector_cfg.context_frames]
            context = videos.to(self.world_model.device)
            bc_last = context[:, -1]
            bc_actions = self._policy_action(bc_last)
            bc_sequence = bc_actions.unsqueeze(1).repeat(
                1, self.collector_cfg.horizon
            )
            bc_rollout = self.world_model.rollout(context, self._one_hot(bc_sequence))
            bc_score = self.vlm.score_frames(torch.unbind(bc_rollout[:, -1]))
            plan_actions, plan_score = self.planner.plan(context)
            better = plan_score > bc_score
            selected_actions = torch.where(
                better[:, None], plan_actions, bc_sequence
            )
            selected_obs = context[:, -1]
            self.buffer.append(
                {
                    "observations": selected_obs.cpu(),
                    "actions": selected_actions[:, 0].cpu(),
                }
            )
            total += videos.shape[0]
            if total >= self.collector_cfg.max_episodes:
                break
        return self._flush()

    def _flush(self) -> Path:
        if not self.buffer:
            raise RuntimeError("No samples collected for DAgger.")
        observations = torch.stack([sample["observations"] for sample in self.buffer])
        actions = torch.stack([sample["actions"] for sample in self.buffer])
        data_path = self.output_dir / "dagger_iter.npz"
        if data_path.exists() and not self.collector_cfg.overwrite:
            raise FileExistsError(f"{data_path} exists; enable overwrite to replace.")
        np.savez(
            data_path,
            observations=observations.numpy(),
            actions=actions.numpy(),
        )
        manifest = {"files": [data_path.name]}
        (self.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        self.buffer.clear()
        return data_path
