from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from dataclasses import fields
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
import wandb
import numpy as np
from collections import defaultdict

from datasets.mineworld_data.mineworld_frame_dataset import MineWorldFrameDataset
from utils.SequentialBatchSampler import PerVideoSequentialBatchSampler

ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT_DIR / "configurations"

from algorithms.foundation_dagger.policy import (
    ActionHeadConfig,
    PolicyConfig,
    FoundationBCPolicy,
)
from algorithms.foundation_dagger.vpt_model.action_mapping import CameraHierarchicalMapping
from algorithms.foundation_dagger.vpt_model.actions import ActionTransformer

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

data_root = Path("/Users/willi1/foundation-dagger/diffusion-forcing-transformer/data/test_data")
#data_root = Path("/Users/willi1/foundation-dagger/diffusion-forcing-transformer/data/mineworld/MineRLBasaltMakeWaterfall-v0")
#data_root = Path("/Users/willi1/foundation-dagger/diffusion-forcing-transformer/data/mineworld/MineRLBasaltBuildVillageHouse-v0")
context_frames = 1
recursive = False
max_open_captures = 2

dataset = MineWorldFrameDataset(
    data_root=data_root,
    context_frames=context_frames,
    recursive=recursive,
    max_open_captures=max_open_captures
)

# print(dataset[0][0])
# print(dataset[0][1])
# print(dataset[0][2])
# print(dataset[0][3])

# print(len(dataset))
# for i in range(100):
#     print(dataset[i][1])

def _coerce_policy_config(raw_cfg: Dict, dataset: MineWorldFrameDataset) -> PolicyConfig:
    cfg = deepcopy(raw_cfg)
    valid_fields = {field.name for field in fields(PolicyConfig)}
    filtered_cfg = {key: value for key, value in cfg.items() if key in valid_fields}

    num_button_combos = len(dataset.action_mapper.BUTTONS_COMBINATIONS)
    num_camera_bins = dataset.action_mapper.n_camera_bins
    action_cfg = filtered_cfg.get("action")
    if isinstance(action_cfg, dict):
        action_cfg.setdefault("buttons_classes", num_button_combos)
        action_cfg.setdefault("camera_bins", num_camera_bins)
        filtered_cfg["action"] = ActionHeadConfig(**action_cfg)
    elif isinstance(action_cfg, ActionHeadConfig):
        pass
    else:
        filtered_cfg["action"] = ActionHeadConfig(
            buttons_classes=num_button_combos,
            camera_bins=num_camera_bins,
            use_camera_gate=True,
        )
    return PolicyConfig(**filtered_cfg)

channel_mean = IMAGENET_MEAN.view(1, 1, 3, 1, 1)
channel_std = IMAGENET_STD.view(1, 1, 3, 1, 1)

if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()
with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
    cfg = compose(
        config_name="config",
        overrides=[
            "dataset=mineworld_frames",
            "algorithm=policy_bc",
            "experiment=mineworld_bc_train",
        ],
    )
policy_cfg_dict = OmegaConf.to_container(cfg.algorithm.policy, resolve=True)
policy_cfg = _coerce_policy_config(policy_cfg_dict, dataset)

action_mapper = CameraHierarchicalMapping(n_camera_bins=policy_cfg.action.camera_bins)
action_transformer = ActionTransformer(
    camera_maxval=10,
    camera_binsize=2,
    camera_mu=10,
    camera_quantization_scheme="mu_law",
)

label_counter = Counter()

for i in tqdm(range(len(dataset)), desc="Processing dataset"):
    label = dataset[i][1]          # label is a tensor of shape [2]
    label = tuple(label.tolist())  # make it hashable (e.g. (1, 63))
    label_counter[label] += 1

print("Top 40 label combos:")
for (a, b), count in label_counter.most_common(40):
    print(f"  ({a}, {b}): {count}")
    
#print("Top predicted combos:", sorted(data_counter.items(), key=lambda x: -x[1])[:10], "Total: ", sum(data_counter.values()))
