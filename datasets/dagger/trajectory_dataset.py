from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from omegaconf import DictConfig


@dataclass
class TrajectoryStoreConfig:
    save_dir: str = "data/dagger"
    split: str = "training"
    in_memory: bool = True
    metadata_name: str = "manifest.json"


class DaggerTrajectoryDataset(torch.utils.data.Dataset):
    """
    Lightweight dataset for BC/DAgger tuples.
    Files are stored as npz with keys: observations (N,C,H,W) and actions (N,).
    """

    def __init__(self, cfg: DictConfig, split: str = "training") -> None:
        super().__init__()
        store_cfg = TrajectoryStoreConfig(**cfg.store)
        self.cfg = store_cfg
        self.split = split or store_cfg.split
        self.root = Path(store_cfg.save_dir) / self.split
        self.root.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.root / store_cfg.metadata_name
        self.files = self._discover_files()
        self.in_memory = store_cfg.in_memory
        self.cache: Optional[Dict[int, torch.Tensor]] = None
        if self.in_memory:
            self._load_cache()

    def _discover_files(self) -> List[Path]:
        files = sorted(self.root.glob("*.npz"))
        if self.metadata_path.exists():
            manifest = json.loads(self.metadata_path.read_text())
            order = [self.root / item for item in manifest.get("files", [])]
            files = [path for path in order if path.exists()]
        return files

    def _load_cache(self) -> None:
        obs_list, act_list = [], []
        for fpath in self.files:
            sample = np.load(fpath)
            obs_list.append(torch.from_numpy(sample["observations"]))
            act_list.append(torch.from_numpy(sample["actions"]))
        if obs_list:
            self.cache = {
                "observations": torch.cat(obs_list, dim=0),
                "actions": torch.cat(act_list, dim=0).long(),
            }
        else:
            self.cache = {
                "observations": torch.zeros(0, 3, 1, 1),
                "actions": torch.zeros(0, dtype=torch.long),
            }

    def __len__(self) -> int:
        if self.cache is not None:
            return self.cache["actions"].shape[0]
        total = 0
        for fpath in self.files:
            sample = np.load(fpath)
            total += sample["actions"].shape[0]
        return total

    def _getitem_stream(self, index: int):
        running = 0
        for fpath in self.files:
            sample = np.load(fpath)
            actions = sample["actions"]
            if index < running + actions.shape[0]:
                local_idx = index - running
                obs = torch.from_numpy(sample["observations"][local_idx])
                act = torch.tensor(int(actions[local_idx]), dtype=torch.long)
                return {"observations": obs, "actions": act}
            running += actions.shape[0]
        raise IndexError(index)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if self.cache is not None:
            return {
                "observations": self.cache["observations"][index],
                "actions": self.cache["actions"][index],
            }
        return self._getitem_stream(index)
