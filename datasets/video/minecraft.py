from typing import Any, Dict, Optional
import os
import tarfile
import torch
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm
from torchvision.transforms import InterpolationMode
from torchvision import transforms
from .base_video import (
    BaseVideoDataset,
    BaseSimpleVideoDataset,
    BaseAdvancedVideoDataset,
)


class _PartStream:
    """
    Stream multiple tar part files sequentially without loading all of them into
    memory at once. Each part is deleted as soon as it is consumed.
    """

    def __init__(self, save_dir, suffixes):
        self.save_dir = save_dir
        self.suffixes = suffixes
        self._index = 0
        self._current_file = None
        self._current_path = None

    def _next_path(self):
        if self._index >= len(self.suffixes):
            return None
        suffix = self.suffixes[self._index]
        identifier = f"minecraft_marsh_dataset_{suffix}"
        return self.save_dir / identifier / f"minecraft.tar.part{suffix}"

    def read(self, size: int = -1) -> bytes:
        while True:
            if self._current_file is None:
                path = self._next_path()
                if path is None:
                    return b""
                self._current_path = path
                self._current_file = open(path, "rb")
            chunk = self._current_file.read(size)
            if chunk:
                return chunk
            self._consume_current()

    def close(self):
        self._consume_current(final=True)

    def _consume_current(self, final: bool = False):
        if self._current_file is None:
            return
        self._current_file.close()
        if self._current_path is not None:
            try:
                self._current_path.unlink()
            except FileNotFoundError:
                pass
            try:
                self._current_path.parent.rmdir()
            except OSError:
                pass
        self._current_file = None
        self._current_path = None
        self._index += 1
        if final:
            while self._index < len(self.suffixes):
                path = self._next_path()
                if path is None:
                    break
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
                try:
                    path.parent.rmdir()
                except OSError:
                    pass
                self._index += 1


class MinecraftBaseVideoDataset(BaseVideoDataset):
    _ALL_SPLITS = ["training", "validation"]

    def download_dataset(self):
        from internetarchive import download

        part_suffixes = [
            "aa",
            "ab",
            "ac",
            "ad",
            "ae",
            "af",
            "ag",
            "ah",
            "ai",
            "aj",
            "ak",
        ]
        combined_tar_path = self.save_dir / "minecraft_combined.tar"
        if combined_tar_path.exists():
            with open(combined_tar_path, "rb") as tar_file:
                try:
                    os.unlink(combined_tar_path)
                except FileNotFoundError:
                    pass
                with tarfile.open(fileobj=tar_file, mode="r|*") as combined_archive:
                    combined_archive.extractall(self.save_dir)
        else:
            for part_suffix in part_suffixes:
                identifier = f"minecraft_marsh_dataset_{part_suffix}"
                file_name = f"minecraft.tar.part{part_suffix}"
                download(identifier, file_name, destdir=self.save_dir, verbose=True)
            part_stream = _PartStream(self.save_dir, part_suffixes)
            try:
                with tarfile.open(fileobj=part_stream, mode="r|*") as combined_archive:
                    combined_archive.extractall(self.save_dir)
            finally:
                part_stream.close()
        (self.save_dir / "minecraft/test").rename(self.save_dir / "validation")
        (self.save_dir / "minecraft/train").rename(self.save_dir / "training")
        (self.save_dir / "minecraft").rmdir()

    def video_length(self, video_metadata: Dict[str, Any]) -> int:
        return 300

    def build_transform(self):
        return transforms.Resize(
            (self.resolution, self.resolution),
            interpolation=InterpolationMode.NEAREST_EXACT,
            antialias=True,
        )


class MinecraftSimpleVideoDataset(MinecraftBaseVideoDataset, BaseSimpleVideoDataset):
    """
    Minecraft simple video dataset
    """

    def __init__(self, cfg: DictConfig, split: str = "training"):
        if split == "test":
            split = "validation"
        BaseSimpleVideoDataset.__init__(self, cfg, split)


class MinecraftAdvancedVideoDataset(
    MinecraftBaseVideoDataset, BaseAdvancedVideoDataset
):
    """
    Minecraft advanced video dataset
    """

    def __init__(
        self,
        cfg: DictConfig,
        split: str = "training",
        current_epoch: Optional[int] = None,
    ):
        if split == "test":
            split = "validation"
        BaseAdvancedVideoDataset.__init__(self, cfg, split, current_epoch)

    def load_cond(
        self, video_metadata: Dict[str, Any], start_frame: int, end_frame: int
    ) -> torch.Tensor:
        path = video_metadata["video_paths"].with_suffix(".npz")
        actions = np.load(path)["actions"][start_frame:end_frame]
        return torch.from_numpy(np.eye(4)[actions]).float()
