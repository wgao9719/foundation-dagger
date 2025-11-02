from datasets.mineworld_data.mcdataset import MCDataset
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Tuple
import json
from typing import List, Dict


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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
        max_open_captures: int = 2,
    ) -> None:
        # Disable OpenCV's own threading to avoid exhausting ffmpeg worker pools
        try:
            cv2.setNumThreads(0)
        except AttributeError:
            pass
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
        zero_bins = self.mc_dataset.camera_quantizer.discretize(np.array([0.0, 0.0]))
        cam0_idx = int(zero_bins[0])
        cam1_idx = int(zero_bins[1])
        self.camera_center_tokens = (
            self.action_vocab[f"cam_0_{cam0_idx}"],
            self.action_vocab[f"cam_1_{cam1_idx}"],
        )
        self.camera_token_indices = (1, 2)
        self.normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        self._capture_cache: dict[Path, cv2.VideoCapture] = {}
        self._last_frame_index: dict[Path, int] = {}
        self._video_ids: dict[Path, int] = {}
        self._build_index(recursive=recursive)
        self._max_open_captures = max(1, int(max_open_captures))
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
