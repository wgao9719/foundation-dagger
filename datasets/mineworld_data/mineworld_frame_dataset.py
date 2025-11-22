from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.mineworld_data.mcdataset import MCDataset
from evaluation.agent import ACTION_TRANSFORMER_KWARGS, AGENT_RESOLUTION, resize_image
from evaluation.data_loader import (
    CURSOR_FILE,
    MINEREC_ORIGINAL_HEIGHT_PX,
    composite_images_with_alpha,
)
from evaluation.run_inverse_dynamics_model import json_action_to_env_action
from algorithms.foundation_dagger.vpt_model.action_mapping import CameraHierarchicalMapping
from algorithms.foundation_dagger.vpt_model.actions import ActionTransformer, Buttons


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
        recursive: bool = True,
        max_open_captures: int = 12,
    ) -> None:
        # Disable OpenCV's own threading to avoid exhausting ffmpeg worker pools
        try:
            cv2.setNumThreads(0)
        except AttributeError:
            pass
        self.data_root = Path(data_root)
        self.context_frames = context_frames
        self.samples: list[tuple[Path, int, int]] = []
        self.actions_cache: dict[Path, list[dict]] = {}
        self.video_step_infos: dict[int, list[Dict[str, object]]] = {}
        self.mc_dataset = MCDataset()
        self.mc_dataset.make_action_vocab(action_vocab_offset=0)
        self.action_vocab = self.mc_dataset.action_vocab
        self.action_vocab_size = len(self.action_vocab)
        self.action_length = self.mc_dataset.action_length
        self.action_mapper = CameraHierarchicalMapping(n_camera_bins=11)
        self.action_transformer = ActionTransformer(**ACTION_TRANSFORMER_KWARGS)
        self.agent_actions: dict[tuple[int, int], Dict[str, np.ndarray]] = {}
        self.esc_actions: dict[tuple[int, int], int] = {}
        self._capture_cache: dict[Path, cv2.VideoCapture] = {}
        self._last_frame_index: dict[Path, int] = {}
        self._video_ids: dict[Path, int] = {}
        self._window_cache: dict[int, tuple[int, torch.Tensor]] = {}
        self._cursor_image_rgb, self._cursor_alpha = self._load_cursor()
        self._build_index(recursive=recursive)
        self._max_open_captures = max(1, int(max_open_captures))
        self._capture_order: list[Path] = []

    def _load_cursor(self) -> tuple[np.ndarray, np.ndarray]:
        cursor_image = cv2.imread(CURSOR_FILE, cv2.IMREAD_UNCHANGED)
        if cursor_image is None:
            raise FileNotFoundError(f"Cursor image missing at {CURSOR_FILE}")
        cursor_image = cursor_image[:16, :16, :]
        cursor_alpha = cursor_image[:, :, 3:] / 255.0
        cursor_rgb = cursor_image[:, :, :3]
        return cursor_rgb, cursor_alpha
        
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
            video_steps = self.video_step_infos.setdefault(video_id, [])

            attack_is_stuck = False
            last_hotbar = 0

            for step_idx in range(usable):
                base_step = actions[step_idx]

                if step_idx == 0:
                    if base_step.get("mouse", {}).get("newButtons", []) == [0]:
                        attack_is_stuck = True
                elif attack_is_stuck:
                    if 0 in base_step.get("mouse", {}).get("newButtons", []):
                        attack_is_stuck = False

                step_data = dict(base_step)
                mouse_data = dict(step_data.get("mouse", {}))
                if attack_is_stuck:
                    mouse_data["buttons"] = [b for b in mouse_data.get("buttons", []) if b != 0]
                step_data["mouse"] = mouse_data

                action_dict, is_null_action = json_action_to_env_action(step_data)

                current_hotbar = step_data.get("hotbar", last_hotbar)
                if current_hotbar != last_hotbar:
                    action_dict[f"hotbar.{current_hotbar + 1}"] = 1
                last_hotbar = current_hotbar

                if is_null_action:
                    continue

                normalized_action: Dict[str, np.ndarray | int] = {
                    key: value.copy() if isinstance(value, np.ndarray) else int(value)
                    for key, value in action_dict.items()
                }
                agent_action_np = self._env_action_to_agent_np(normalized_action)
                self.agent_actions[(video_id, step_idx)] = agent_action_np
                self.esc_actions[(video_id, step_idx)] = int(normalized_action.get("ESC", 0))

                video_steps.append(
                    dict(
                        frame_idx=step_idx,
                        action=normalized_action,
                        is_gui_open=bool(step_data.get("isGuiOpen", False)),
                        cursor_x=float(mouse_data.get("x", 0.0)),
                        cursor_y=float(mouse_data.get("y", 0.0)),
                        agent_action=agent_action_np,
                    )
                )

                if len(video_steps) >= self.context_frames:
                    end_idx = len(video_steps) - 1
                    self.samples.append((video_path, video_id, end_idx))

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

    def _frame_to_tensor(
        self,
        frame: np.ndarray,
        is_gui_open: bool,
        cursor_x: float,
        cursor_y: float,
    ) -> torch.Tensor:
        if is_gui_open:
            camera_scaling_factor = frame.shape[0] / MINEREC_ORIGINAL_HEIGHT_PX
            x_pos = int(cursor_x * camera_scaling_factor)
            y_pos = int(cursor_y * camera_scaling_factor)
            composite_images_with_alpha(frame, self._cursor_image_rgb, self._cursor_alpha, x_pos, y_pos)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.asarray(np.clip(frame, 0, 255), dtype=np.uint8)
        frame = resize_image(frame, AGENT_RESOLUTION)
        tensor = torch.from_numpy(frame)
        return tensor

    def _env_action_to_agent_np(self, env_action: Dict[str, object]) -> Dict[str, np.ndarray]:
        camera = np.asarray(env_action["camera"], dtype=np.float32)
        env_format: Dict[str, np.ndarray] = {
            "camera": camera[None],
        }
        for button_name in Buttons.ALL:
            env_format[button_name] = np.asarray([env_action.get(button_name, 0)], dtype=np.int64)

        minerl_action = self.action_transformer.env2policy(env_format)
        if minerl_action["camera"].ndim == 1:
            minerl_action = {k: v[None] for k, v in minerl_action.items()}
        agent_action_np = self.action_mapper.from_factored(minerl_action)
        agent_action: Dict[str, np.ndarray] = {}
        for key, value in agent_action_np.items():
            agent_action[key] = value[0].copy()
        return agent_action

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

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        video_path, video_id, end_idx = self.samples[index]
        step_infos = self.video_step_infos[video_id]
        start_idx = end_idx - self.context_frames + 1
        cached = self._window_cache.get(int(video_id))
        cache_hit = cached is not None and cached[0] == end_idx - 1
        frames_tensor: torch.Tensor | None = None
        if cache_hit:
            prev_frames = cached[1]
            frames_tensor = torch.empty_like(prev_frames)
            frames_tensor[:-1] = prev_frames[1:]
            new_step = step_infos[end_idx]
            frame = self._read_frame(video_path, new_step["frame_idx"])
            frames_tensor[-1] = self._frame_to_tensor(
                frame,
                is_gui_open=new_step["is_gui_open"],
                cursor_x=new_step["cursor_x"],
                cursor_y=new_step["cursor_y"],
            )
        else:
            frames_tensor = None
            steps = step_infos[start_idx : end_idx + 1]
            for offset, step in enumerate(steps):
                frame = self._read_frame(video_path, step["frame_idx"])
                tensor = self._frame_to_tensor(
                    frame,
                    is_gui_open=step["is_gui_open"],
                    cursor_x=step["cursor_x"],
                    cursor_y=step["cursor_y"],
                )
                if frames_tensor is None:
                    frames_tensor = torch.empty(
                        (self.context_frames, *tensor.shape),
                        dtype=tensor.dtype,
                        device=tensor.device,
                    )
                frames_tensor[offset] = tensor
        assert frames_tensor is not None
        self._window_cache[int(video_id)] = (end_idx, frames_tensor.clone())

        final_step = step_infos[end_idx]
        # Construct a compact, tensor-friendly label from the already-discretized agent action
        agent_action = self.agent_actions[(int(video_id), int(final_step["frame_idx"]))]
        buttons_idx = int(agent_action["buttons"][0])
        camera_idx = int(agent_action["camera"][0])
        label = torch.tensor([buttons_idx, camera_idx], dtype=torch.long)
        return (
            frames_tensor,
            label,
            torch.tensor(video_id, dtype=torch.long),
            torch.tensor(final_step["frame_idx"], dtype=torch.long),
        )

    def get_agent_action(self, video_id: int, frame_idx: int) -> Dict[str, torch.Tensor]:
        agent_np = self.agent_actions.get((int(video_id), int(frame_idx)))
        if agent_np is None:
            raise KeyError(f"Missing agent action for video_id={video_id}, frame_idx={frame_idx}")
        return {key: torch.from_numpy(value.copy()) for key, value in agent_np.items()}

    def get_esc_flag(self, video_id: int, frame_idx: int) -> int:
        return int(self.esc_actions.get((int(video_id), int(frame_idx)), 0))

    def get_frame_index(self, video_id: int, step_position: int) -> int:
        video_steps = self.video_step_infos.get(int(video_id))
        if video_steps is None:
            raise KeyError(f"Unknown video_id={video_id}")
        if step_position < 0 or step_position >= len(video_steps):
            raise IndexError(
                f"step_position={step_position} out of range for video_id={video_id} "
                f"with {len(video_steps)} steps"
            )
        return int(video_steps[int(step_position)]["frame_idx"])


class MineWorldDecodedFrameDataset(Dataset):
    """
    Dataset that returns MineWorld context windows from pre-decoded frame tensors.

    The decoded data should be produced via scripts/decode_mineworld_frames.py, which
    stores one .pt tensor bundle per trajectory along with a manifest describing the
    available videos. Each bundle contains resized RGB frames along with the discrete
    action labels and ESC flags aligned to the final frame in each context window.
    """

    MANIFEST_VERSION = 1

    def __init__(
        self,
        decoded_root: Path,
        context_frames: int = 8,
        manifest_name: str = "manifest.json",
        max_cached_videos: int = 4,
    ) -> None:
        self.data_root = Path(decoded_root)
        if not self.data_root.exists():
            raise FileNotFoundError(f"Decoded data root {self.data_root} is missing.")
        self.context_frames = int(context_frames)
        if self.context_frames <= 0:
            raise ValueError("context_frames must be positive.")
        self.manifest_path = self.data_root / manifest_name
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Decoded manifest missing at {self.manifest_path}. "
                "Generate it via scripts/decode_mineworld_frames.py."
            )
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        version = int(manifest.get("version", 0))
        if version != self.MANIFEST_VERSION:
            raise ValueError(
                f"Unsupported manifest version {version}. "
                f"Expected {self.MANIFEST_VERSION}."
            )
        videos = manifest.get("videos", [])
        if not videos:
            raise ValueError(f"Manifest at {self.manifest_path} lists no videos.")

        self.samples: list[tuple[Path, int, int]] = []
        self._video_records: dict[int, dict[str, object]] = {}
        self._decoded_cache: dict[int, dict[str, torch.Tensor]] = {}
        self._cache_order: list[int] = []
        self._max_cached_videos = max(1, int(max_cached_videos))
        self.video_step_infos: dict[int, list[Dict[str, object]]] = {}
        self.agent_actions: dict[tuple[int, int], Dict[str, np.ndarray]] = {}
        self.esc_actions: dict[tuple[int, int], int] = {}

        for entry in videos:
            video_id = int(entry["video_id"])
            rel_path = entry.get("decoded_file")
            if not rel_path:
                continue
            decoded_path = self.data_root / rel_path
            if not decoded_path.exists():
                raise FileNotFoundError(
                    f"Decoded tensor file {decoded_path} listed in manifest is missing."
                )
            num_steps = int(entry.get("num_steps", 0))
            if num_steps <= 0:
                continue
            self._video_records[video_id] = {
                "path": decoded_path,
                "original_video": entry.get("original_video"),
            }
            # Track frame indices for compatibility with sequential logic in training scripts.
            step_infos = [{"frame_idx": int(idx)} for idx in range(num_steps)]
            self.video_step_infos[video_id] = step_infos
            for end_idx in range(self.context_frames - 1, num_steps):
                self.samples.append((decoded_path, video_id, end_idx))

        if not self.samples:
            raise RuntimeError(
                f"No usable context windows found under {self.data_root}. "
                "Ensure the decoded dataset was generated with sufficient frames."
            )

        # Mirror the original dataset interfaces so downstream code can reuse them.
        self.mc_dataset = MCDataset()
        self.mc_dataset.make_action_vocab(action_vocab_offset=0)
        self.action_vocab = self.mc_dataset.action_vocab
        self.action_vocab_size = len(self.action_vocab)
        self.action_length = self.mc_dataset.action_length
        self.action_mapper = CameraHierarchicalMapping(n_camera_bins=11)
        self.action_transformer = ActionTransformer(**ACTION_TRANSFORMER_KWARGS)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _, video_id, end_idx = self.samples[index]
        video_id = int(video_id)
        end_idx = int(end_idx)
        data = self._load_video_bundle(video_id)
        start_idx = end_idx - self.context_frames + 1
        frames = data["frames"][start_idx : end_idx + 1]
        buttons_idx = int(data["buttons"][end_idx].item())
        camera_idx = int(data["camera"][end_idx].item())
        label = torch.tensor([buttons_idx, camera_idx], dtype=torch.long)
        frame_idx = int(data["frame_indices"][end_idx].item())
        return (
            frames.contiguous(),
            label,
            torch.tensor(video_id, dtype=torch.long),
            torch.tensor(frame_idx, dtype=torch.long),
        )

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_decoded_cache"] = {}
        state["_cache_order"] = []
        return state

    def _load_video_bundle(self, video_id: int) -> dict:
        video_id = int(video_id)
        cached = self._decoded_cache.get(video_id)
        if cached is not None:
            return cached
        record = self._video_records.get(video_id)
        if record is None:
            raise KeyError(f"Unknown decoded video_id={video_id}")
        path = record["path"]
        data = torch.load(path, map_location="cpu")
        required_keys = {"frames", "frame_indices", "buttons", "camera", "esc"}
        missing = required_keys.difference(data.keys())
        if missing:
            raise KeyError(
                f"Decoded file {path} missing required keys: {', '.join(sorted(missing))}"
            )
        # Ensure tensors use consistent dtypes and layouts.
        data["frames"] = data["frames"].contiguous()
        data["frame_indices"] = data["frame_indices"].to(torch.long).contiguous()
        data["buttons"] = data["buttons"].to(torch.long).contiguous()
        data["camera"] = data["camera"].to(torch.long).contiguous()
        data["esc"] = data["esc"].to(torch.long).contiguous()
        data["_frame_to_local"] = {
            int(idx): pos for pos, idx in enumerate(data["frame_indices"].tolist())
        }
        self._decoded_cache[video_id] = data
        if video_id in self._cache_order:
            self._cache_order.remove(video_id)
        self._cache_order.append(video_id)
        if len(self._cache_order) > self._max_cached_videos:
            oldest = self._cache_order.pop(0)
            self._decoded_cache.pop(oldest, None)
        return data

    def get_agent_action(self, video_id: int, frame_idx: int) -> Dict[str, torch.Tensor]:
        data = self._load_video_bundle(video_id)
        frame_to_local = data["_frame_to_local"]
        local_idx = frame_to_local.get(int(frame_idx))
        if local_idx is None:
            raise KeyError(f"Missing agent action for video_id={video_id}, frame_idx={frame_idx}")
        buttons_value = data["buttons"][local_idx].view(1)
        camera_value = data["camera"][local_idx].view(1)
        return {
            "buttons": buttons_value.clone(),
            "camera": camera_value.clone(),
        }

    def get_esc_flag(self, video_id: int, frame_idx: int) -> int:
        data = self._load_video_bundle(video_id)
        frame_to_local = data["_frame_to_local"]
        local_idx = frame_to_local.get(int(frame_idx))
        if local_idx is None:
            return 0
        return int(data["esc"][local_idx].item())

    def get_frame_index(self, video_id: int, step_position: int) -> int:
        data = self._load_video_bundle(video_id)
        if step_position < 0 or step_position >= data["frame_indices"].numel():
            raise IndexError(
                f"step_position={step_position} out of range for video_id={video_id} "
                f"with {data['frame_indices'].numel()} steps"
            )
        return int(data["frame_indices"][int(step_position)].item())
