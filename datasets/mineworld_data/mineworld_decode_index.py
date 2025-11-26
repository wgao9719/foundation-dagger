from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List
import signal

import cv2
import numpy as np

from datasets.mineworld_data.mineworld_frame_dataset import _load_actions
from evaluation.agent import ACTION_TRANSFORMER_KWARGS
from evaluation.run_inverse_dynamics_model import json_action_to_env_action
from algorithms.foundation_dagger.vpt_model.action_mapping import CameraHierarchicalMapping
from algorithms.foundation_dagger.vpt_model.actions import ActionTransformer, Buttons


class MineWorldDecodeIndex:
    """
    Lightweight metadata builder for MineWorld video decoding.
    """

    def __init__(
        self,
        data_root: Path,
        recursive: bool = True,
        probe_timeout_seconds: float = 10.0,
    ) -> None:
        self.data_root = Path(data_root)
        self.records: List[Dict[str, object]] = []
        self._action_mapper = CameraHierarchicalMapping(n_camera_bins=11)
        self._action_transformer = ActionTransformer(**ACTION_TRANSFORMER_KWARGS)
        self._probe_timeout = max(0.0, float(probe_timeout_seconds))
        self._build(recursive=recursive)

    def _env_action_to_agent_np(self, env_action: Dict[str, object]) -> Dict[str, np.ndarray]:
        camera = np.asarray(env_action["camera"], dtype=np.float32)
        env_format: Dict[str, np.ndarray] = {"camera": camera[None]}
        for button_name in Buttons.ALL:
            env_format[button_name] = np.asarray([env_action.get(button_name, 0)], dtype=np.int64)

        minerl_action = self._action_transformer.env2policy(env_format)
        if minerl_action["camera"].ndim == 1:
            minerl_action = {k: v[None] for k, v in minerl_action.items()}
        agent_action_np = self._action_mapper.from_factored(minerl_action)
        agent_action: Dict[str, np.ndarray] = {}
        for key, value in agent_action_np.items():
            agent_action[key] = value[0].copy()
        return agent_action

    def _build(self, recursive: bool) -> None:
        pattern = "**/*.mp4" if recursive else "*.mp4"
        video_paths = sorted(self.data_root.glob(pattern))
        if not video_paths:
            raise FileNotFoundError(
                f"No .mp4 files found under {self.data_root}. "
                "Pass the MineWorld data directory via --data-root."
            )

        video_id = 0
        for video_path in video_paths:
            action_path = video_path.with_suffix(".jsonl")
            try:
                actions = _load_actions(action_path)
            except FileNotFoundError:
                continue

            cap = self._open_video_with_timeout(video_path)
            if not cap.isOpened():
                continue
            grabbed = self._grab_with_timeout(cap)
            if not grabbed:
                cap.release()
                continue
            frame_count = self._frame_count_with_timeout(cap)
            cap.release()

            usable = min(frame_count, len(actions))
            if usable <= 0:
                continue

            simplified_steps: List[Dict[str, float]] = []
            buttons: List[int] = []
            camera: List[int] = []
            esc_flags: List[int] = []

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

                simplified_steps.append(
                    dict(
                        frame_idx=step_idx,
                        is_gui_open=bool(step_data.get("isGuiOpen", False)),
                        cursor_x=float(mouse_data.get("x", 0.0)),
                        cursor_y=float(mouse_data.get("y", 0.0)),
                    )
                )
                buttons_arr = np.asarray(agent_action_np["buttons"]).reshape(-1)
                camera_arr = np.asarray(agent_action_np["camera"]).reshape(-1)
                buttons.append(int(buttons_arr[0]))
                camera.append(int(camera_arr[0]))
                esc_flags.append(int(normalized_action.get("ESC", 0)))

            if not simplified_steps:
                continue

            relative_video = video_path.relative_to(self.data_root)
            self.records.append(
                {
                    "video_id": video_id,
                    "video_path": video_path,
                    "relative_video": relative_video,
                    "step_infos": simplified_steps,
                    "buttons": buttons,
                    "camera": camera,
                    "esc": esc_flags,
                }
            )
            video_id += 1

    def iter_records(self) -> List[Dict[str, object]]:
        return self.records

    @contextmanager
    def _timeout_guard(self):
        if self._probe_timeout <= 0 or not hasattr(signal, "SIGALRM"):
            yield
            return

        def handler(signum, frame):
            raise TimeoutError()

        previous = signal.signal(signal.SIGALRM, handler)
        signal.setitimer(signal.ITIMER_REAL, self._probe_timeout)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)

    def _open_video_with_timeout(self, video_path: Path) -> cv2.VideoCapture:
        cap: cv2.VideoCapture | None = None
        try:
            with self._timeout_guard():
                cap = cv2.VideoCapture(str(video_path))
        except TimeoutError:
            if cap is not None:
                cap.release()
            print(f"[decode-index] Timed out opening {video_path}; skipping.")
            return cv2.VideoCapture()
        if not cap or not cap.isOpened():
            if cap is not None:
                cap.release()
            print(f"[decode-index] Could not open {video_path}; skipping.")
            return cv2.VideoCapture()
        return cap

    def _grab_with_timeout(self, cap: cv2.VideoCapture) -> bool:
        try:
            with self._timeout_guard():
                return cap.grab()
        except TimeoutError:
            print("[decode-index] Timed out grabbing initial frame; skipping video.")
            return False

    def _frame_count_with_timeout(self, cap: cv2.VideoCapture) -> int:
        try:
            with self._timeout_guard():
                return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        except TimeoutError:
            print("[decode-index] Timed out reading frame count; skipping video.")
            return 0
