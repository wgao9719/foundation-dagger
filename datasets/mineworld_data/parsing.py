import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import cv2

from datasets.mineworld_data.mcdataset import MCDataset
from evaluation.run_inverse_dynamics_model import json_action_to_env_action
from algorithms.foundation_dagger.vpt_model.action_mapping import CameraHierarchicalMapping
from algorithms.foundation_dagger.vpt_model.actions import ActionTransformer, Buttons
from evaluation.agent import ACTION_TRANSFORMER_KWARGS

def load_actions(json_path: Path) -> List[Dict]:
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

def env_action_to_agent_np(
    env_action: Dict[str, object],
    action_transformer: ActionTransformer,
    action_mapper: CameraHierarchicalMapping
) -> Dict[str, np.ndarray]:
    camera = np.asarray(env_action["camera"], dtype=np.float32)
    env_format: Dict[str, np.ndarray] = {
        "camera": camera[None],
    }
    for button_name in Buttons.ALL:
        env_format[button_name] = np.asarray([env_action.get(button_name, 0)], dtype=np.int64)

    minerl_action = action_transformer.env2policy(env_format)
    if minerl_action["camera"].ndim == 1:
        minerl_action = {k: v[None] for k, v in minerl_action.items()}
    agent_action_np = action_mapper.from_factored(minerl_action)
    agent_action: Dict[str, np.ndarray] = {}
    for key, value in agent_action_np.items():
        agent_action[key] = value[0].copy()
    return agent_action

def process_video_actions(
    video_path: Path,
    action_path: Path,
    context_frames: int,
    check_video_frame_count: bool = True,
    max_frames: Optional[int] = None
) -> Tuple[List[Dict[str, object]], List[Dict[str, np.ndarray]], List[int], int]:
    """
    Parses action logs and aligns them with video frames.
    Returns:
        - step_infos: List of dicts with metadata for each frame
        - agent_actions: List of agent actions corresponding to each frame
        - esc_flags: List of ESC flags
        - frame_count: Number of usable frames
    """
    actions = load_actions(action_path)
    
    frame_count = len(actions)
    if max_frames is not None:
        frame_count = min(frame_count, max_frames)
    elif check_video_frame_count:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return [], [], [], 0
        video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        frame_count = min(video_frame_count, frame_count)

    if frame_count == 0:
        return [], [], [], 0

    mc_dataset = MCDataset()
    # action_vocab stuff is not strictly needed here unless we need to quantize, 
    # but MCDataset initializes CameraQuantizer which we might need implicitly? 
    # Actually env_action_to_agent_np uses ActionTransformer which handles quantization.
    
    action_mapper = CameraHierarchicalMapping(n_camera_bins=11)
    action_transformer = ActionTransformer(**ACTION_TRANSFORMER_KWARGS)

    step_infos = []
    agent_actions_list = []
    esc_flags_list = []

    attack_is_stuck = False
    last_hotbar = 0

    for step_idx in range(frame_count):
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
            # Skip this frame? Original code 'continue's, effectively dropping this frame from the dataset.
            # However, for video decoding, if we skip a frame in the action list, we must ensure
            # we also skip the corresponding video frame or that the frame index logic handles it.
            # In original code:
            # video_steps.append(dict(frame_idx=step_idx, ...))
            # So 'step_idx' tracks the VIDEO frame index.
            # If is_null_action, we just don't add it to the usable dataset.
            continue

        normalized_action: Dict[str, np.ndarray | int] = {
            key: value.copy() if isinstance(value, np.ndarray) else int(value)
            for key, value in action_dict.items()
        }
        agent_action_np = env_action_to_agent_np(normalized_action, action_transformer, action_mapper)
        
        agent_actions_list.append(agent_action_np)
        esc_flags_list.append(int(normalized_action.get("ESC", 0)))

        step_infos.append(
            dict(
                frame_idx=step_idx,
                action=normalized_action,
                is_gui_open=bool(step_data.get("isGuiOpen", False)),
                cursor_x=float(mouse_data.get("x", 0.0)),
                cursor_y=float(mouse_data.get("y", 0.0)),
                agent_action=agent_action_np,
            )
        )

    return step_infos, agent_actions_list, esc_flags_list, frame_count

