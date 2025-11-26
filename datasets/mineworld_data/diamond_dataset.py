"""
Dataset and action mapping for MineRL task_diamond format.

This handles the MineRL competition data format with:
- 64x64 video frames in recording.mp4
- Actions and observations in rendered.npz
- Explicit inventory observations (no hotbar in frame)
- Factored action space with binary buttons + categorical craft/smelt/equip actions
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# =============================================================================
# Action Space Definition
# =============================================================================

@dataclass
class MineRLActionVocab:
    """
    Vocabulary for the MineRL factored action space.
    
    Actions are split into:
    - 8 binary buttons (forward, left, back, right, jump, sneak, sprint, attack)
    - 2D camera (discretized into bins)
    - 5 categorical actions (place, equip, craft, nearbyCraft, nearbySmelt)
    """
    
    # Binary button names (order matters for indexing)
    BINARY_BUTTONS: List[str] = field(default_factory=lambda: [
        "forward", "left", "back", "right", "jump", "sneak", "sprint", "attack"
    ])
    
    # Categorical action vocabularies (discovered from data exploration)
    # Each includes "none" as index 0
    PLACE_VOCAB: List[str] = field(default_factory=lambda: [
        "none", "cobblestone", "crafting_table", "dirt", "furnace", "stone", "torch"
    ])
    
    EQUIP_VOCAB: List[str] = field(default_factory=lambda: [
        "none", "iron_axe", "iron_pickaxe", "stone_axe", "stone_pickaxe", "wooden_pickaxe"
    ])
    
    CRAFT_VOCAB: List[str] = field(default_factory=lambda: [
        "none", "crafting_table", "planks", "stick", "torch"
    ])
    
    NEARBY_CRAFT_VOCAB: List[str] = field(default_factory=lambda: [
        "none", "furnace", "iron_axe", "iron_pickaxe", "stone_axe", "stone_pickaxe", "wooden_pickaxe"
    ])
    
    NEARBY_SMELT_VOCAB: List[str] = field(default_factory=lambda: [
        "none", "coal", "iron_ingot"
    ])
    
    # Inventory item names (order from the npz keys)
    INVENTORY_ITEMS: List[str] = field(default_factory=lambda: [
        "coal", "cobblestone", "crafting_table", "dirt", "furnace",
        "iron_axe", "iron_ingot", "iron_ore", "iron_pickaxe", "log",
        "planks", "stick", "stone", "stone_axe", "stone_pickaxe",
        "torch", "wooden_axe", "wooden_pickaxe"
    ])
    
    # Equipped item types
    EQUIPPED_TYPES: List[str] = field(default_factory=lambda: [
        "none", "other", "air"  # Add more if discovered
    ])
    
    def __post_init__(self):
        # Build reverse mappings
        self.place_to_idx = {v: i for i, v in enumerate(self.PLACE_VOCAB)}
        self.equip_to_idx = {v: i for i, v in enumerate(self.EQUIP_VOCAB)}
        self.craft_to_idx = {v: i for i, v in enumerate(self.CRAFT_VOCAB)}
        self.nearby_craft_to_idx = {v: i for i, v in enumerate(self.NEARBY_CRAFT_VOCAB)}
        self.nearby_smelt_to_idx = {v: i for i, v in enumerate(self.NEARBY_SMELT_VOCAB)}
        self.equipped_type_to_idx = {v: i for i, v in enumerate(self.EQUIPPED_TYPES)}
    
    @property
    def num_binary_buttons(self) -> int:
        return len(self.BINARY_BUTTONS)
    
    @property
    def num_place_classes(self) -> int:
        return len(self.PLACE_VOCAB)
    
    @property
    def num_equip_classes(self) -> int:
        return len(self.EQUIP_VOCAB)
    
    @property
    def num_craft_classes(self) -> int:
        return len(self.CRAFT_VOCAB)
    
    @property
    def num_nearby_craft_classes(self) -> int:
        return len(self.NEARBY_CRAFT_VOCAB)
    
    @property
    def num_nearby_smelt_classes(self) -> int:
        return len(self.NEARBY_SMELT_VOCAB)
    
    @property
    def num_inventory_items(self) -> int:
        return len(self.INVENTORY_ITEMS)
    
    @property
    def num_equipped_types(self) -> int:
        return len(self.EQUIPPED_TYPES)


class MineRLActionMapping:
    """
    Maps between raw npz action format and model-friendly tensors.
    
    Camera actions are discretized into bins (default 11 per axis).
    Binary buttons are kept as 0/1.
    Categorical actions are converted to integer indices.
    """
    
    def __init__(
        self,
        n_camera_bins: int = 11,
        camera_max_angle: float = 180.0,
    ):
        assert n_camera_bins % 2 == 1, "n_camera_bins should be odd for symmetric binning"
        self.n_camera_bins = n_camera_bins
        self.camera_max_angle = camera_max_angle
        self.camera_null_bin = n_camera_bins // 2
        self.vocab = MineRLActionVocab()
        
        # Precompute camera bin edges
        # Bins go from -camera_max_angle to +camera_max_angle
        self._bin_edges = np.linspace(
            -camera_max_angle, camera_max_angle, n_camera_bins + 1
        )
    
    def discretize_camera(self, camera: np.ndarray) -> np.ndarray:
        """
        Discretize continuous camera angles into bins.
        
        Args:
            camera: Array of shape (..., 2) with pitch/yaw angles
            
        Returns:
            Array of same shape with bin indices
        """
        # Clip to valid range
        clipped = np.clip(camera, -self.camera_max_angle, self.camera_max_angle)
        # Digitize returns 1-indexed bins, subtract 1 for 0-indexed
        # Also clip to valid range in case of edge cases
        bins = np.digitize(clipped, self._bin_edges[1:-1])
        return bins.astype(np.int64)
    
    def bins_to_angles(self, bins: np.ndarray) -> np.ndarray:
        """Convert bin indices back to angle centers."""
        bin_centers = (self._bin_edges[:-1] + self._bin_edges[1:]) / 2
        return bin_centers[bins]
    
    def encode_categorical(self, values: np.ndarray, vocab_map: Dict[str, int]) -> np.ndarray:
        """Convert string categorical values to indices."""
        result = np.zeros(len(values), dtype=np.int64)
        for i, v in enumerate(values):
            result[i] = vocab_map.get(str(v), 0)  # Default to 0 ("none") if unknown
        return result
    
    def process_npz_actions(
        self, npz_data: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Convert raw npz action arrays to model-ready format.
        
        Returns dict with:
        - binary_buttons: [T, 8] int64 array
        - camera: [T, 2] int64 array (binned)
        - camera_continuous: [T, 2] float32 array (original)
        - place: [T] int64 array
        - equip: [T] int64 array
        - craft: [T] int64 array
        - nearby_craft: [T] int64 array
        - nearby_smelt: [T] int64 array
        """
        # Get sequence length from any action
        T = len(npz_data["action$forward"])
        
        # Binary buttons
        binary_buttons = np.stack([
            npz_data[f"action${btn}"] for btn in self.vocab.BINARY_BUTTONS
        ], axis=-1).astype(np.int64)
        
        # Camera
        camera_continuous = npz_data["action$camera"].astype(np.float32)
        camera_binned = self.discretize_camera(camera_continuous)
        
        # Categorical actions
        place = self.encode_categorical(
            npz_data["action$place"], self.vocab.place_to_idx
        )
        equip = self.encode_categorical(
            npz_data["action$equip"], self.vocab.equip_to_idx
        )
        craft = self.encode_categorical(
            npz_data["action$craft"], self.vocab.craft_to_idx
        )
        nearby_craft = self.encode_categorical(
            npz_data["action$nearbyCraft"], self.vocab.nearby_craft_to_idx
        )
        nearby_smelt = self.encode_categorical(
            npz_data["action$nearbySmelt"], self.vocab.nearby_smelt_to_idx
        )
        
        return {
            "binary_buttons": binary_buttons,
            "camera": camera_binned,
            "camera_continuous": camera_continuous,
            "place": place,
            "equip": equip,
            "craft": craft,
            "nearby_craft": nearby_craft,
            "nearby_smelt": nearby_smelt,
        }
    
    def process_npz_observations(
        self, npz_data: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Convert raw npz observation arrays to model-ready format.
        
        Returns dict with:
        - inventory: [T, 18] int32 array (item counts)
        - equipped_type: [T] int64 array
        - equipped_damage: [T] int32 array
        - equipped_max_damage: [T] int32 array
        """
        # Note: observations have T+1 elements (initial state + T steps)
        # We'll return all of them; the dataset can align as needed
        
        # Stack inventory items
        inventory = np.stack([
            npz_data[f"observation$inventory${item}"]
            for item in self.vocab.INVENTORY_ITEMS
        ], axis=-1).astype(np.int32)
        
        # Equipped item
        equipped_type = self.encode_categorical(
            npz_data["observation$equipped_items.mainhand.type"],
            self.vocab.equipped_type_to_idx,
        )
        equipped_damage = npz_data["observation$equipped_items.mainhand.damage"].astype(np.int32)
        equipped_max_damage = npz_data["observation$equipped_items.mainhand.maxDamage"].astype(np.int32)
        
        return {
            "inventory": inventory,
            "equipped_type": equipped_type,
            "equipped_damage": equipped_damage,
            "equipped_max_damage": equipped_max_damage,
        }
    
    @property
    def camera_null_idx(self) -> int:
        """Index of the null camera action (no movement)."""
        return self.camera_null_bin * self.n_camera_bins + self.camera_null_bin
    
    def get_action_space_info(self) -> Dict[str, int]:
        """Return dict describing action space sizes."""
        return {
            "num_binary_buttons": self.vocab.num_binary_buttons,
            "camera_bins_per_axis": self.n_camera_bins,
            "camera_total_bins": self.n_camera_bins * self.n_camera_bins,
            "num_place_classes": self.vocab.num_place_classes,
            "num_equip_classes": self.vocab.num_equip_classes,
            "num_craft_classes": self.vocab.num_craft_classes,
            "num_nearby_craft_classes": self.vocab.num_nearby_craft_classes,
            "num_nearby_smelt_classes": self.vocab.num_nearby_smelt_classes,
        }


# =============================================================================
# Dataset Class
# =============================================================================

class MineRLDiamondDataset(Dataset):
    """
    Dataset for MineRL ObtainDiamond/ObtainIronPickaxe format.
    
    Each trajectory folder contains:
    - recording.mp4: 64x64 video frames
    - rendered.npz: actions and observations
    - metadata.json: trajectory metadata
    
    The dataset returns sliding windows of frames along with:
    - Inventory observations (explicit, not in frame)
    - Multi-head action labels
    """
    
    def __init__(
        self,
        data_root: Path,
        context_frames: int = 8,
        n_camera_bins: int = 11,
        target_resolution: Optional[Tuple[int, int]] = None,
        max_open_captures: int = 12,
        skip_null_actions: bool = True,
    ) -> None:
        """
        Args:
            data_root: Root directory containing trajectory folders
            context_frames: Number of frames in each context window
            n_camera_bins: Number of bins per camera axis
            target_resolution: Optional (H, W) to resize frames. None keeps 64x64.
            max_open_captures: Max number of video captures to keep open
            skip_null_actions: Whether to skip frames with all-null actions
        """
        try:
            cv2.setNumThreads(0)
        except AttributeError:
            pass
        
        self.data_root = Path(data_root)
        self.context_frames = context_frames
        self.target_resolution = target_resolution
        self.skip_null_actions = skip_null_actions
        
        self.action_mapper = MineRLActionMapping(n_camera_bins=n_camera_bins)
        
        # Caches
        self._capture_cache: Dict[Path, cv2.VideoCapture] = {}
        self._last_frame_index: Dict[Path, int] = {}
        self._capture_order: List[Path] = []
        self._max_open_captures = max(1, int(max_open_captures))
        
        # Data storage
        self._trajectory_data: Dict[int, Dict] = {}  # video_id -> loaded data
        self._video_paths: Dict[int, Path] = {}  # video_id -> video path
        self.samples: List[Tuple[int, int]] = []  # (video_id, end_frame_idx)
        
        self._build_index()
    
    def _build_index(self) -> None:
        """Scan data_root for trajectory folders and build sample index."""
        # Find all trajectory folders (those containing rendered.npz)
        trajectory_dirs = []
        for root, dirs, files in os.walk(self.data_root):
            if "rendered.npz" in files and "recording.mp4" in files:
                trajectory_dirs.append(Path(root))
        
        if not trajectory_dirs:
            raise FileNotFoundError(
                f"No trajectory folders found under {self.data_root}. "
                "Expected folders with rendered.npz and recording.mp4."
            )
        
        trajectory_dirs = sorted(trajectory_dirs)
        
        for traj_idx, traj_dir in enumerate(trajectory_dirs):
            video_id = traj_idx
            video_path = traj_dir / "recording.mp4"
            npz_path = traj_dir / "rendered.npz"
            
            # Load npz data
            try:
                npz_data = dict(np.load(npz_path, allow_pickle=True))
            except Exception as e:
                print(f"Warning: Failed to load {npz_path}: {e}")
                continue
            
            # Get video info
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                cap.release()
                continue
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            # Process actions and observations
            actions = self.action_mapper.process_npz_actions(npz_data)
            observations = self.action_mapper.process_npz_observations(npz_data)
            
            # Actions have T elements, observations have T+1 (initial + T)
            num_actions = len(actions["binary_buttons"])
            usable_frames = min(frame_count, num_actions)
            
            if usable_frames < self.context_frames:
                continue
            
            # Store trajectory data
            self._trajectory_data[video_id] = {
                "actions": actions,
                "observations": observations,
                "num_frames": usable_frames,
            }
            self._video_paths[video_id] = video_path
            
            # Build samples - each sample is (video_id, end_frame_idx)
            # where end_frame_idx is the frame we predict the action for
            for end_idx in range(self.context_frames - 1, usable_frames):
                # Optionally skip null actions
                if self.skip_null_actions:
                    if self._is_null_action(actions, end_idx):
                        continue
                self.samples.append((video_id, end_idx))
        
        if not self.samples:
            raise RuntimeError(
                f"No usable samples found under {self.data_root}. "
                "Check that trajectories have sufficient frames."
            )
        
        print(f"MineRLDiamondDataset: {len(self.samples)} samples from "
              f"{len(self._video_paths)} trajectories")
    
    def _is_null_action(self, actions: Dict[str, np.ndarray], idx: int) -> bool:
        """Check if action at idx is effectively null (no movement, no actions)."""
        # Check binary buttons - all zero?
        if actions["binary_buttons"][idx].sum() > 0:
            return False
        # Check camera - null bin?
        cam = actions["camera"][idx]
        null_bin = self.action_mapper.camera_null_bin
        if cam[0] != null_bin or cam[1] != null_bin:
            return False
        # Check categorical actions - all "none" (idx 0)?
        if actions["place"][idx] != 0:
            return False
        if actions["equip"][idx] != 0:
            return False
        if actions["craft"][idx] != 0:
            return False
        if actions["nearby_craft"][idx] != 0:
            return False
        if actions["nearby_smelt"][idx] != 0:
            return False
        return True
    
    def _get_capture(self, video_path: Path) -> cv2.VideoCapture:
        """Get or create a VideoCapture for the given path, with LRU eviction."""
        cap = self._capture_cache.get(video_path)
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open {video_path}")
            self._capture_cache[video_path] = cap
            self._last_frame_index[video_path] = -1
            self._capture_order.append(video_path)
            # Evict oldest if over limit
            if len(self._capture_order) > self._max_open_captures:
                oldest = self._capture_order.pop(0)
                old_cap = self._capture_cache.pop(oldest, None)
                if old_cap is not None and old_cap.isOpened():
                    old_cap.release()
                self._last_frame_index.pop(oldest, None)
        else:
            # Refresh LRU order
            if video_path in self._capture_order:
                self._capture_order.remove(video_path)
            self._capture_order.append(video_path)
        return cap
    
    def _read_frame(self, video_path: Path, frame_idx: int) -> np.ndarray:
        """Read a single frame from video."""
        cap = self._get_capture(video_path)
        last_idx = self._last_frame_index.get(video_path, -1)
        
        # Sequential read optimization
        if last_idx + 1 == frame_idx:
            success, frame = cap.read()
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = cap.read()
        
        if not success:
            raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
        
        self._last_frame_index[video_path] = frame_idx
        return frame
    
    def _process_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Convert BGR frame to RGB tensor, optionally resize."""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.target_resolution is not None:
            frame = cv2.resize(
                frame, 
                (self.target_resolution[1], self.target_resolution[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        return torch.from_numpy(frame.copy())
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Returns a dict with:
        - frames: [T, H, W, C] uint8 tensor
        - inventory: [T, 18] int32 tensor (inventory counts for context window)
        - equipped_type: [T] int64 tensor
        - action_buttons: [8] int64 tensor (binary buttons for final frame)
        - action_camera: [2] int64 tensor (camera bins for final frame)
        - action_place: int64 scalar
        - action_equip: int64 scalar
        - action_craft: int64 scalar
        - action_nearby_craft: int64 scalar
        - action_nearby_smelt: int64 scalar
        - video_id: int64 scalar
        - frame_idx: int64 scalar
        """
        video_id, end_idx = self.samples[index]
        start_idx = end_idx - self.context_frames + 1
        
        video_path = self._video_paths[video_id]
        traj_data = self._trajectory_data[video_id]
        actions = traj_data["actions"]
        observations = traj_data["observations"]
        
        # Read frames
        frames_list = []
        for fidx in range(start_idx, end_idx + 1):
            frame = self._read_frame(video_path, fidx)
            frames_list.append(self._process_frame(frame))
        frames = torch.stack(frames_list, dim=0)  # [T, H, W, C]
        
        # Get observations for context window
        # Observations are 1-indexed relative to actions (obs[i] is state before action[i-1])
        # So for action at end_idx, we use observations from start_idx to end_idx+1
        obs_start = start_idx
        obs_end = end_idx + 1
        inventory = torch.from_numpy(
            observations["inventory"][obs_start:obs_end].copy()
        )  # [T, 18]
        equipped_type = torch.from_numpy(
            observations["equipped_type"][obs_start:obs_end].copy()
        )  # [T]
        
        # Get action labels (for final frame only)
        action_buttons = torch.from_numpy(actions["binary_buttons"][end_idx].copy())
        action_camera = torch.from_numpy(actions["camera"][end_idx].copy())
        action_place = torch.tensor(actions["place"][end_idx], dtype=torch.int64)
        action_equip = torch.tensor(actions["equip"][end_idx], dtype=torch.int64)
        action_craft = torch.tensor(actions["craft"][end_idx], dtype=torch.int64)
        action_nearby_craft = torch.tensor(actions["nearby_craft"][end_idx], dtype=torch.int64)
        action_nearby_smelt = torch.tensor(actions["nearby_smelt"][end_idx], dtype=torch.int64)
        
        return {
            "frames": frames,
            "inventory": inventory,
            "equipped_type": equipped_type,
            "action_buttons": action_buttons,
            "action_camera": action_camera,
            "action_place": action_place,
            "action_equip": action_equip,
            "action_craft": action_craft,
            "action_nearby_craft": action_nearby_craft,
            "action_nearby_smelt": action_nearby_smelt,
            "video_id": torch.tensor(video_id, dtype=torch.int64),
            "frame_idx": torch.tensor(end_idx, dtype=torch.int64),
        }
    
    def __getstate__(self) -> dict:
        """For pickling (multiprocessing DataLoader)."""
        state = self.__dict__.copy()
        state["_capture_cache"] = {}
        state["_last_frame_index"] = {}
        state["_capture_order"] = []
        return state
    
    def __del__(self) -> None:
        """Release video captures."""
        for cap in self._capture_cache.values():
            if cap.isOpened():
                cap.release()
    
    def get_action_space_info(self) -> Dict[str, int]:
        """Return action space sizes for model construction."""
        return self.action_mapper.get_action_space_info()
    
    def get_inventory_dim(self) -> int:
        """Return dimension of inventory vector."""
        return self.action_mapper.vocab.num_inventory_items
    
    def get_num_equipped_types(self) -> int:
        """Return number of equipped item type classes."""
        return self.action_mapper.vocab.num_equipped_types


# =============================================================================
# Decoded Dataset (Fast Loading from Pre-decoded Tensors)
# =============================================================================

class MineRLDiamondDecodedDataset(Dataset):
    """
    Fast dataset for pre-decoded MineRL Diamond trajectories.
    
    Loads frames from .pt tensor bundles created by decode_diamond_frames.py
    instead of decoding MP4 videos on-the-fly.
    """
    
    def __init__(
        self,
        decoded_root: Path,
        context_frames: int = 8,
        n_camera_bins: int = 11,
        skip_null_actions: bool = True,
        max_cached_bundles: int = 8,
    ) -> None:
        self.decoded_root = Path(decoded_root)
        self.context_frames = context_frames
        self.skip_null_actions = skip_null_actions
        self.action_mapper = MineRLActionMapping(n_camera_bins=n_camera_bins)
        
        # Load manifest
        manifest_path = self.decoded_root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found at {manifest_path}")
        with open(manifest_path) as f:
            self.manifest = json.load(f)
        
        # Bundle cache (LRU)
        self._bundle_cache: Dict[int, Dict] = {}
        self._cache_order: List[int] = []
        self._max_cached = max(1, max_cached_bundles)
        
        # Build index
        self._trajectory_data: Dict[int, Dict] = {}
        self.samples: List[Tuple[int, int]] = []
        self._build_index()
    
    def _load_bundle(self, traj_id: int) -> Dict:
        """Load a trajectory bundle with LRU caching."""
        if traj_id in self._bundle_cache:
            # Move to end of LRU
            self._cache_order.remove(traj_id)
            self._cache_order.append(traj_id)
            return self._bundle_cache[traj_id]
        
        # Load from disk
        traj_info = self.manifest["trajectories"][traj_id]
        bundle_path = self.decoded_root / traj_info["decoded_file"]
        bundle = torch.load(bundle_path, weights_only=False)
        
        # Cache
        self._bundle_cache[traj_id] = bundle
        self._cache_order.append(traj_id)
        
        # Evict if over limit
        while len(self._cache_order) > self._max_cached:
            old_id = self._cache_order.pop(0)
            self._bundle_cache.pop(old_id, None)
        
        return bundle
    
    def _build_index(self) -> None:
        """Build sample index from manifest."""
        for traj_info in self.manifest["trajectories"]:
            traj_id = traj_info["traj_id"]
            bundle = self._load_bundle(traj_id)
            
            # Process actions
            npz_data = {k: bundle[k] for k in bundle.get("npz_keys", []) if k in bundle}
            if not npz_data:
                # Fallback: find action keys
                npz_data = {k: v for k, v in bundle.items() 
                           if k.startswith("action$") or k.startswith("observation$")}
            
            actions = self.action_mapper.process_npz_actions(npz_data)
            observations = self.action_mapper.process_npz_observations(npz_data)
            
            num_frames = len(bundle["frames"])
            num_actions = len(actions["binary_buttons"])
            usable = min(num_frames, num_actions)
            
            if usable < self.context_frames:
                continue
            
            self._trajectory_data[traj_id] = {
                "actions": actions,
                "observations": observations,
                "num_frames": usable,
            }
            
            # Build samples
            for end_idx in range(self.context_frames - 1, usable):
                if self.skip_null_actions and self._is_null_action(actions, end_idx):
                    continue
                self.samples.append((traj_id, end_idx))
        
        # Clear cache after indexing (will reload on demand)
        self._bundle_cache.clear()
        self._cache_order.clear()
        
        print(f"MineRLDiamondDecodedDataset: {len(self.samples)} samples from "
              f"{len(self._trajectory_data)} trajectories")
    
    def _is_null_action(self, actions: Dict[str, np.ndarray], idx: int) -> bool:
        """Check if action at idx is null."""
        if actions["binary_buttons"][idx].sum() > 0:
            return False
        cam = actions["camera"][idx]
        null_bin = self.action_mapper.camera_null_bin
        if cam[0] != null_bin or cam[1] != null_bin:
            return False
        for key in ["place", "equip", "craft", "nearby_craft", "nearby_smelt"]:
            if actions[key][idx] != 0:
                return False
        return True
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        traj_id, end_idx = self.samples[index]
        start_idx = end_idx - self.context_frames + 1
        
        bundle = self._load_bundle(traj_id)
        traj_data = self._trajectory_data[traj_id]
        actions = traj_data["actions"]
        observations = traj_data["observations"]
        
        # Get frames from bundle
        frames = bundle["frames"][start_idx:end_idx + 1]  # [T, H, W, C]
        
        # Observations
        obs_start, obs_end = start_idx, end_idx + 1
        inventory = torch.from_numpy(observations["inventory"][obs_start:obs_end].copy())
        equipped_type = torch.from_numpy(observations["equipped_type"][obs_start:obs_end].copy())
        
        # Actions
        return {
            "frames": frames,
            "inventory": inventory,
            "equipped_type": equipped_type,
            "action_buttons": torch.from_numpy(actions["binary_buttons"][end_idx].copy()),
            "action_camera": torch.from_numpy(actions["camera"][end_idx].copy()),
            "action_place": torch.tensor(actions["place"][end_idx], dtype=torch.int64),
            "action_equip": torch.tensor(actions["equip"][end_idx], dtype=torch.int64),
            "action_craft": torch.tensor(actions["craft"][end_idx], dtype=torch.int64),
            "action_nearby_craft": torch.tensor(actions["nearby_craft"][end_idx], dtype=torch.int64),
            "action_nearby_smelt": torch.tensor(actions["nearby_smelt"][end_idx], dtype=torch.int64),
            "video_id": torch.tensor(traj_id, dtype=torch.int64),
            "frame_idx": torch.tensor(end_idx, dtype=torch.int64),
        }
    
    def get_action_space_info(self) -> Dict[str, int]:
        return self.action_mapper.get_action_space_info()
    
    def get_inventory_dim(self) -> int:
        return self.action_mapper.vocab.num_inventory_items
    
    def get_num_equipped_types(self) -> int:
        return self.action_mapper.vocab.num_equipped_types
    
    def get_trajectory_data(self) -> Dict[int, Dict]:
        """Direct access to trajectory data for fast class weight estimation."""
        return self._trajectory_data


# =============================================================================
# Convenience functions
# =============================================================================

def collate_minerl_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for MineRLDiamondDataset.
    
    Stacks all tensors in the batch dict.
    """
    result = {}
    for key in batch[0].keys():
        values = [item[key] for item in batch]
        if values[0].dim() == 0:  # scalar
            result[key] = torch.stack(values)
        else:
            result[key] = torch.stack(values)
    return result


if __name__ == "__main__":
    # Quick test
    import sys
    
    data_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "/Users/willi1/foundation-dagger/diffusion-forcing-transformer/data/task_diamond"
    )
    
    print(f"Testing MineRLDiamondDataset with data_root={data_root}")
    
    dataset = MineRLDiamondDataset(
        data_root=data_root,
        context_frames=8,
        n_camera_bins=11,
    )
    
    print(f"\nDataset size: {len(dataset)}")
    print(f"Action space info: {dataset.get_action_space_info()}")
    print(f"Inventory dim: {dataset.get_inventory_dim()}")
    
    # Test a sample
    sample = dataset[0]
    print("\nSample keys and shapes:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape} {value.dtype}")
        else:
            print(f"  {key}: {value}")
    
    print("\nSample action values:")
    print(f"  buttons: {sample['action_buttons'].tolist()}")
    print(f"  camera: {sample['action_camera'].tolist()}")
    print(f"  place: {sample['action_place'].item()}")
    print(f"  equip: {sample['action_equip'].item()}")
    print(f"  craft: {sample['action_craft'].item()}")
    print(f"  nearby_craft: {sample['action_nearby_craft'].item()}")
    print(f"  nearby_smelt: {sample['action_nearby_smelt'].item()}")
    
    print("\nSample inventory (last frame):")
    inv = sample["inventory"][-1]
    vocab = dataset.action_mapper.vocab
    for i, item_name in enumerate(vocab.INVENTORY_ITEMS):
        if inv[i] > 0:
            print(f"  {item_name}: {inv[i].item()}")

