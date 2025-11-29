"""
Evaluate a Diamond BC policy (MineRLPolicy) inside a MineRL ObtainDiamond environment.

This script evaluates policies trained with train_diamond_bc.py, which use:
- 64x64 frame inputs
- Inventory and equipped item observations
- Factored action space with binary buttons + categorical actions

Example:
    python scripts/minerl_evaluation_diamond.py \
        --checkpoint checkpoints/diamond_bc_best.ckpt \
        --env-id MineRLObtainDiamondShovel-v0 \
        --episodes 5 \
        --max-steps 3000
"""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import gym
import minerl  # noqa: F401 - ensures MineRL envs are registered
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs" / "diamond_eval"

# MineRL Diamond environment resolution
DIAMOND_RESOLUTION = (64, 64)  # H, W

# Available MineRL Diamond environments
DIAMOND_ENVS = [
    "MineRLObtainDiamond-v0",
    "MineRLObtainDiamondShovel-v0", 
    "MineRLObtainIronPickaxe-v0",
    "MineRLObtainIronPickaxeDense-v0",
    "MineRLObtainDiamondDense-v0",
]


def _reset_env(env, *, seed=None) -> Tuple[Dict[str, Any], Dict]:
    """Handle Gymnasium/Gym reset return conventions."""
    if seed is not None:
        try:
            env.seed(seed)
        except (AttributeError, TypeError):
            pass
    reset_out = env.reset()
    if isinstance(reset_out, tuple):
        obs, info = reset_out
    else:
        obs, info = reset_out, {}
    return obs, info


def _step_env(env, action) -> Tuple[Dict[str, Any], float, bool, Dict]:
    """Handle Gymnasium/Gym step return conventions."""
    step_out = env.step(action)
    if len(step_out) == 5:
        obs, reward, terminated, truncated, info = step_out
        done = bool(terminated or truncated)
    else:
        obs, reward, done, info = step_out
    return obs, reward, done, info


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _extract_frame(obs: Any) -> Optional[np.ndarray]:
    """Extract POV frame from observation dict."""
    if isinstance(obs, dict):
        frame = obs.get("pov")
        if frame is not None:
            return frame
    return None


class EpisodeVideoRecorder:
    """Simple RGB video recorder backed by OpenCV."""

    def __init__(self, path: Path, fps: int) -> None:
        self.path = path
        self.fps = fps
        self._writer = None

    def append(self, frame: Any) -> None:
        if frame is None:
            return
        if self._writer is None:
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, (width, height))
            if not self._writer.isOpened():
                self._writer.release()
                self._writer = None
                raise RuntimeError(f"Could not open video writer at {self.path}")
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self._writer.write(bgr_frame)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


def _print_action_space(action_space: gym.spaces.Dict) -> None:
    """Print action space details for debugging."""
    print("\nEnvironment Action Space:")
    for key, space in action_space.spaces.items():
        if isinstance(space, gym.spaces.Discrete):
            print(f"  {key}: Discrete({space.n})")
        elif isinstance(space, gym.spaces.Box):
            print(f"  {key}: Box{space.shape} [{space.low.flat[0]:.1f}, {space.high.flat[0]:.1f}]")
        elif hasattr(space, 'values'):
            # Enum-like space
            print(f"  {key}: Enum({getattr(space, 'values', 'unknown')})")
        else:
            print(f"  {key}: {type(space).__name__}")


def _print_observation_space(obs_space) -> None:
    """Print observation space details for debugging."""
    print("\nEnvironment Observation Space:")
    if isinstance(obs_space, gym.spaces.Dict):
        for key, space in obs_space.spaces.items():
            if isinstance(space, gym.spaces.Box):
                print(f"  {key}: Box{space.shape}")
            elif isinstance(space, gym.spaces.Dict):
                print(f"  {key}: Dict with {len(space.spaces)} sub-spaces")
            else:
                print(f"  {key}: {type(space).__name__}")


class DiamondBCAgent:
    """
    Agent wrapper for MineRLPolicy (Diamond BC).
    
    Handles:
    - Frame buffer for context windows
    - Inventory/equipped item extraction from observations
    - Action mapping from model output to environment format
    """
    
    # Inventory items in the order expected by the model
    INVENTORY_ITEMS = [
        "coal", "cobblestone", "crafting_table", "dirt", "furnace",
        "iron_axe", "iron_ingot", "iron_ore", "iron_pickaxe", "log",
        "planks", "stick", "stone", "stone_axe", "stone_pickaxe",
        "torch", "wooden_axe", "wooden_pickaxe"
    ]
    
    # Equipped item type mapping
    EQUIPPED_TYPES = ["none", "other", "air"]
    
    # Categorical action vocabularies (index -> item name)
    PLACE_VOCAB = ["none", "cobblestone", "crafting_table", "dirt", "furnace", "stone", "torch"]
    EQUIP_VOCAB = ["none", "iron_axe", "iron_pickaxe", "stone_axe", "stone_pickaxe", "wooden_pickaxe"]
    CRAFT_VOCAB = ["none", "crafting_table", "planks", "stick", "torch"]
    NEARBY_CRAFT_VOCAB = ["none", "furnace", "iron_axe", "iron_pickaxe", "stone_axe", "stone_pickaxe", "wooden_pickaxe"]
    NEARBY_SMELT_VOCAB = ["none", "coal", "iron_ingot"]

    def __init__(
        self,
        checkpoint_path: Path,
        device: Optional[str] = None,
        context_frames: int = 8,
        n_camera_bins: int = 11,
        deterministic: bool = True,
        temperature: float = 1.0,
        camera_temperature: float = 1.0,
    ) -> None:
        import sys
        if str(ROOT_DIR) not in sys.path:
            sys.path.insert(0, str(ROOT_DIR))
        
        from algorithms.foundation_dagger.diamond_policy import (
            MineRLPolicy,
            MineRLPolicyConfig,
        )
        
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.context_frames = max(1, int(context_frames))
        self.n_camera_bins = n_camera_bins
        self.deterministic = deterministic
        self.temperature = temperature
        self.camera_temperature = camera_temperature
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Reconstruct config from checkpoint
        if "config" in checkpoint:
            cfg = checkpoint["config"]
            if isinstance(cfg, dict):
                cfg = MineRLPolicyConfig(**cfg)
        else:
            # Fallback to defaults
            cfg = MineRLPolicyConfig(
                backbone="small_cnn",
                vision_dim=256,
                n_camera_bins=n_camera_bins,
            )
        
        # Get action space info from checkpoint or use defaults
        action_space_info = checkpoint.get("action_space_info", {
            "num_place_classes": len(self.PLACE_VOCAB),
            "num_equip_classes": len(self.EQUIP_VOCAB),
            "num_craft_classes": len(self.CRAFT_VOCAB),
            "num_nearby_craft_classes": len(self.NEARBY_CRAFT_VOCAB),
            "num_nearby_smelt_classes": len(self.NEARBY_SMELT_VOCAB),
        })
        
        # Build model
        self.policy = MineRLPolicy(
            cfg=cfg,
            action_space_info=action_space_info,
            num_inventory_items=len(self.INVENTORY_ITEMS),
            num_equipped_types=len(self.EQUIPPED_TYPES),
        )
        
        # Load weights
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.policy.load_state_dict(state_dict)
        self.policy = self.policy.to(self.device)
        self.policy.eval()
        
        # Frame buffer for context window
        self.frame_buffer: deque = deque(maxlen=self.context_frames)
        self.inventory_buffer: deque = deque(maxlen=self.context_frames)
        self.equipped_buffer: deque = deque(maxlen=self.context_frames)
        
        # Precompute camera bin centers for action decoding
        # MUST use cfg.n_camera_bins (from checkpoint), not the cmdline n_camera_bins
        camera_max_angle = 5.0  # Must match dataset's camera_max_angle
        actual_bins = cfg.n_camera_bins
        bin_edges = np.linspace(-camera_max_angle, camera_max_angle, actual_bins + 1)
        self.camera_bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        self.n_camera_bins = actual_bins  # Override with actual value from checkpoint
        
        # Action space info (set via set_action_space)
        self._action_space = None
        self._action_keys = set()
        
        print(f"DiamondBCAgent loaded from {checkpoint_path}")
        print(f"  Device: {self.device}")
        print(f"  Context frames: {self.context_frames}")
        print(f"  Camera bins: {self.n_camera_bins}")
        print(f"  Model params: {sum(p.numel() for p in self.policy.parameters()):,}")

    def reset(self) -> None:
        """Reset agent state (clear buffers)."""
        self.frame_buffer.clear()
        self.inventory_buffer.clear()
        self.equipped_buffer.clear()
        self._debug_step_count = 0  # Reset debug counter

    def _extract_inventory(self, obs: Dict[str, Any]) -> np.ndarray:
        """Extract inventory counts from observation."""
        inventory = np.zeros(len(self.INVENTORY_ITEMS), dtype=np.int32)
        
        obs_inventory = obs.get("inventory", {})
        for i, item in enumerate(self.INVENTORY_ITEMS):
            if item in obs_inventory:
                inventory[i] = int(obs_inventory[item])
        
        return inventory

    def _extract_equipped_type(self, obs: Dict[str, Any]) -> int:
        """Extract equipped item type index from observation."""
        equipped = obs.get("equipped_items", {})
        mainhand = equipped.get("mainhand", {})
        item_type = str(mainhand.get("type", "none"))
        
        # Map to index
        if item_type in self.EQUIPPED_TYPES:
            return self.EQUIPPED_TYPES.index(item_type)
        elif item_type == "air" or item_type == "":
            return self.EQUIPPED_TYPES.index("air") if "air" in self.EQUIPPED_TYPES else 0
        else:
            return self.EQUIPPED_TYPES.index("other") if "other" in self.EQUIPPED_TYPES else 0

    def _prepare_frame(self, obs: Dict[str, Any]) -> torch.Tensor:
        """Prepare frame tensor from observation."""
        frame = obs["pov"]
        
        # Resize if needed (model expects 64x64)
        h, w = frame.shape[:2]
        if (h, w) != DIAMOND_RESOLUTION:
            frame = cv2.resize(frame, (DIAMOND_RESOLUTION[1], DIAMOND_RESOLUTION[0]))
        
        return torch.from_numpy(frame.copy())  # [H, W, C] uint8

    def _build_context(
        self,
        frame: torch.Tensor,
        inventory: np.ndarray,
        equipped_type: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build context tensors from buffers."""
        # Initialize buffers if empty
        if len(self.frame_buffer) == 0:
            for _ in range(self.context_frames):
                self.frame_buffer.append(frame.clone())
                self.inventory_buffer.append(inventory.copy())
                self.equipped_buffer.append(equipped_type)
        else:
            self.frame_buffer.append(frame)
            self.inventory_buffer.append(inventory)
            self.equipped_buffer.append(equipped_type)
        
        # Stack into tensors
        frames = torch.stack(list(self.frame_buffer), dim=0).unsqueeze(0)  # [1, T, H, W, C]
        
        inventory_tensor = torch.from_numpy(
            np.stack(list(self.inventory_buffer), axis=0)
        ).unsqueeze(0)  # [1, T, num_items]
        
        equipped_tensor = torch.tensor(
            list(self.equipped_buffer), dtype=torch.int64
        ).unsqueeze(0)  # [1, T]
        
        return frames.to(self.device), inventory_tensor.to(self.device), equipped_tensor.to(self.device)

    def _decode_camera(self, pitch_idx: int, yaw_idx: int) -> np.ndarray:
        """Convert camera bin indices to continuous angles."""
        pitch = self.camera_bin_centers[pitch_idx]
        yaw = self.camera_bin_centers[yaw_idx]
        return np.array([pitch, yaw], dtype=np.float32)

    def set_action_space(self, action_space: gym.spaces.Dict) -> None:
        """Configure agent to match the environment's action space."""
        self._action_space = action_space
        self._action_keys = set(action_space.spaces.keys())
        
        # Detect categorical action format (Enum vs string)
        for key in ["craft", "place", "equip", "nearbyCraft", "nearbySmelt"]:
            if key in self._action_keys:
                space = action_space.spaces[key]
                if hasattr(space, 'values'):
                    # Enum space - get valid values
                    print(f"  {key} values: {space.values}")

    def _actions_to_env(self, actions: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Convert policy actions to MineRL environment format."""
        env_action = {}
        
        # Binary buttons - model returns 0 for removed buttons in minimal mode
        button_map = {
            "forward": "forward",
            "left": "left", 
            "back": "back",
            "right": "right",
            "jump": "jump",
            "sneak": "sneak",
            "sprint": "sprint",
            "attack": "attack",
        }
        
        for policy_key, env_key in button_map.items():
            if env_key in self._action_keys and policy_key in actions:
                env_action[env_key] = int(actions[policy_key].item())
        
        # Camera
        if "camera" in self._action_keys:
            pitch_idx = actions["camera_pitch"].item()
            yaw_idx = actions["camera_yaw"].item()
            env_action["camera"] = self._decode_camera(pitch_idx, yaw_idx)
            # DEBUG: Print action predictions for first 100 steps
            if hasattr(self, '_debug_step_count'):
                self._debug_step_count += 1
            else:
                self._debug_step_count = 1
            if self._debug_step_count <= 100 and hasattr(self, '_last_logits'):
                # Camera
                pitch_probs = self._last_logits["camera_pitch"][:, -1].exp().cpu().numpy()[0]
                yaw_probs = self._last_logits["camera_yaw"][:, -1].exp().cpu().numpy()[0]
                
                # Buttons (only those that exist in the model)
                btn_names = ["fwd", "left", "back", "right", "jump", "sneak", "sprint", "attack"]
                btn_probs = []
                for name in btn_names:
                    key = f"button_{name}"
                    if key in self._last_logits:
                        p1 = self._last_logits[key][:, -1].exp().cpu().numpy()[0][1]  # prob of action=1
                        btn_probs.append(f"{name[:3]}:{p1:.2f}")
                    else:
                        btn_probs.append(f"{name[:3]}:N/A")
                # Craft (show top prediction)
                craft_probs = self._last_logits["craft"][:, -1].exp().cpu().numpy()[0]
                craft_idx = craft_probs.argmax()
                craft_str = f"craft:{craft_idx}({craft_probs[craft_idx]:.2f})"
                # Print
                pitch_str = " ".join([f"b{i}:{p:.2f}" for i, p in enumerate(pitch_probs)])
                yaw_str = " ".join([f"b{i}:{p:.2f}" for i, p in enumerate(yaw_probs)])
                btn_str = " ".join(btn_probs)
                print(f"  Step {self._debug_step_count}: btns=[{btn_str}] cam_p=[{pitch_str}] cam_y=[{yaw_str}] {craft_str}")
        
        # Categorical actions - handle both string and enum formats
        categorical_map = {
            "place": (self.PLACE_VOCAB, "place"),
            "equip": (self.EQUIP_VOCAB, "equip"),
            "craft": (self.CRAFT_VOCAB, "craft"),
            "nearby_craft": (self.NEARBY_CRAFT_VOCAB, "nearbyCraft"),
            "nearby_smelt": (self.NEARBY_SMELT_VOCAB, "nearbySmelt"),
        }
        
        for policy_key, (vocab, env_key) in categorical_map.items():
            if env_key in self._action_keys:
                idx = actions[policy_key].item()
                value = vocab[idx]
                
                # Check if the space is Enum-like (has 'values' attribute)
                space = self._action_space.spaces[env_key]
                if hasattr(space, 'values'):
                    # Enum space - find matching value or use 'none'
                    valid_values = list(space.values)
                    if value in valid_values:
                        env_action[env_key] = value
                    else:
                        env_action[env_key] = valid_values[0]  # Default to first value
                else:
                    env_action[env_key] = value
        
        # Fill in any missing required actions with defaults
        for key, space in self._action_space.spaces.items():
            if key not in env_action:
                if isinstance(space, gym.spaces.Discrete):
                    env_action[key] = 0
                elif isinstance(space, gym.spaces.Box):
                    env_action[key] = np.zeros(space.shape, dtype=space.dtype)
                elif hasattr(space, 'values'):
                    env_action[key] = list(space.values)[0]
                else:
                    env_action[key] = space.sample()
        
        # ALWAYS force sneak/sprint to 0 (mixed keybinds in training data)
        # This MUST be outside all conditional blocks to apply every step
        env_action["sneak"] = 0
        env_action["sprint"] = 0
        
        return env_action

    def get_action(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Get action for given observation."""
        # Extract observations
        frame = self._prepare_frame(obs)
        inventory = self._extract_inventory(obs)
        equipped_type = self._extract_equipped_type(obs)
        
        # Build context
        frames, inventory_tensor, equipped_tensor = self._build_context(
            frame, inventory, equipped_type
        )
        
        # Get model prediction
        with torch.no_grad():
            # Get raw logits for debugging
            self._last_logits = self.policy(frames, inventory_tensor, equipped_tensor)
            actions = self.policy.predict_action(
                frames, inventory_tensor, equipped_tensor,
                deterministic=self.deterministic,
                temperature=self.temperature,
                camera_temperature=self.camera_temperature,
            )
        
        # Convert to environment format
        env_action = self._actions_to_env(actions)
        
        return env_action


def _make_warmup_action(action_space: gym.spaces.Dict) -> Dict[str, Any]:
    """Construct a simple action that walks forward while looking around."""
    action = {}
    
    # Initialize all actions to default values
    for key, space in action_space.spaces.items():
        if isinstance(space, gym.spaces.Discrete):
            action[key] = 0
        elif isinstance(space, gym.spaces.Box):
            action[key] = np.zeros(space.shape, dtype=space.dtype)
        else:
            action[key] = "none"
    
    # Walk forward (no sprint - mixed keybinds in training data)
    if "forward" in action:
        action["forward"] = 1
    
    return action


def evaluate_policy(args: argparse.Namespace) -> None:
    """Run evaluation loop."""
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Create environment first to get action space
    print(f"\nCreating environment: {args.env_id}")
    env = gym.make(args.env_id)
    
    # Print environment spaces for debugging
    if args.verbose:
        _print_action_space(env.action_space)
        _print_observation_space(env.observation_space)
    
    # Create agent
    agent = DiamondBCAgent(
        checkpoint_path=checkpoint_path,
        device=args.device,
        context_frames=args.context_frames,
        n_camera_bins=args.n_camera_bins,
        deterministic=not args.stochastic,
        temperature=args.temperature,
        camera_temperature=args.camera_temperature,
    )
    
    # Configure agent for this environment's action space
    agent.set_action_space(env.action_space)
    
    # Setup output directory
    output_root = _ensure_dir(Path(args.output_dir).expanduser().resolve())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = _ensure_dir(output_root / f"{checkpoint_path.stem}_{timestamp}")
    print(f"Output directory: {run_dir}")
    
    # Warmup action
    warmup_action = _make_warmup_action(env.action_space) if args.warmup_steps > 0 else None
    
    # Evaluation metrics
    all_rewards = []
    all_steps = []
    
    for episode_idx in range(args.episodes):
        seed = (args.seed + episode_idx) if args.seed is not None else None
        obs, _ = _reset_env(env, seed=seed)
        agent.reset()
        
        total_reward = 0.0
        steps = 0
        done = False
        
        # Setup video recording
        video_path = run_dir / f"episode_{episode_idx:03d}.mp4"
        recorder = EpisodeVideoRecorder(video_path, fps=args.video_fps)
        recorder.append(_extract_frame(obs))
        
        # Warmup steps
        if warmup_action is not None:
            print(f"Episode {episode_idx + 1}: Running {args.warmup_steps} warmup steps...")
            for _ in range(args.warmup_steps):
                obs, _, done, _ = _step_env(env, warmup_action)
                recorder.append(_extract_frame(obs))
                if done:
                    break
            agent.reset()
        
        # Main evaluation loop
        print(f"Episode {episode_idx + 1}: Running policy...")
        try:
            last_action = None
            while not done and (args.max_steps is None or steps < args.max_steps):
                # Sample new action every 2 steps
                if steps % 2 == 0 or last_action is None:
                    last_action = agent.get_action(obs)
                
                action = last_action
                obs, reward, done, info = _step_env(env, action)
                
                if args.render:
                    env.render()
                
                recorder.append(_extract_frame(obs))
                
                total_reward += float(reward)
                steps += 1
                
                # Print progress
                if steps % 500 == 0:
                    print(f"  Step {steps}: reward={total_reward:.2f}")
                    
        finally:
            recorder.close()
        
        all_rewards.append(total_reward)
        all_steps.append(steps)
        
        print(f"Episode {episode_idx + 1}: reward={total_reward:.2f}, steps={steps}, video={video_path}")
    
    # Summary statistics
    print("\n" + "=" * 50)
    print("Evaluation Summary")
    print("=" * 50)
    print(f"Episodes: {args.episodes}")
    print(f"Environment: {args.env_id}")
    print(f"Checkpoint: {checkpoint_path.name}")
    print(f"Mean reward: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    print(f"Mean steps: {np.mean(all_steps):.0f} ± {np.std(all_steps):.0f}")
    print(f"Max reward: {np.max(all_rewards):.2f}")
    print(f"Videos saved to: {run_dir}")
    
    # Save results to file
    results = {
        "checkpoint": str(checkpoint_path),
        "env_id": args.env_id,
        "episodes": args.episodes,
        "rewards": all_rewards,
        "steps": all_steps,
        "mean_reward": float(np.mean(all_rewards)),
        "std_reward": float(np.std(all_rewards)),
        "mean_steps": float(np.mean(all_steps)),
        "max_reward": float(np.max(all_rewards)),
    }
    
    import json
    results_path = run_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")
    
    env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a Diamond BC policy checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default="MineRLObtainDiamondShovel-v0",
        choices=DIAMOND_ENVS,
        help=f"MineRL environment ID. Available: {DIAMOND_ENVS}",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to store evaluation results and videos.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=6000,
        help="Maximum steps per episode (default: 6000, roughly 5 minutes at 20 fps).",
    )
    parser.add_argument(
        "--context-frames",
        type=int,
        default=8,
        help="Number of context frames for the policy.",
    )
    parser.add_argument(
        "--n-camera-bins",
        type=int,
        default=11,
        help="Number of camera bins per axis (must match training).",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=20,
        help="Frames per second for recorded videos.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default=None,
        help="Computation device override.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base random seed for environment resets.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the MineRL environment locally during evaluation.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic action sampling instead of argmax.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for buttons/categorical in stochastic mode.",
    )
    parser.add_argument(
        "--camera-temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for camera in stochastic mode.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=20,
        help="Number of scripted forward steps to take after reset before running policy.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed environment information (action/observation spaces).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    evaluate_policy(parse_args())

