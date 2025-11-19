"""
python -m algorithms.mineworld.mineworld_testing \
    --checkpoint "/checkpoints/700M_16f.ckpt" \
    --output "outputs/mineworld/test_video.mp4"
"""


from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
from omegaconf import DictConfig, OmegaConf

from algorithms.mineworld.model import MineWorldModel
from utils import print0

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ALGO_CONFIG = REPO_ROOT / "configurations" / "algorithm" / "mineworld_700M_16f.yaml"
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "test_data"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "mineworld" / "smoke_test.mp4"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a MineWorld smoke test to verify checkpoints and dependencies."
    )
    parser.add_argument(
        "--algo-config",
        type=Path,
        default=DEFAULT_ALGO_CONFIG,
        help="Path to the MineWorld algorithm config (defaults to mineworld_700M_16f).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint override (local path or pretrained:<name>).",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Specific context video (.mp4). If omitted, the first sample under --data-root is used.",
    )
    parser.add_argument(
        "--actions",
        type=Path,
        default=None,
        help="Action log (.jsonl) paired with --video. Defaults to <video>.jsonl.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Directory containing sample MineWorld rollouts (defaults to data/test_data).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search --data-root recursively when auto-selecting a sample.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to write the generated video (defaults to outputs/mineworld/smoke_test.mp4).",
    )
    parser.add_argument(
        "--context-frames",
        type=int,
        default=8,
        help="Number of conditioning frames to feed into the tokenizer.",
    )
    parser.add_argument(
        "--prediction-frames",
        type=int,
        default=8,
        help="Number of frames to predict.",
    )
    parser.add_argument(
        "--accelerate-algo",
        choices=("naive", "image_diagd"),
        default="naive",
        help="Generation backend to use.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=4,
        help="Image diagonal decoding window size (only used for image_diagd).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=6,
        help="Frame rate for the rendered output video.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.8,
        help="Top-p nucleus sampling threshold.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling limit (omit for disabled).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    parser.add_argument(
        "--no-copy-actions",
        action="store_true",
        help="Disable copying the action jsonl next to the generated video.",
    )
    return parser.parse_args()


def _resolve_sample_paths(
    video: Optional[Path],
    actions: Optional[Path],
    data_root: Path,
    recursive: bool,
) -> Tuple[Path, Path]:
    if video is not None:
        video_path = video.expanduser().resolve()
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        actions_path = actions.expanduser().resolve() if actions else video_path.with_suffix(".jsonl")
        if not actions_path.exists():
            raise FileNotFoundError(
                f"Missing action log for {video_path.name}. Expected {actions_path}."
            )
        return video_path, actions_path

    data_root = data_root.expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    pattern = "**/*.mp4" if recursive else "*.mp4"
    for candidate in sorted(data_root.glob(pattern)):
        action_path = candidate.with_suffix(".jsonl")
        if action_path.exists():
            return candidate, action_path

    raise FileNotFoundError(
        f"No mp4/jsonl pairs found under {data_root} (recursive={recursive}). "
        "Specify --video/--actions to point at an existing sample."
    )


def _load_algorithm_config(algo_config: Path, checkpoint_override: Optional[str]) -> DictConfig:
    cfg_path = algo_config.expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Algorithm config not found: {cfg_path}")
    cfg = OmegaConf.load(str(cfg_path))
    if checkpoint_override:
        cfg["checkpoint"] = checkpoint_override
    if not cfg.get("checkpoint"):
        raise ValueError(
            "Algorithm config must define `checkpoint`. "
            "Provide one in the yaml or pass --checkpoint."
        )
    return cfg


def main() -> None:
    args = _parse_args()

    if not torch.cuda.is_available():
        print0("[red][MineWorld][/red] MineWorld testing requires a CUDA-capable GPU.")
        raise SystemExit(1)

    try:
        video_path, actions_path = _resolve_sample_paths(
            args.video, args.actions, args.data_root, args.recursive
        )
    except FileNotFoundError as exc:
        print0(f"[red][MineWorld][/red] {exc}")
        raise SystemExit(1)

    try:
        algo_cfg = _load_algorithm_config(args.algo_config, args.checkpoint)
    except (FileNotFoundError, ValueError) as exc:
        print0(f"[red][MineWorld][/red] {exc}")
        raise SystemExit(1)

    output_path = args.output.expanduser()
    if not output_path.is_absolute():
        output_path = output_path.resolve()

    print0("[bold cyan][MineWorld][/bold cyan] Starting smoke test run.")
    print0(f"  Video:   {video_path}")
    print0(f"  Actions: {actions_path}")
    print0(f"  Output:  {output_path}")

    try:
        world_model = MineWorldModel(algo_cfg)
    except Exception as exc:  # pragma: no cover - initialization errors are fatal
        print0(f"[red][MineWorld][/red] Failed to create MineWorldModel: {exc}")
        raise SystemExit(1) from exc

    sampler_cfg = {"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k}

    result = world_model.generate_sequence(
        video_path=video_path,
        actions_path=actions_path,
        output_path=output_path,
        context_frames=args.context_frames,
        prediction_frames=args.prediction_frames,
        accelerate_algo=args.accelerate_algo,
        window_size=args.window_size,
        sampler=sampler_cfg,
        fps=args.fps,
        overwrite=args.overwrite,
        copy_actions=not args.no_copy_actions,
    )

    if result.skipped:
        print0(
            "[yellow][MineWorld][/yellow] Generation skipped because the output already exists. "
            "Rerun with --overwrite to force regeneration."
        )
        return

    print0(
        "[bold green][MineWorld][/bold green] Smoke test complete: "
        f"{result.generated_frames} frames | {result.token_count} tokens | "
        f"{result.elapsed:.2f}s elapsed."
    )
    print0(f"[bold green]Output stored at:[/bold green] {result.output_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print0("[yellow][MineWorld][/yellow] Interrupted by user.")
        sys.exit(130)
