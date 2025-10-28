from __future__ import annotations

from pathlib import Path
from typing import List

from omegaconf import DictConfig, OmegaConf

from algorithms.mineworld.model import GenerationResult, MineWorldModel
from utils import print0
from utils.distributed_utils import is_rank_zero
from .base_exp import BaseExperiment


class MineWorldInferenceExperiment(BaseExperiment):
    """
    Hydra-controlled MineWorld rollout pipeline.
    """

    compatible_algorithms = {
        "mineworld_700M_16f": MineWorldModel,
    }

    def __init__(
        self,
        root_cfg: DictConfig,
        logger=None,
        ckpt_path=None,
    ) -> None:
        super().__init__(root_cfg, logger, ckpt_path)

    def generate(self) -> None:
        if not self.algo:
            self.algo = self._build_algo()

        dataset_cfg = OmegaConf.to_container(self.root_cfg.dataset, resolve=True)
        gen_cfg = OmegaConf.to_container(self.cfg.generation, resolve=True)

        data_root = Path(dataset_cfg["data_root"]).expanduser()
        recursive = dataset_cfg.get("recursive", False)
        pattern = dataset_cfg.get("video_glob", "*.mp4")
        glob_pattern = f"**/{pattern}" if recursive else pattern

        videos = sorted(data_root.glob(glob_pattern))
        if not videos:
            raise FileNotFoundError(
                f"No videos found under {data_root} matching pattern '{pattern}'."
            )

        action_ext = dataset_cfg.get("action_extension", ".jsonl")
        limit = gen_cfg.get("limit")

        results: List[GenerationResult] = []
        processed = 0
        for video_path in videos:
            actions_path = video_path.with_suffix(action_ext)
            output_path = Path(gen_cfg["output_dir"]).expanduser() / video_path.name
            if not actions_path.exists():
                print0(
                    f"[yellow][MineWorld][/yellow] Missing action file for {video_path.name}, skipping."
                )
                results.append(
                    GenerationResult(
                        video_path=video_path,
                        actions_path=actions_path,
                        output_path=output_path,
                        elapsed=0.0,
                        generated_frames=0,
                        token_count=0,
                        skipped=True,
                    )
                )
                continue
            result = self.algo.generate_sequence(
                video_path=video_path,
                actions_path=actions_path if actions_path.exists() else None,
                output_path=output_path,
                context_frames=int(gen_cfg["context_frames"]),
                prediction_frames=int(gen_cfg["num_frames"]),
                accelerate_algo=gen_cfg.get("accelerate_algo", "naive"),
                window_size=int(gen_cfg.get("window_size", 2)),
                sampler=gen_cfg.get("sampler", {}),
                fps=int(gen_cfg.get("fps", 6)),
                overwrite=bool(gen_cfg.get("overwrite", False)),
                copy_actions=bool(gen_cfg.get("copy_actions", True)),
            )
            results.append(result)
            if not result.skipped:
                processed += 1
            if limit is not None and processed >= int(limit):
                break

        if is_rank_zero():
            self._summarize(results)

    def _summarize(self, results: List[GenerationResult]) -> None:
        completed = [res for res in results if not res.skipped]
        skipped = len(results) - len(completed)
        if not completed:
            print0("[yellow][MineWorld][/yellow] No videos were generated.")
            return
        total_time = sum(res.elapsed for res in completed)
        total_frames = sum(res.generated_frames for res in completed)
        total_tokens = sum(res.token_count for res in completed)
        print0(
            "[bold cyan][MineWorld][/bold cyan] Generation complete: "
            f"{len(completed)} videos, {skipped} skipped."
        )
        print0(
            f"  Avg time/video: {total_time / len(completed):.2f}s | "
            f"Avg frames/video: {total_frames / len(completed):.1f} | "
            f"Avg tokens/frame: {(total_tokens / total_frames):.1f}"
        )
