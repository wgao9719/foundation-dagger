# Repository Guidelines

## Project Structure & Module Organization
`main.py` is the Hydra entry point that wires datasets, algorithms, and experiment recipes. Core model and sampler code lives in `algorithms/` (`dfot/`, `vae/`, plus shared kernels). Dataset wrappers sit in `datasets/video/`, while experiment scaffolds (Lightning modules, task runners) live under `experiments/`. Hydra configuration defaults and reusable bundles are in `configurations/` (`config.yaml`, `algorithm/`, `dataset/`, `shortcut/`). Shared utilities for logging, cluster submission, checkpoints, and geometry inhabit `utils/`. Generated runs land in `outputs/`, and demo-ready assets are staged in `huggingface/` or `data/`.

## Build, Test, and Development Commands
Create the recommended environment with `conda create -n dfot python=3.10` followed by `conda activate dfot` and `pip install -r requirements.txt`. Verify your setup with the lightweight RealEstate10K mini rollout: `python -m main +name=single_image_to_short dataset=realestate10k_mini algorithm=dfot_video_pose experiment=video_generation @diffusion/continuous`. For more frames, append overrides such as `dataset.n_frames=200 algorithm.tasks.prediction.history_guidance.guidance_scale=4.0`. Launch full RealEstate10K training (multi-GPU) via `python -m main +name=RE10k dataset=realestate10k algorithm=dfot_video_pose experiment=video_generation @diffusion/continuous`. Switch W&B logging off with `wandb.mode=disabled` for offline runs.

## Coding Style & Naming Conventions
Write Python with 4-space indents, trailing commas on multiline literals, and explicit type hints for Hydra-facing APIs. Prefer `snake_case` for variables and functions, `PascalCase` for Lightning modules, and kebab-case YAML filenames (`dataset/realestate10k_mini.yaml`). Place shared constants in `utils/` rather than duplicating them inside experiments. Keep overrides declarative; avoid embedding absolute paths or machine-specific tweaks inside committed configs.

## Testing Guidelines
This repository leans on task-level validation instead of unit suites. Run a deterministic smoke test before sending changes: `python -m main +name=dev_smoke dataset=realestate10k_mini experiment=video_generation experiment.tasks=[validation] wandb.mode=disabled outputs=outputs/dev`. For new datasets, add a minimal YAML under `configurations/dataset/` and rerun the smoke test with reduced `dataset.n_frames` and `algorithm.tasks.prediction.max_batch_size`.

## Commit & Pull Request Guidelines
This archive ships without `.git`, but upstream commits use concise, imperative subjects (`Add stabilized history guidance sweep`). Provide a short body outlining motivation, key modules touched, and validation commands. Pull requests should summarize configuration deltas, cite any new artifacts or pretrained weights, and link related issues or experiment dashboards. Include screenshots or video grids whenever the change alters rendering quality or datasets.

## Hydra & Environment Tips
Reuse existing `@diffusion/continuous` or other shortcut bundles before authoring new overrides. Set `WANDB__SERVICE_WAIT=300` for long syncs, and rely on `utils.cluster_utils.submit_slurm_job` or `python -m main +submit=true` when dispatching cluster jobs. Never commit API keys; read secrets from environment variables referenced inside `config.yaml`.
