#!/usr/bin/env python3
"""Split MineWorld-style datasets into train/validation directories.

Source directory layout::

    source_root/
        TaskA/
            sample_0001.mp4
            sample_0001.jsonl
        TaskB/
            ...

The script mirrors the task folder names under ``train`` and ``val`` inside
the destination directory. By default it creates symlinks to avoid copying
large video files, but ``--mode copy`` is available when symlinks are not an
option.

    python scripts/split_mineworld.py \
      --source data/mineworld \
      --destination data/mineworld_split \
      --val-fraction 0.2 \
      --seed 42
"""


from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


SUPPORTED_SUFFIXES = {'.mp4', '.jsonl'}


@dataclass
class Sample:
    stem: str
    files: List[Path]


def discover_samples(task_dir: Path) -> List[Sample]:
    suffix_map: Dict[str, Dict[str, Path]] = {}
    for path in task_dir.iterdir():
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            continue
        suffix_map.setdefault(path.stem, {})[suffix] = path

    samples: List[Sample] = []
    for stem, files in suffix_map.items():
        missing = sorted(SUPPORTED_SUFFIXES - set(files.keys()))
        if missing:
            raise FileNotFoundError(
                f"Sample '{stem}' in {task_dir} is missing files: {', '.join(missing)}"
            )
        samples.append(Sample(stem=stem, files=sorted(files.values())))
    return sorted(samples, key=lambda sample: sample.stem)


def materialize(samples: Iterable[Sample], destination: Path, mode: str) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        for src in sample.files:
            dst = destination / src.name
            if dst.exists():
                continue
            if mode == 'symlink':
                dst.symlink_to(src.resolve())
            else:
                shutil.copy2(src, dst)


def split_task(
    task_dir: Path,
    train_dir: Path,
    val_dir: Path,
    val_fraction: float,
    seed: int,
    mode: str,
) -> None:
    samples = discover_samples(task_dir)
    if not samples:
        return
    rng = random.Random(seed)
    rng.shuffle(samples)

    val_count = int(len(samples) * val_fraction)
    if val_fraction > 0 and val_count == 0 and len(samples) > 1:
        val_count = 1
    if val_count >= len(samples) and len(samples) > 1:
        val_count = len(samples) - 1

    val_samples = samples[:val_count]
    train_samples = samples[val_count:]

    materialize(train_samples, train_dir, mode)
    materialize(val_samples, val_dir, mode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='MineWorld data root.')
    parser.add_argument('--destination', type=Path, required=True, help='Output directory for splits.')
    parser.add_argument('--val-fraction', type=float, default=0.2, help='Fraction for validation set.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for the split.')
    parser.add_argument(
        '--mode', choices=('symlink', 'copy'), default='symlink',
        help='Create symlinks (default) or copy files into the destination.'
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    destination = args.destination.expanduser()

    if not source.exists():
        raise FileNotFoundError(f'Source directory {source} does not exist.')
    destination.mkdir(parents=True, exist_ok=True)

    train_root = destination / 'train'
    val_root = destination / 'val'
    train_root.mkdir(exist_ok=True)
    val_root.mkdir(exist_ok=True)

    for task_dir in sorted(p for p in source.iterdir() if p.is_dir()):
        train_dir = train_root / task_dir.name
        val_dir = val_root / task_dir.name
        split_task(
            task_dir=task_dir,
            train_dir=train_dir,
            val_dir=val_dir,
            val_fraction=args.val_fraction,
            seed=args.seed,
            mode=args.mode,
        )
        train_count = len(list(train_dir.glob('*.mp4')))
        val_count = len(list(val_dir.glob('*.mp4')))
        print(f"{task_dir.name}: {train_count} train / {val_count} val videos")


if __name__ == '__main__':
    main()
