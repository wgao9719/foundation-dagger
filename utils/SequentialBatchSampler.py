from __future__ import annotations

import math
from typing import Dict, List
from collections import deque
import random

class PerVideoSequentialBatchSampler:
    """
    Generates batches that contain at most one sample per video while advancing
    each video's samples in chronological order. The video ordering is reshuffled
    every epoch to avoid fixed inter-video patterns.
    """

    def __init__(
        self,
        video_to_positions: Dict[int, List[int]],
        batch_size: int,
        seed: int,
        drop_last: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        non_empty_videos = {vid: list(pos) for vid, pos in video_to_positions.items() if pos}
        if not non_empty_videos:
            raise ValueError("No samples available for the batch sampler.")
        if batch_size > len(non_empty_videos):
            raise ValueError(
                f"batch_size ({batch_size}) exceeds the number of unique videos ({len(non_empty_videos)}); "
                "cannot yield batches with unique video IDs."
            )
        self.video_to_positions = non_empty_videos
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        total = sum(len(pos_list) for pos_list in self.video_to_positions.values())
        if self.drop_last:
            return total // self.batch_size
        return math.ceil(total / self.batch_size)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        initial_order = list(self.video_to_positions.keys())
        rng.shuffle(initial_order)
        offsets = {vid: 0 for vid in initial_order}
        queue = deque([vid for vid in initial_order if self.video_to_positions[vid]])
        batch: List[int] = []

        while queue:
            vid = queue.popleft()
            positions = self.video_to_positions[vid]
            offset = offsets[vid]
            if offset >= len(positions):
                continue
            batch.append(positions[offset])
            offsets[vid] = offset + 1
            if offsets[vid] < len(positions):
                queue.append(vid)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if batch and not self.drop_last:
            yield batch
