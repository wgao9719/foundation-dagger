"""
Locality-preserving sampler that groups samples by video to maximize cache hits.

When using multiple DataLoader workers with a video-based dataset, each worker maintains
its own cache of loaded video bundles. Random shuffling causes constant cache misses as
workers access samples from different videos. This sampler groups samples by video,
ensuring workers tend to access the same video multiple times before moving to the next,
dramatically reducing disk I/O.
"""

from __future__ import annotations

import random
from typing import Dict, Iterator, List, Optional

from torch.utils.data import Sampler


class VideoGroupedSampler(Sampler[int]):
    """
    Sampler that groups samples by video_id for cache-friendly data loading.
    
    Strategy:
    1. Shuffle the order of videos each epoch
    2. For each video, yield samples in small random chunks (not strictly sequential)
    3. This provides both randomness and locality
    
    The chunk_size parameter controls the trade-off:
    - Larger chunks = better cache locality, less randomness within epoch
    - Smaller chunks = more randomness, slightly worse locality
    """
    
    def __init__(
        self,
        samples: List[tuple],  # List of (path, video_id, frame_idx) tuples
        chunk_size: int = 256,
        seed: int = 0,
        shuffle_videos: bool = True,
        shuffle_within_video: bool = True,
    ) -> None:
        """
        Args:
            samples: Dataset samples list, each tuple contains (path, video_id, frame_idx)
            chunk_size: Number of samples from same video to yield before potentially switching.
                       Set to a large value (e.g., 9999999) to process entire videos sequentially.
            seed: Random seed for reproducibility
            shuffle_videos: Whether to shuffle video order each epoch
            shuffle_within_video: Whether to shuffle sample order within each video
        """
        self.samples = samples
        self.chunk_size = max(1, chunk_size)
        self.seed = seed
        self.shuffle_videos = shuffle_videos
        self.shuffle_within_video = shuffle_within_video
        self.epoch = 0
        
        # Build video_id -> sample indices mapping
        self.video_to_indices: Dict[int, List[int]] = {}
        for idx, (_, video_id, _) in enumerate(samples):
            video_id = int(video_id)
            if video_id not in self.video_to_indices:
                self.video_to_indices[video_id] = []
            self.video_to_indices[video_id].append(idx)
        
        self.video_ids = list(self.video_to_indices.keys())
        self._total_samples = len(samples)
    
    def set_epoch(self, epoch: int) -> None:
        """Set epoch for deterministic shuffling across distributed workers."""
        self.epoch = epoch
    
    def __len__(self) -> int:
        return self._total_samples
    
    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        
        # Get video order for this epoch
        video_order = list(self.video_ids)
        if self.shuffle_videos:
            rng.shuffle(video_order)
        
        # For each video, get its samples (potentially shuffled)
        video_samples: Dict[int, List[int]] = {}
        for vid in video_order:
            indices = list(self.video_to_indices[vid])
            if self.shuffle_within_video:
                rng.shuffle(indices)
            video_samples[vid] = indices
        
        # Yield samples in chunks, interleaving videos
        video_positions = {vid: 0 for vid in video_order}
        active_videos = [vid for vid in video_order if video_samples[vid]]
        
        while active_videos:
            # Shuffle active videos for this round to add some variety
            if self.shuffle_videos:
                rng.shuffle(active_videos)
            
            next_active = []
            for vid in active_videos:
                pos = video_positions[vid]
                samples_list = video_samples[vid]
                end_pos = min(pos + self.chunk_size, len(samples_list))
                
                # Yield chunk of samples from this video
                for i in range(pos, end_pos):
                    yield samples_list[i]
                
                video_positions[vid] = end_pos
                if end_pos < len(samples_list):
                    next_active.append(vid)
            
            active_videos = next_active


class VideoSequentialSampler(Sampler[int]):
    """
    Sampler that yields all samples from each video sequentially before moving to the next.
    
    This provides maximum cache locality - each video is fully processed before moving on.
    Video order is shuffled each epoch for training variety.
    """
    
    def __init__(
        self,
        samples: List[tuple],
        seed: int = 0,
        shuffle_videos: bool = True,
        shuffle_within_video: bool = False,
    ) -> None:
        self.samples = samples
        self.seed = seed
        self.shuffle_videos = shuffle_videos
        self.shuffle_within_video = shuffle_within_video
        self.epoch = 0
        
        # Build video_id -> sample indices mapping
        self.video_to_indices: Dict[int, List[int]] = {}
        for idx, (_, video_id, _) in enumerate(samples):
            video_id = int(video_id)
            if video_id not in self.video_to_indices:
                self.video_to_indices[video_id] = []
            self.video_to_indices[video_id].append(idx)
        
        self.video_ids = list(self.video_to_indices.keys())
        self._total_samples = len(samples)
    
    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
    
    def __len__(self) -> int:
        return self._total_samples
    
    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        
        video_order = list(self.video_ids)
        if self.shuffle_videos:
            rng.shuffle(video_order)
        
        for vid in video_order:
            indices = list(self.video_to_indices[vid])
            if self.shuffle_within_video:
                rng.shuffle(indices)
            yield from indices


class WorkerAwareVideoSampler(Sampler[int]):
    """
    Advanced sampler that assigns specific videos to specific workers for maximum locality.
    
    When combined with worker_init_fn, this ensures each worker only ever needs to load
    a small subset of videos, completely eliminating cross-worker cache thrashing.
    
    Usage:
        sampler = WorkerAwareVideoSampler(dataset.samples, num_workers=40)
        loader = DataLoader(dataset, sampler=sampler, num_workers=40,
                           worker_init_fn=sampler.worker_init_fn)
    """
    
    def __init__(
        self,
        samples: List[tuple],
        num_workers: int,
        seed: int = 0,
        shuffle_within_worker: bool = True,
    ) -> None:
        self.samples = samples
        self.num_workers = max(1, num_workers)
        self.seed = seed
        self.shuffle_within_worker = shuffle_within_worker
        self.epoch = 0
        
        # Build video_id -> sample indices mapping  
        self.video_to_indices: Dict[int, List[int]] = {}
        for idx, (_, video_id, _) in enumerate(samples):
            video_id = int(video_id)
            if video_id not in self.video_to_indices:
                self.video_to_indices[video_id] = []
            self.video_to_indices[video_id].append(idx)
        
        self.video_ids = list(self.video_to_indices.keys())
        self._total_samples = len(samples)
        
        # Assign videos to workers (round-robin by sample count for balance)
        self._assign_videos_to_workers()
    
    def _assign_videos_to_workers(self) -> None:
        """Assign videos to workers, balancing by sample count."""
        # Sort videos by sample count (descending) for better load balancing
        videos_by_size = sorted(
            self.video_ids,
            key=lambda v: len(self.video_to_indices[v]),
            reverse=True
        )
        
        # Greedy assignment to balance worker loads
        worker_loads = [0] * self.num_workers
        self.worker_to_videos: Dict[int, List[int]] = {i: [] for i in range(self.num_workers)}
        
        for vid in videos_by_size:
            # Assign to worker with minimum load
            min_worker = min(range(self.num_workers), key=lambda w: worker_loads[w])
            self.worker_to_videos[min_worker].append(vid)
            worker_loads[min_worker] += len(self.video_to_indices[vid])
        
        # Build flat list of indices ordered by worker
        self._ordered_indices: List[int] = []
        self._worker_ranges: Dict[int, tuple] = {}
        
        start = 0
        for worker_id in range(self.num_workers):
            worker_indices = []
            for vid in self.worker_to_videos[worker_id]:
                worker_indices.extend(self.video_to_indices[vid])
            
            self._ordered_indices.extend(worker_indices)
            self._worker_ranges[worker_id] = (start, start + len(worker_indices))
            start += len(worker_indices)
    
    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
    
    def __len__(self) -> int:
        return self._total_samples
    
    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        
        # For each worker, shuffle its assigned samples
        all_indices = []
        for worker_id in range(self.num_workers):
            start, end = self._worker_ranges[worker_id]
            worker_indices = list(self._ordered_indices[start:end])
            if self.shuffle_within_worker:
                rng.shuffle(worker_indices)
            all_indices.extend(worker_indices)
        
        yield from all_indices
    
    def get_worker_videos(self, worker_id: int) -> List[int]:
        """Get list of video IDs assigned to a specific worker."""
        return self.worker_to_videos.get(worker_id % self.num_workers, [])

