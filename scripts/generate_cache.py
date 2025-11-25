
import argparse
import sys
from pathlib import Path
# from tqdm.auto import tqdm
import torch

# Add root to path so we can import dataset
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from datasets.mineworld_data.mineworld_frame_dataset import MineWorldDecodedFrameDataset

def generate_cache(data_root: Path):
    print(f"Initializing dataset from {data_root}...")
    # Initialize dataset with memmap enabled
    dataset = MineWorldDecodedFrameDataset(
        decoded_root=data_root,
        context_frames=1,
        use_memmap_frames=True,
        max_cached_videos=1  # Minimize memory usage, we just want to trigger generation
    )
    
    print(f"Found {len(dataset._video_records)} videos.")
    print("Iterating through videos to ensure .npy cache files are generated...")
    
    # We iterate through video_records directly to trigger _load_video_bundle for each
    # This avoids iterating through every single frame sample
    for i, video_id in enumerate(dataset._video_records.keys()):
        if i % 10 == 0:
            print(f"Processing video {i}/{len(dataset._video_records)}...", end='\r')
        try:
            # This method triggers the check and generation of .npy files
            dataset._load_video_bundle(video_id)
        except Exception as e:
            print(f"Failed to generate cache for video {video_id}: {e}")

    print("\nCache generation complete. You can now run training safely.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=Path, help="Path to the decoded dataset directory")
    args = parser.parse_args()
    
    generate_cache(args.data_root)

