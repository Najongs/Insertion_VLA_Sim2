#!/usr/bin/env python
"""
Convert REAL folder-structured VLA episodes to LeRobot v3.0 dataset format.
Handles heterogeneous datesets with different views, frame-state synchronization,
and variable tasks based on folder names.

Dataset structure example:
- /data/public/NAS/Insertion_VLA_Sim2/Dataset/dataset/New_dataset2/Blue_point/data_collection_20251108_055533
  - View1/
    - ZED_..._1762548933.781.jpg
  - View2/
  - View.../
  - metadata.json
  - robot_states.npz

Usage:
    python convert_folder_to_lerobot.py --dataset-dir /path/to/New_dataset2 --repo-id my_repo --num-workers 8
"""

import argparse
import json
import logging
import subprocess
import sys
import time
import traceback
from pathlib import Path
import re

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "lerobot" / "src"))
from lerobot.datasets.lerobot_dataset import LeRobotDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def map_task_name(folder_name: str) -> str:
    """Guess the task instruction based on the parent folder name."""
    task_map = {
        "Blue_point": "Insert the needle towards the blue point",
        "Red_point": "Insert the needle towards the red point",
        "Green_point": "Insert the needle towards the green point",
        "White_point": "Insert the needle towards the white point",
        "Yellow_point": "Insert the needle towards the yellow point",
        "Eye_trocar": "Insert the needle through the eye phantom trocar opening"
    }
    # Fallback: replace underscores with spaces
    fallback = f"Move the tool to the {folder_name.replace('_', ' ').lower()}"
    return task_map.get(folder_name, fallback)

def parse_image_timestamp(filename: str) -> float:
    """Extract Unix timestamp from filename like ZED_41182735_left_1762548933.781.jpg"""
    # Just look for the float pattern at the end: _1762548933.781.jpg
    match = re.search(r"_([\d\.]+)\.je?pg$", filename, re.IGNORECASE)
    if match:
        return float(match.group(1))
    
    # As a fallback if the regex fails, just split by underscore and remove extension
    try:
        stem = Path(filename).stem
        parts = stem.split('_')
        return float(parts[-1])
    except Exception as e:
        raise ValueError(f"Could not parse timestamp from {filename}: {e}")

def create_features(cameras: list[str]) -> dict:
    """Dynamically create LeRobot FEATURES schema dict including found cameras."""
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": {"axes": ["x", "y", "z", "rx", "ry", "rz"]},
        },
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": {"axes": ["dx", "dy", "dz", "drx", "dry", "drz"]},
        },
    }
    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": (480, 640, 3), # Expected standard, will auto-resize if different via PIL
            "names": ["height", "width", "channels"],
        }
    return features

# ─── Single shard conversion ──────────────────────────────────────────────────

def convert_shard(
    episode_dirs: list[Path],
    output_dir: Path,
    repo_id: str,
    fps: int,
    vcodec: str,
    image_writer_threads: int,
    shard_id: int,
    total_shards: int,
):
    shard_repo_id = f"{repo_id}_shard{shard_id:03d}" if total_shards > 1 else repo_id
    shard_output = output_dir / shard_repo_id if total_shards > 1 else output_dir

    if shard_output.exists():
        import shutil
        shutil.rmtree(shard_output)

    total_episodes = len(episode_dirs)
    logger.info(f"[Shard {shard_id}/{total_shards}] Converting {total_episodes} episodes → {shard_output}")

    # To initialize LeRobotDataset, we need to know the camera views which must be consistent
    # across ALL episodes in this specific repo. We'll use the first episode to determine schema.
    first_ep = episode_dirs[0]
    with open(first_ep / "metadata.json", "r") as f:
        meta = json.load(f)
    
    cameras = meta.get("camera_views", [])
    if not cameras: # fallback logic if metadata is empty
        cameras = [cam.name for cam in first_ep.iterdir() if cam.is_dir() and cam.name.startswith("View")]
        cameras.sort()
    
    features = create_features(cameras)

    dataset = LeRobotDataset.create(
        repo_id=shard_repo_id,
        fps=fps,
        robot_type="surgical_robot_dynamic",
        features=features,
        root=str(shard_output),
        use_videos=True,
        image_writer_threads=image_writer_threads,
        batch_encoding_size=1,
        vcodec=vcodec,
    )

    start_time = time.time()
    total_frames = 0
    skipped = 0

    for ep_idx, ep_dir in enumerate(episode_dirs):
        try:
            # 1. Infer task naturally from parent directory (e.g. Blue_point)
            task_desc = map_task_name(ep_dir.parent.name)

            # 2. Load 100Hz npz array
            npz_path = ep_dir / "robot_states.npz"
            if not npz_path.exists():
                raise FileNotFoundError(f"Missing {npz_path}")
            
            states_npz = np.load(str(npz_path))
            t_states = states_npz["timestamps"]
            poses = states_npz["poses"]

            # Load action horizon
            with open(ep_dir / "metadata.json", "r") as f:
                meta = json.load(f)
            
            # Usually 8 or similar future sequence step
            action_horizon = meta.get("action_horizon", 1)  

            # 3. Read image sequences from the main View (e.g. View1) to dictate temporal flow
            main_view = cameras[0]
            view_dir = ep_dir / main_view
            image_paths = sorted(list(view_dir.glob("*.jpg")) + list(view_dir.glob("*.jpeg")))

            for img_path in image_paths:
                img_stamp = parse_image_timestamp(img_path.name)
                
                # Find closest index in robot_states timestamp
                state_idx = np.argmin(np.abs(t_states - img_stamp))
                
                # Fetch state
                state = poses[state_idx].astype(np.float32)

                # Fetch action using lookahead
                # Action is the target pose `action_horizon` steps into the future, clipped to end of array
                next_idx = min(state_idx + action_horizon, len(poses) - 1)
                action = poses[next_idx].astype(np.float32)

                frame = {
                    "task": task_desc,
                    "observation.state": state,
                    "action": action,
                }

                # Load all corresponding camera images
                # We assume images for other views have exactly the same timestamp in their filename,
                # OR they are sorted in the same order. Let's just use the same file name, replacing if needed,
                # actually, sorting by timestamp in each View directory is safer.
                
                # To be absolutely robust, let's load other views by matching their timestamp closest to img_stamp 
                # OR simply using the index if we assume frame dropping is symmetric.
                # Assuming symmetric frames for all Views (i.e. length matches).
                for cam in cameras: # including main view
                    cam_dir = ep_dir / cam
                    cam_img_paths = sorted(list(cam_dir.glob("*.jpg")) + list(cam_dir.glob("*.jpeg")))
                    
                    # We can't rely on strict indices because some Views have dropped/extra frames.
                    # We must find the closest timestamp in cam_img_paths to `img_stamp`.
                    
                    if not cam_img_paths:
                        raise ValueError(f"No frames found for {cam} in {ep_dir}")
                    
                    # Parse all timestamps once for this camera view if needed, 
                    # but doing it iteratively is fine for ~200 frames.
                    # Optimization: create a list of (timestamp, path)
                    cam_stamps = [parse_image_timestamp(p.name) for p in cam_img_paths]
                    
                    # Find closest image by timestamp
                    closest_idx = np.argmin(np.abs(np.array(cam_stamps) - img_stamp))
                    target_path = cam_img_paths[closest_idx]

                    # Read image
                    img = Image.open(str(target_path)).convert("RGB")
                    # Resize gently if it strictly must match 480x640.
                    if img.size != (640, 480):
                        img = img.resize((640, 480), Image.BILINEAR)
                    frame[f"observation.images.{cam}"] = img
                
                dataset.add_frame(frame)
            
            dataset.save_episode()
            total_frames += len(image_paths)

        except Exception as e:
            logger.error(f"[Shard {shard_id}] Failed ep {ep_idx} ({ep_dir.name}): {e}")
            logger.error(traceback.format_exc())
            skipped += 1
            try:
                if dataset.episode_buffer is not None and dataset.episode_buffer["size"] > 0:
                    dataset.clear_episode_buffer()
                else:
                    dataset.episode_buffer = dataset.create_episode_buffer()
            except Exception:
                dataset.episode_buffer = dataset.create_episode_buffer()
            continue

        if (ep_idx + 1) % 10 == 0 or (ep_idx + 1) == total_episodes:
            elapsed = time.time() - start_time
            eps_per_sec = (ep_idx + 1 - skipped) / elapsed if elapsed > 0 else 0
            eta = (total_episodes - ep_idx - 1) / eps_per_sec if eps_per_sec > 0 else 0
            logger.info(
                f"[Shard {shard_id}] [{ep_idx + 1}/{total_episodes}] "
                f"frames={total_frames:,} | skipped={skipped} | "
                f"{eps_per_sec:.2f} ep/s | ETA: {eta / 3600:.1f}h"
            )

    logger.info(f"[Shard {shard_id}] Finalizing...")
    dataset.finalize()

    elapsed = time.time() - start_time
    logger.info(f"[Shard {shard_id}] Done! {total_episodes - skipped}/{total_episodes} episodes in {elapsed / 3600:.1f}h")
    return shard_output

def merge_shards(output_dir: Path, repo_id: str, num_shards: int):
    from lerobot.datasets.aggregate import aggregate_datasets
    shard_dirs = []
    roots = []
    for i in range(num_shards):
        shard_repo = f"{repo_id}_shard{i:03d}"
        shard_path = output_dir / shard_repo
        if shard_path.exists():
            shard_dirs.append(shard_repo)
            roots.append(shard_path)

    if not shard_dirs:
        logger.error("No shards found to merge!")
        return

    aggr_path = output_dir / repo_id
    if aggr_path.exists():
        import shutil
        logger.warning(f"Removing existing dataset at {aggr_path}")
        shutil.rmtree(aggr_path)

    logger.info(f"Merging {len(shard_dirs)} shards into {aggr_path}...")
    aggregate_datasets(repo_ids=shard_dirs, aggr_repo_id=repo_id, roots=roots, aggr_root=aggr_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, default="/data/public/NAS/Insertion_VLA_Sim2/Dataset/dataset/New_dataset2")
    parser.add_argument("--output-dir", type=str, default="/data/public/NAS/Insertion_VLA_Sim2/TRAIN/SmolVLA/lerobot_multi_datasets")
    parser.add_argument("--repo-id", type=str, default="new_dataset_2")
    parser.add_argument("--fps", type=int, default=15) # Video encoded FPS
    parser.add_argument("--vcodec", type=str, default="h264")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument("--merge-only", action="store_true")
    
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    dataset_dir = Path(args.dataset_dir)

    if args.merge_only:
        merge_shards(output_dir, args.repo_id, args.num_workers)
        return

    # Find total episodes by finding metadata.json
    metadata_files = sorted(list(dataset_dir.rglob("metadata.json")))
    episode_dirs = [f.parent for f in metadata_files]

    if args.max_episodes is not None:
        episode_dirs = episode_dirs[: args.max_episodes]
    
    logger.info(f"Total Valid Episodes Found: {len(episode_dirs)}")

    if args.num_workers == 1 and args.shard_id is None:
        convert_shard(episode_dirs, output_dir, args.repo_id, args.fps, args.vcodec, args.image_writer_threads, 0, 1)
        return

    num_workers = args.num_workers
    shard_size = (len(episode_dirs) + num_workers - 1) // num_workers
    shards = [episode_dirs[i * shard_size : (i + 1) * shard_size] for i in range(num_workers)]

    if args.shard_id is not None:
        sid = args.shard_id
        if sid >= len(shards):
            return
        convert_shard(shards[sid], output_dir, args.repo_id, args.fps, args.vcodec, args.image_writer_threads, sid, num_workers)
        return

    logger.info(f"Launching {num_workers} worker subprocesses...")
    processes = []
    for sid in range(len(shards)):
        if not shards[sid]:
            continue
        cmd = [
            sys.executable, __file__,
            "--dataset-dir", str(dataset_dir),
            "--output-dir", str(output_dir),
            "--repo-id", args.repo_id,
            "--fps", str(args.fps),
            "--vcodec", args.vcodec,
            "--num-workers", str(num_workers),
            "--shard-id", str(sid),
        ]
        log_file = output_dir / f"shard_{sid:03d}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        f_log = open(log_file, "w")
        p = subprocess.Popen(cmd, stdout=f_log, stderr=subprocess.STDOUT)
        processes.append((p, f_log, sid))

    start = time.time()
    failed = []
    for p, f_log, sid in processes:
        p.wait()
        f_log.close()
        if p.returncode != 0:
            failed.append(sid)

    if failed:
        logger.error(f"Failed shards: {failed}.")
    else:
        logger.info(f"All workers finished. Merging...")
        merge_shards(output_dir, args.repo_id, num_workers)

if __name__ == "__main__":
    main()
