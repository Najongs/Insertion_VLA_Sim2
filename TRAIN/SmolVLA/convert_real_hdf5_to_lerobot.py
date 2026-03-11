#!/usr/bin/env python
"""
Convert REAL HDF5 VLA episodes to LeRobot v3.0 dataset format.
Supports multiprocessing by splitting episodes across workers.

Dataset structure:
- HDF5 source: /data/public/NAS/Insertion_VLA_Sim2/Dataset/dataset/New_dataset/collected_data/Eye_trocar/Eye_trocar/
  - action: (N, 6) delta pose
  - observations/ee_pose: (N, 6)
  - observations/qpos: (N, 6)
  - observations/images/{camera1, camera2, camera3}: (N, 480, 640, 3) uint8 arrays
  - timestamp: (N,)

- LeRobot target:
  - observation.state: (6,) ee_pose
  - observation.images.{camera1,camera2,camera3}: (480, 640, 3) video
  - action: (6,) delta pose

Usage (single process):
    python convert_real_hdf5_to_lerobot.py --num-workers 1

Usage (multiprocess - 8 workers):
    python convert_real_hdf5_to_lerobot.py --num-workers 8

Usage (run only shard 0 of 8):
    python convert_real_hdf5_to_lerobot.py --num-workers 8 --shard-id 0

Merge shards after all complete:
    python convert_real_hdf5_to_lerobot.py --merge-only --num-workers 8
"""

import argparse
import logging
import subprocess
import sys
import time
import traceback
from pathlib import Path

import h5py
import numpy as np
import cv2
from PIL import Image

# Add lerobot to path
sys.path.insert(0, str(Path(__file__).parent / "lerobot" / "src"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Feature schema ────────────────────────────────────────────────────────────

FEATURES = {
    "observation.state": {
        "dtype": "float32",
        "shape": (6,),
        "names": {"axes": ["x", "y", "z", "rx", "ry", "rz"]},
    },
    "observation.images.camera1": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.images.camera2": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.images.camera3": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channels"],
    },
    "action": {
        "dtype": "float32",
        "shape": (6,),
        "names": {"axes": ["dx", "dy", "dz", "drx", "dry", "drz"]},
    },
}

TASK_DESCRIPTION = "Insert the needle through the eye phantom trocar opening"
CAMERA_KEYS = ["camera1", "camera2", "camera3"]

def decode_jpeg(raw_data) -> np.ndarray:
    """Decode JPEG bytes to RGB numpy array (H, W, 3) uint8."""
    if isinstance(raw_data, np.ndarray):
        jpeg_bytes = raw_data.flatten().astype(np.uint8)
    elif isinstance(raw_data, (bytes, bytearray)):
        jpeg_bytes = np.frombuffer(raw_data, dtype=np.uint8)
    else:
        jpeg_bytes = np.array(raw_data, dtype=np.uint8).flatten()

    if not jpeg_bytes.flags["C_CONTIGUOUS"]:
        jpeg_bytes = np.ascontiguousarray(jpeg_bytes)

    img_bgr = cv2.imdecode(jpeg_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Failed to decode JPEG (size={jpeg_bytes.size})")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def detect_jpeg_format(h5f: h5py.File) -> bool:
    """Detect if images are JPEG-encoded objects or raw arrays."""
    cam_names = sorted(h5f["observations"]["images"].keys())
    sample = h5f["observations"]["images"][cam_names[0]][0]
    if isinstance(sample, np.ndarray):
        if len(sample.shape) == 3 and sample.shape[2] == 3:
            return False
        if len(sample.shape) == 1 and sample.dtype == np.uint8:
            return True
    return True

# ─── Single shard conversion ──────────────────────────────────────────────────

def convert_shard(
    h5_files: list[Path],
    output_dir: Path,
    repo_id: str,
    fps: int,
    vcodec: str,
    image_writer_threads: int,
    shard_id: int,
    total_shards: int,
):
    """Convert a shard (subset) of HDF5 episodes to LeRobot format."""
    shard_repo_id = f"{repo_id}_shard{shard_id:03d}" if total_shards > 1 else repo_id
    shard_output = output_dir / shard_repo_id if total_shards > 1 else output_dir

    total_episodes = len(h5_files)
    logger.info(f"[Shard {shard_id}/{total_shards}] Converting {total_episodes} episodes → {shard_output}")

    dataset = LeRobotDataset.create(
        repo_id=shard_repo_id,
        fps=fps,
        robot_type="surgical_robot_6dof",
        features=FEATURES,
        root=str(shard_output),
        use_videos=True,
        image_writer_threads=image_writer_threads,
        batch_encoding_size=1,
        vcodec=vcodec,
    )

    start_time = time.time()
    total_frames = 0
    skipped = 0

    for ep_idx, h5_path in enumerate(h5_files):
        try:
            with h5py.File(str(h5_path), "r") as h5f:
                num_frames = h5f["action"].shape[0]
                is_jpeg = detect_jpeg_format(h5f)

                # Action and state arrays
                actions = h5f["action"][:].astype(np.float32)
                ee_poses = h5f["observations"]["ee_pose"][:].astype(np.float32)

                # Optional: Retrieve phase if newly implemented, defaulting to generic string
                phases = h5f["phase"][:] if "phase" in h5f else None

                for frame_idx in range(num_frames):
                    state = ee_poses[frame_idx]
                    action = actions[frame_idx]

                    # Static task for Eye_trocar if not phased
                    task = TASK_DESCRIPTION
                    
                    frame = {
                        "observation.state": state,
                        "action": action,
                        "task": task,
                    }

                    for cam_key in CAMERA_KEYS:
                        raw_img = h5f["observations"]["images"][cam_key][frame_idx]
                        if is_jpeg:
                            img_rgb = decode_jpeg(raw_img)
                        else:
                            # Real datasets load natively structured as np.arrays (uint8)
                            img_rgb = raw_img

                        frame[f"observation.images.{cam_key}"] = Image.fromarray(
                            img_rgb.astype(np.uint8)
                        )

                    dataset.add_frame(frame)

                dataset.save_episode()
                total_frames += num_frames

        except Exception as e:
            logger.error(f"[Shard {shard_id}] Failed ep {ep_idx} ({h5_path.name}): {e}")
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

        if (ep_idx + 1) % 100 == 0 or (ep_idx + 1) == total_episodes:
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
    logger.info(
        f"[Shard {shard_id}] Done! {total_episodes - skipped}/{total_episodes} episodes, "
        f"{total_frames:,} frames in {elapsed / 3600:.1f}h"
    )
    return shard_output

# ─── Merge shards ─────────────────────────────────────────────────────────────

def merge_shards(output_dir: Path, repo_id: str, num_shards: int):
    """Merge multiple shard datasets into one using LeRobot's aggregate."""
    from lerobot.datasets.aggregate import aggregate_datasets

    shard_dirs = []
    roots = []
    for i in range(num_shards):
        shard_repo = f"{repo_id}_shard{i:03d}"
        shard_path = output_dir / shard_repo
        if shard_path.exists():
            shard_dirs.append(shard_repo)
            roots.append(shard_path)
        else:
            logger.warning(f"Shard {i} not found at {shard_path}, skipping")

    if len(shard_dirs) == 0:
        logger.error("No shards found to merge!")
        return

    aggr_path = output_dir / repo_id
    if aggr_path.exists():
        import shutil
        logger.warning(f"Removing existing merged dataset at {aggr_path}")
        shutil.rmtree(aggr_path)

    logger.info(f"Merging {len(shard_dirs)} shards into {aggr_path}...")
    aggregate_datasets(
        repo_ids=shard_dirs,
        aggr_repo_id=repo_id,
        roots=roots,
        aggr_root=aggr_path,
    )
    logger.info(f"Merged dataset saved to: {aggr_path}")

# ─── Main orchestrator ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convert REAL HDF5 VLA episodes to LeRobot format. Traverses directory recursively.")
    parser.add_argument("--hdf5-dir", type=str, default="/data/public/NAS/Insertion_VLA_Sim2/Dataset/dataset/New_dataset/collected_data/Eye_trocar/Eye_trocar")
    parser.add_argument("--output-dir", type=str, default="/data/public/NAS/Insertion_VLA_Sim2/TRAIN/SmolVLA/lerobot_real_dataset")
    parser.add_argument("--repo-id", type=str, default="eye_trocar_real")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--vcodec", type=str, default="h264")
    parser.add_argument("--image-writer-threads", type=int, default=4)

    # Multiprocessing
    parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel workers (each converts a shard)")
    parser.add_argument("--shard-id", type=int, default=None, help="Run only this shard (for manual parallel execution)")
    parser.add_argument("--merge-only", action="store_true", help="Only merge existing shards, skip conversion")
    parser.add_argument("--no-merge", action="store_true", help="Skip merge step (convert shards only)")

    args = parser.parse_args()

    hdf5_dir = Path(args.hdf5_dir)
    output_dir = Path(args.output_dir)

    # Discover episodes recursively using rglob
    h5_files = sorted(list(hdf5_dir.rglob("*.h5")))
    if args.max_episodes is not None:
        h5_files = h5_files[: args.max_episodes]
    logger.info(f"Total episodes found recursively: {len(h5_files)}")

    # ─── Merge only mode ──────────────────────────────────────────────────
    if args.merge_only:
        merge_shards(output_dir, args.repo_id, args.num_workers)
        return

    # ─── Single worker mode ───────────────────────────────────────────────
    if args.num_workers == 1 and args.shard_id is None:
        convert_shard(
            h5_files=h5_files,
            output_dir=output_dir,
            repo_id=args.repo_id,
            fps=args.fps,
            vcodec=args.vcodec,
            image_writer_threads=args.image_writer_threads,
            shard_id=0,
            total_shards=1,
        )
        return

    # ─── Split into shards ────────────────────────────────────────────────
    num_workers = args.num_workers
    shard_size = (len(h5_files) + num_workers - 1) // num_workers
    shards = []
    for i in range(num_workers):
        start = i * shard_size
        end = min(start + shard_size, len(h5_files))
        if start < len(h5_files):
            shards.append(h5_files[start:end])

    logger.info(f"Split {len(h5_files)} episodes into {len(shards)} shards (~{shard_size} episodes each)")

    # ─── Run single shard ─────────────────────────────────────────────────
    if args.shard_id is not None:
        sid = args.shard_id
        if sid >= len(shards):
            logger.error(f"Shard {sid} out of range (max {len(shards) - 1})")
            return
        convert_shard(
            h5_files=shards[sid],
            output_dir=output_dir,
            repo_id=args.repo_id,
            fps=args.fps,
            vcodec=args.vcodec,
            image_writer_threads=args.image_writer_threads,
            shard_id=sid,
            total_shards=num_workers,
        )
        return

    # ─── Launch all shards as subprocesses ────────────────────────────────
    logger.info(f"Launching {num_workers} worker subprocesses...")
    processes = []
    for sid in range(len(shards)):
        cmd = [
            sys.executable, __file__,
            "--hdf5-dir", str(hdf5_dir),
            "--output-dir", str(output_dir),
            "--repo-id", args.repo_id,
            "--fps", str(args.fps),
            "--vcodec", args.vcodec,
            "--image-writer-threads", str(max(1, args.image_writer_threads // num_workers)),
            "--num-workers", str(num_workers),
            "--shard-id", str(sid),
        ]
        if args.max_episodes is not None:
            cmd += ["--max-episodes", str(args.max_episodes)]

        log_file = output_dir / f"shard_{sid:03d}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"  Shard {sid}: {len(shards[sid])} episodes → log: {log_file}")
        f_log = open(log_file, "w")
        p = subprocess.Popen(cmd, stdout=f_log, stderr=subprocess.STDOUT)
        processes.append((p, f_log, sid))

    # Wait for all
    logger.info("Waiting for all workers to finish...")
    start = time.time()
    failed = []
    for p, f_log, sid in processes:
        p.wait()
        f_log.close()
        if p.returncode != 0:
            logger.error(f"Shard {sid} failed with return code {p.returncode}")
            failed.append(sid)
        else:
            logger.info(f"Shard {sid} completed successfully")

    elapsed = time.time() - start
    logger.info(f"All workers finished in {elapsed / 3600:.1f}h ({len(failed)} failed)")

    if failed:
        logger.error(f"Failed shards: {failed}. Fix and re-run with --shard-id <id>")
        logger.error("Then merge with: --merge-only")
        return

    # ─── Merge shards ─────────────────────────────────────────────────────
    if not args.no_merge:
        merge_shards(output_dir, args.repo_id, num_workers)

if __name__ == "__main__":
    main()
