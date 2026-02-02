#!/usr/bin/env python
"""
Compute and save dataset normalization statistics.

This script computes mean/std statistics from the entire dataset
for use in normalized training and evaluation.

Usage:
python compute_dataset_stats.py \
    --dataset_root /home/najo/NAS/VLA/Insertion_VLA_Sim2/Dataset \
    --output dataset_stats.yaml \
    --max_episodes 1000
"""

import argparse
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from normalization_utils import compute_dataset_stats, save_stats


def main():
    parser = argparse.ArgumentParser(description="Compute dataset normalization statistics")

    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Path to dataset root directory"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="dataset_stats.yaml",
        help="Output YAML file path"
    )

    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Maximum number of episodes to process (None = all)"
    )

    parser.add_argument(
        "--use_ee_pose_delta",
        action="store_true",
        help="Compute action stats from EE pose delta instead of recorded action"
    )

    args = parser.parse_args()

    print("="*80)
    print("COMPUTING DATASET STATISTICS")
    print("="*80)
    print(f"Dataset root: {args.dataset_root}")
    print(f"Max episodes: {args.max_episodes if args.max_episodes else 'All'}")
    print(f"Output file:  {args.output}")
    print(f"Use EE Pose Delta: {args.use_ee_pose_delta}")
    print("="*80 + "\n")

    # Compute statistics
    stats = compute_dataset_stats(
        dataset_root=args.dataset_root,
        max_episodes=args.max_episodes,
        use_ee_pose_delta=args.use_ee_pose_delta
    )

    # Save statistics
    save_stats(stats, args.output)

    print("\n" + "="*80)
    print("DONE!")
    print("="*80)
    print(f"Statistics saved to: {args.output}")
    print("\nYou can now use this file for:")
    print("  1. Training with normalization")
    print("  2. Evaluation with normalization")
    print("="*80)


if __name__ == "__main__":
    main()
