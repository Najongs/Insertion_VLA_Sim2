#!/usr/bin/env python3
"""
Evaluate a LeRobot pretrained_model checkpoint on dataset episodes (no gym env needed).

Loads the model + preprocessor/postprocessor from a pretrained_model directory,
reads episodes from the LeRobot dataset, runs frame-by-frame inference,
and compares predicted actions against ground truth.

Usage:
python eval_pretrained_on_dataset.py \
    --pretrained_path /data/public/NAS/Insertion_VLA_Sim2/TRAIN/SmolVLA/outputs/train/smolvla_lerobot_dataset/checkpoints/150000/pretrained_model \
    --dataset_path /data/public/NAS/Insertion_VLA_Sim2/TRAIN/SmolVLA/lerobot_dataset/insertion_vla_sim2 \
    --episode_index 30299 \
    --device cuda \
    --output_dir outputs/eval_dataset
"""

import argparse
import json
import sys
import os
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add lerobot to path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "lerobot" / "src"))

from lerobot.policies.factory import make_policy, make_pre_post_processors, make_policy_config
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


def load_episode_via_lerobot_api(dataset_path: str, episode_index: int):
    """Load episode using LeRobot's dataset API for proper video decoding."""
    dataset = LeRobotDataset(
        repo_id="insertion_vla_sim2",
        root=dataset_path,
        episodes=[episode_index],
    )

    num_frames = dataset.num_frames
    print(f"Episode {episode_index}: {num_frames} frames")

    # Collect all frames
    all_data = []
    for idx in range(num_frames):
        item = dataset[idx]
        all_data.append(item)

    return all_data, num_frames


def run_evaluation(
    pretrained_path: str,
    dataset_path: str,
    episode_index: int,
    device: str = "cuda",
    output_dir: str = "outputs/eval_dataset",
    max_frames: int = 0,
):
    """Run frame-by-frame evaluation on a dataset episode."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pretrained_path = Path(pretrained_path)

    # 1. Load model
    print("=" * 70)
    print("Loading model from:", pretrained_path)
    print("=" * 70)

    # Load config from pretrained_model to get the policy type
    with open(pretrained_path / "config.json") as f:
        model_config = json.load(f)
    policy_type = model_config["type"]

    # Create policy config with pretrained_path set
    policy_cfg = make_policy_config(
        policy_type,
        pretrained_path=str(pretrained_path),
        device=device,
        use_amp=False,
    )

    # Load dataset metadata for feature info
    ds_meta = LeRobotDatasetMetadata(
        repo_id="insertion_vla_sim2",
        root=dataset_path,
    )

    policy = make_policy(cfg=policy_cfg, ds_meta=ds_meta)
    policy.eval()

    preprocessor_overrides = {
        "device_processor": {"device": device},
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(pretrained_path),
        preprocessor_overrides=preprocessor_overrides,
    )

    print(f"  Policy type: {type(policy).__name__}")
    print(f"  Device: {device}")

    # 2. Load episode data via LeRobot dataset API
    print("\n" + "=" * 70)
    print(f"Loading episode {episode_index} from dataset")
    print("=" * 70)

    all_frames_data, num_frames = load_episode_via_lerobot_api(dataset_path, episode_index)

    if max_frames > 0:
        num_frames = min(num_frames, max_frames)
        all_frames_data = all_frames_data[:num_frames]

    print(f"  Total frames to evaluate: {num_frames}")

    # Print sample data info
    sample = all_frames_data[0]
    print(f"  Sample keys: {list(sample.keys())}")
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"    {k}: shape={v.shape}, dtype={v.dtype}")
        elif isinstance(v, str):
            print(f"    {k}: '{v[:80]}'")

    # 3. Run frame-by-frame inference
    print("\n" + "=" * 70)
    print("Running inference...")
    print("=" * 70)

    gt_actions = []
    pred_actions = []
    states = []

    policy.reset()

    for frame_idx in range(num_frames):
        if frame_idx % 50 == 0:
            print(f"  Frame {frame_idx}/{num_frames}")

        item = all_frames_data[frame_idx]

        # Ground truth action
        gt_action = item["action"].numpy()
        gt_actions.append(gt_action)

        # State
        state = item["observation.state"].numpy()
        states.append(state)

        # Build observation dict (exclude action, metadata fields)
        obs = {}
        for k, v in item.items():
            if k in ("action", "frame_index", "episode_index", "index",
                      "timestamp", "task_index", "next.done", "next.reward",
                      "next.success"):
                continue
            if isinstance(v, torch.Tensor):
                obs[k] = v
            elif isinstance(v, str) and k == "task":
                obs[k] = v

        # Apply preprocessor (handles batching, tokenization, normalization, device)
        obs_processed = preprocessor(obs)

        # Inference
        with torch.inference_mode():
            action_pred = policy.select_action(obs_processed)

        # Apply postprocessor (unnormalize, move to cpu)
        action_pred = postprocessor(action_pred)

        # Convert to numpy
        action_np = action_pred.squeeze().cpu().numpy()
        pred_actions.append(action_np)

    gt_actions = np.array(gt_actions)
    pred_actions = np.array(pred_actions)
    states = np.array(states)

    # 4. Compute metrics
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    # Per-frame MSE
    frame_mse = np.mean((gt_actions - pred_actions) ** 2, axis=1)
    position_mse = np.mean((gt_actions[:, :3] - pred_actions[:, :3]) ** 2, axis=1)
    rotation_mse = np.mean((gt_actions[:, 3:6] - pred_actions[:, 3:6]) ** 2, axis=1)

    # Overall metrics
    print(f"\nEpisode {episode_index}: {num_frames} frames")
    print(f"\nOverall Action MSE:    {frame_mse.mean():.6f} (std: {frame_mse.std():.6f})")
    print(f"Position MSE (dx,dy,dz): {position_mse.mean():.6f} (std: {position_mse.std():.6f})")
    print(f"Rotation MSE (drx,dry,drz): {rotation_mse.mean():.6f} (std: {rotation_mse.std():.6f})")

    # Per-dimension error
    dim_names = ["dx", "dy", "dz", "drx", "dry", "drz"]
    print("\nPer-dimension MSE:")
    for i, name in enumerate(dim_names):
        if i < gt_actions.shape[1]:
            dim_mse = np.mean((gt_actions[:, i] - pred_actions[:, i]) ** 2)
            print(f"  {name}: {dim_mse:.6f}")

    # Trajectory error (accumulated position)
    pred_trajectory = [states[0, :3].copy()]
    for i in range(1, num_frames):
        pred_pos = pred_trajectory[-1] + pred_actions[i - 1, :3]
        pred_trajectory.append(pred_pos)
    pred_trajectory = np.array(pred_trajectory)
    gt_trajectory = states[:, :3]
    traj_errors = np.sqrt(np.sum((gt_trajectory - pred_trajectory) ** 2, axis=1))

    print(f"\nTrajectory Error (Euclidean, accumulated):")
    print(f"  Mean:  {traj_errors.mean():.4f}")
    print(f"  Std:   {traj_errors.std():.4f}")
    print(f"  Max:   {traj_errors.max():.4f}")
    print(f"  Final: {traj_errors[-1]:.4f}")

    # 5. Save results
    # CSV
    import csv

    csv_path = output_dir / "frame_by_frame_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["Frame", "Action_MSE", "Position_MSE", "Rotation_MSE"]
        for name in dim_names:
            header.extend([f"GT_{name}", f"Pred_{name}", f"Error_{name}"])
        writer.writerow(header)

        for i in range(num_frames):
            row = [i, frame_mse[i], position_mse[i], rotation_mse[i]]
            for d in range(min(len(dim_names), gt_actions.shape[1])):
                row.extend([
                    gt_actions[i, d],
                    pred_actions[i, d],
                    (gt_actions[i, d] - pred_actions[i, d]) ** 2,
                ])
            writer.writerow(row)
    print(f"\nCSV saved: {csv_path}")

    # Summary YAML
    summary = {
        "episode_index": int(episode_index),
        "num_frames": int(num_frames),
        "pretrained_path": str(pretrained_path),
        "action_mse_mean": float(frame_mse.mean()),
        "action_mse_std": float(frame_mse.std()),
        "position_mse_mean": float(position_mse.mean()),
        "position_mse_std": float(position_mse.std()),
        "rotation_mse_mean": float(rotation_mse.mean()),
        "rotation_mse_std": float(rotation_mse.std()),
        "trajectory_error_mean": float(traj_errors.mean()),
        "trajectory_error_std": float(traj_errors.std()),
        "trajectory_error_max": float(traj_errors.max()),
        "trajectory_error_final": float(traj_errors[-1]),
    }
    summary_path = output_dir / "summary.yaml"
    with open(summary_path, "w") as f:
        yaml.dump(summary, f, default_flow_style=False)
    print(f"Summary saved: {summary_path}")

    # 6. Plots
    print("\nGenerating plots...")

    # Plot 1: Error over time
    fig, axes = plt.subplots(4, 1, figsize=(14, 14))
    fig.suptitle(f"Evaluation: Episode {episode_index} (pretrained_model checkpoint)", fontsize=16)

    axes[0].plot(frame_mse, linewidth=1, color="blue")
    axes[0].set_ylabel("Action MSE")
    axes[0].set_title("Overall Action Error")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(position_mse, linewidth=1, color="green")
    axes[1].set_ylabel("Position MSE")
    axes[1].set_title("Position Error (dx, dy, dz)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(rotation_mse, linewidth=1, color="red")
    axes[2].set_ylabel("Rotation MSE")
    axes[2].set_title("Rotation Error (drx, dry, drz)")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(traj_errors, linewidth=1, color="purple")
    axes[3].set_xlabel("Frame")
    axes[3].set_ylabel("Trajectory Error")
    axes[3].set_title("Accumulated Trajectory Error (Euclidean)")
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "error_over_time.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 2: Per-dimension GT vs Pred
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(f"GT vs Predicted Actions - Episode {episode_index}", fontsize=16)

    for i in range(min(6, gt_actions.shape[1])):
        row, col = i // 2, i % 2
        axes[row, col].plot(gt_actions[:, i], label="Ground Truth", linewidth=1.5, alpha=0.7)
        axes[row, col].plot(pred_actions[:, i], label="Predicted", linewidth=1.5, alpha=0.7, linestyle="--")
        axes[row, col].set_xlabel("Frame")
        axes[row, col].set_ylabel(dim_names[i])
        axes[row, col].set_title(f"{dim_names[i]} Comparison")
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "action_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 3: 3D Trajectory
    fig = plt.figure(figsize=(16, 12))

    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax1.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], gt_trajectory[:, 2],
             "b-", linewidth=2, alpha=0.7, label="Ground Truth")
    ax1.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], pred_trajectory[:, 2],
             "r--", linewidth=2, alpha=0.7, label="Predicted")
    ax1.scatter(*gt_trajectory[0], c="green", s=100, marker="o", label="Start")
    ax1.scatter(*gt_trajectory[-1], c="red", s=100, marker="x", label="End")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title("3D Trajectory")
    ax1.legend()

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], "b-", linewidth=2, alpha=0.7, label="GT")
    ax2.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], "r--", linewidth=2, alpha=0.7, label="Pred")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_title("Top View (XY)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis("equal")

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(gt_trajectory[:, 0], gt_trajectory[:, 2], "b-", linewidth=2, alpha=0.7, label="GT")
    ax3.plot(pred_trajectory[:, 0], pred_trajectory[:, 2], "r--", linewidth=2, alpha=0.7, label="Pred")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Z")
    ax3.set_title("Side View (XZ)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axis("equal")

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(gt_trajectory[:, 1], gt_trajectory[:, 2], "b-", linewidth=2, alpha=0.7, label="GT")
    ax4.plot(pred_trajectory[:, 1], pred_trajectory[:, 2], "r--", linewidth=2, alpha=0.7, label="Pred")
    ax4.set_xlabel("Y")
    ax4.set_ylabel("Z")
    ax4.set_title("Front View (YZ)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axis("equal")

    plt.suptitle(f"Trajectory Comparison - Episode {episode_index}", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / "trajectory_3d.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nPlots saved to: {output_dir}")
    print("=" * 70)
    print("Evaluation complete!")
    print("=" * 70)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate pretrained model on dataset episode")
    parser.add_argument("--pretrained_path", type=str, required=True,
                        help="Path to pretrained_model directory")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to LeRobot dataset root")
    parser.add_argument("--episode_index", type=int, default=-1,
                        help="Episode index to evaluate (-1 for last)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="outputs/eval_dataset")
    parser.add_argument("--max_frames", type=int, default=0,
                        help="Max frames to evaluate (0 = all)")

    args = parser.parse_args()

    # Resolve last episode if -1
    episode_index = args.episode_index
    if episode_index == -1:
        info_path = Path(args.dataset_path) / "meta" / "info.json"
        with open(info_path) as f:
            info = json.load(f)
        episode_index = info["total_episodes"] - 1
        print(f"Using last episode: {episode_index}")

    run_evaluation(
        pretrained_path=args.pretrained_path,
        dataset_path=args.dataset_path,
        episode_index=episode_index,
        device=args.device,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
