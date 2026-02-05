#!/usr/bin/env python
"""
Evaluate a trained SmolVLA model on a single episode with proper normalization.

This script applies MEAN_STD normalization to states and actions, matching the
training normalization approach from LeRobot.

Updated to match current inference method from run_smolvla_inference_sim.py.

Usage:
python3 /home/najo/NAS/VLA/Insertion_VLA_Sim2/Eval/evaluate_episode_normalized.py \
    --checkpoint /home/najo/NAS/VLA/Insertion_VLA_Sim2/TRAIN/SmolVLA/outputs/train/smolvla/checkpoints/checkpoint_step_10000.pt \
    --config /home/najo/NAS/VLA/Insertion_VLA_Sim2/TRAIN/SmolVLA/train_config_smolvla_sim.yaml \
    --stats /home/najo/NAS/VLA/Insertion_VLA_Sim2/TRAIN/SmolVLA/dataset_stats.yaml \
    --episode /home/najo/NAS/VLA/Insertion_VLA_Sim2/Dataset/Insert_never_random/worker0_episode_*.h5 \
    --output_dir /home/najo/NAS/VLA/Insertion_VLA_Sim2/Eval/outputs/evaluation
"""

import argparse
import logging
import sys
import os
from pathlib import Path
import yaml
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Optional
import torch
from PIL import Image
import io
import cv2

# Add SmolVLA to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SMOLVLA_DIR = os.path.join(PROJECT_ROOT, "TRAIN/SmolVLA")
sys.path.append(os.path.join(SMOLVLA_DIR, "lerobot/src"))
sys.path.append(SMOLVLA_DIR)

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK, OBS_STATE
from normalization_utils import Normalizer, load_stats
from transformers import AutoTokenizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(path):
    """Load YAML config file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


class SmolVLAEvaluator:
    """SmolVLA evaluator matching current inference method."""

    def __init__(self, config_path, stats_path, checkpoint_path, device):
        self.config = load_config(config_path)
        self.policy_cfg = self.config["policy"]
        self.device = device

        logger.info(f"🚀 Initializing SmolVLA Policy (Device: {device})...")

        # 1. Setup SmolVLA Config
        smolvla_config = SmolVLAConfig(
            vlm_model_name=self.policy_cfg.get("vlm_model_name", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"),
            n_obs_steps=self.policy_cfg.get("n_obs_steps", 1),
            chunk_size=self.policy_cfg.get("chunk_size", 50),
            n_action_steps=self.policy_cfg.get("n_action_steps", 50),
            max_state_dim=self.policy_cfg.get("max_state_dim", 32),
            max_action_dim=self.policy_cfg.get("max_action_dim", 32),
            num_steps=self.policy_cfg.get("num_steps", 10),
            input_features={
                # "observation.images.camera1": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 512, 512)),
                "observation.images.camera2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 512, 512)),
                # "observation.images.camera3": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 512, 512)),
                "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
            },
            output_features={
                "action": PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
            },
        )

        # 2. Create Policy
        self.policy = SmolVLAPolicy(smolvla_config)

        # 3. Load Checkpoint
        if checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f"📂 Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            state_dict = checkpoint.get('policy_state_dict', checkpoint.get('model_state_dict', checkpoint))
            # Remove module. prefix if present
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.policy.load_state_dict(state_dict)
            logger.info(f"  Checkpoint step: {checkpoint.get('step', 'unknown')}")
        else:
            logger.warning("⚠️ No checkpoint found. Using UNTRAINED model.")

        self.policy.to(device)
        self.policy.eval()
        self.policy.reset()

        # 4. Setup Tokenizer & Normalizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.policy_cfg["vlm_model_name"])
        self.normalizer = None
        if os.path.exists(stats_path):
            stats = load_stats(stats_path)
            self.normalizer = Normalizer(stats).to(device)
            logger.info(f"📊 Normalization stats loaded from: {stats_path}")

        # 5. Prepare language tokens
        self.instruction = self.config["dataset"]["task_instruction"]
        tokens = self.tokenizer(
            self.instruction,
            return_tensors="pt",
            padding="max_length",
            max_length=self.policy_cfg.get("tokenizer_max_length", 48),
            truncation=True
        )
        self.lang_tokens = tokens["input_ids"].to(device)
        self.lang_mask = tokens["attention_mask"].to(device)

        logger.info(f"  Task instruction: {self.instruction}")
        logger.info(f"  n_obs_steps: {self.policy.config.n_obs_steps}")
        logger.info(f"  chunk_size: {self.policy.config.chunk_size}")
        logger.info(f"  n_action_steps: {self.policy.config.n_action_steps}")

    @torch.no_grad()
    def step(self, images, state_6d, sensor_val):
        """
        Predict action from observation.

        Args:
            images: list of 3 PIL Images or numpy arrays [H, W, 3] RGB
            state_6d: numpy array [6] (ee_pose)
            sensor_val: float (0.0 or 1.0)

        Returns:
            numpy array [6] (predicted action)
        """
        batch = {}

        # 1. Images: [1, 3, 512, 512] and range [0, 1]
        for i, img in enumerate(images):
            if isinstance(img, np.ndarray):
                # Convert BGR to RGB if needed (assuming input is RGB)
                img_rgb = img if img.shape[2] == 3 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            elif isinstance(img, Image.Image):
                img_array = np.array(img)
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
            else:
                raise TypeError(f"Unsupported image type: {type(img)}")

            batch[f"observation.images.camera{i+1}"] = img_tensor.unsqueeze(0).to(self.device)

        # 2. State (Robot State Normalized + Sensor Appended)
        state_tensor = torch.from_numpy(state_6d).float().unsqueeze(0).to(self.device)

        # Apply normalization only to robot state part (first 6 dims)
        if self.normalizer:
            state_tensor = self.normalizer.normalize(state_tensor, "observation.state")

        # Append Sensor Data (0 or 1)
        sensor_tensor = torch.tensor([[sensor_val]], dtype=torch.float32).to(self.device)
        state_tensor = torch.cat([state_tensor, sensor_tensor], dim=1)  # (1, 7)

        batch[OBS_STATE] = state_tensor

        # 3. Language
        batch[OBS_LANGUAGE_TOKENS] = self.lang_tokens
        batch[OBS_LANGUAGE_ATTENTION_MASK] = self.lang_mask

        # 4. Predict Action
        norm_action = self.policy.select_action(batch)

        # 5. Unnormalize Action
        if self.normalizer:
            unnorm_action = self.normalizer.unnormalize(norm_action.unsqueeze(0), "action").squeeze(0)
        else:
            unnorm_action = norm_action

        # Ensure 1D array and take first 6 elements (pose)
        if unnorm_action.ndim > 1:
            unnorm_action = unnorm_action.flatten()

        return unnorm_action[:6].cpu().numpy()

def load_episode_data(h5_path: str):
    """Load all data from HDF5 episode file."""
    logger.info(f"Loading episode from: {h5_path}")

    # Camera name mapping: policy expects camera1/2/3, but H5 files use descriptive names
    camera_mapping = {
        'camera1': 'side_camera',
        'camera2': 'tool_camera',
        'camera3': 'top_camera',
    }

    with h5py.File(h5_path, 'r') as f:
        # Load images (all frames, all cameras)
        images = {}
        for cam_key, h5_cam_name in camera_mapping.items():
            cam_data = f['observations']['images'][h5_cam_name][:]
            images[cam_key] = cam_data

        # Load actions
        actions = f['action'][:]

        # Load state (ee_pose)
        ee_pose = f['observations']['ee_pose'][:]

        # Load sensor data if available (otherwise set to 0)
        if 'sensor' in f['observations']:
            sensor_data = f['observations']['sensor'][:]
            logger.info("Sensor data found in episode")
        else:
            sensor_data = np.zeros(len(ee_pose), dtype=np.float32)
            logger.info("No sensor data found, using zeros")

        # Load timestamps if available
        timestamps = f['timestamp'][:] if 'timestamp' in f else np.arange(len(actions))

        num_frames = len(actions)
        logger.info(f"Loaded {num_frames} frames")
        logger.info(f"Action shape: {actions.shape}")
        logger.info(f"EE pose shape: {ee_pose.shape}")
        logger.info(f"Sensor data shape: {sensor_data.shape}")

    return {
        'images': images,
        'actions': actions,
        'ee_pose': ee_pose,
        'sensor_data': sensor_data,
        'timestamps': timestamps,
        'num_frames': num_frames,
    }


def decode_image(img_bytes):
    """Decode JPEG bytes to PIL Image."""
    return Image.open(io.BytesIO(img_bytes))


def evaluate_episode(evaluator, episode_data):
    """
    Evaluate policy on entire episode, frame by frame.

    Args:
        evaluator: SmolVLAEvaluator instance
        episode_data: dict containing episode data

    Returns:
        dict with evaluation results
    """
    num_frames = episode_data['num_frames']
    action_dim = episode_data['actions'].shape[1]

    # Storage for results
    results = {
        'frame_idx': [],
        'gt_actions': [],
        'pred_actions': [],
        'action_errors': [],
        'position_errors': [],
        'rotation_errors': [],
        'ee_poses': [],  # Store GT EE poses for trajectory
        'pred_trajectory': [],  # Store predicted trajectory (accumulated)
        'trajectory_errors': [],  # Store trajectory error at each step
    }

    logger.info(f"Evaluating {num_frames} frames...")

    # Initialize predicted trajectory from first GT position
    pred_position = None

    for frame_idx in range(num_frames):
        if frame_idx % 50 == 0:
            logger.info(f"Processing frame {frame_idx}/{num_frames}")

        try:
            # Get ground truth action
            gt_action = episode_data['actions'][frame_idx]

            # Decode images from JPEG bytes
            images = []
            for cam_key in ['camera1', 'camera2', 'camera3']:
                img_bytes = episode_data['images'][cam_key][frame_idx]
                img = decode_image(img_bytes)
                images.append(img)

            # Get state and sensor
            state_6d = episode_data['ee_pose'][frame_idx]
            sensor_val = episode_data['sensor_data'][frame_idx]

            # Predict action using evaluator
            pred_action = evaluator.step(images, state_6d, sensor_val)

            # Handle dimension mismatch
            min_dim = min(len(pred_action), len(gt_action))
            pred_action_cmp = pred_action[:min_dim]
            gt_action_cmp = gt_action[:min_dim]

            # Compute action errors (delta errors)
            action_error = np.mean((pred_action_cmp - gt_action_cmp) ** 2)
            position_error = np.mean((pred_action_cmp[:3] - gt_action_cmp[:3]) ** 2)
            rotation_error = np.mean((pred_action_cmp[3:6] - gt_action_cmp[3:6]) ** 2)

            # Compute trajectory error (accumulated position error)
            if pred_position is None:
                # Initialize from first GT position
                pred_position = state_6d[:3].copy()
            else:
                # Accumulate predicted action (delta)
                pred_position = pred_position + pred_action[:3]

            # Calculate trajectory error (Euclidean distance from GT position)
            gt_position = state_6d[:3]
            trajectory_error = np.sqrt(np.sum((gt_position - pred_position) ** 2))

            # Store results
            results['frame_idx'].append(frame_idx)
            results['gt_actions'].append(gt_action)
            results['pred_actions'].append(pred_action)
            results['action_errors'].append(action_error)
            results['position_errors'].append(position_error)
            results['rotation_errors'].append(rotation_error)
            results['ee_poses'].append(state_6d)  # Store GT EE pose
            results['pred_trajectory'].append(pred_position.copy())  # Store predicted position
            results['trajectory_errors'].append(trajectory_error)  # Store trajectory error

        except Exception as e:
            logger.warning(f"Failed to process frame {frame_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    logger.info("Evaluation complete!")

    # Log trajectory statistics
    if len(results['trajectory_errors']) > 0:
        traj_errors = np.array(results['trajectory_errors'])
        logger.info(f"Trajectory Error Statistics:")
        logger.info(f"  Mean: {traj_errors.mean():.4f} mm")
        logger.info(f"  Std: {traj_errors.std():.4f} mm")
        logger.info(f"  Max: {traj_errors.max():.4f} mm")
        logger.info(f"  Final: {traj_errors[-1]:.4f} mm")

    return results


def create_comparison_dataframe(results):
    """Create pandas DataFrame with detailed comparison."""
    data = []

    for i in range(len(results['frame_idx'])):
        frame_idx = results['frame_idx'][i]
        gt_action = results['gt_actions'][i]
        pred_action = results['pred_actions'][i]

        row = {
            'Frame': frame_idx,
            'Action_Error': results['action_errors'][i],
            'Position_Error': results['position_errors'][i],
            'Rotation_Error': results['rotation_errors'][i],
        }

        # Add per-dimension values
        min_dim = min(len(gt_action), len(pred_action))
        dim_names = ['dx', 'dy', 'dz', 'drx', 'dry', 'drz', 'gripper']

        for d in range(min_dim):
            dim_name = dim_names[d] if d < len(dim_names) else f'dim{d}'
            row[f'GT_{dim_name}'] = gt_action[d]
            row[f'Pred_{dim_name}'] = pred_action[d] if d < len(pred_action) else 0.0
            row[f'Error_{dim_name}'] = (pred_action[d] - gt_action[d]) ** 2 if d < len(pred_action) else 0.0

        data.append(row)

    df = pd.DataFrame(data)
    return df


def plot_results(results, output_dir: Path):
    """Create visualization plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = results['frame_idx']

    # Plot 1: Error over time
    num_plots = 4 if 'trajectory_errors' in results else 3
    fig, axes = plt.subplots(num_plots, 1, figsize=(14, 3*num_plots+1))
    fig.suptitle('Prediction Errors Over Episode (With Normalization)', fontsize=16)

    axes[0].plot(frames, results['action_errors'], linewidth=1.5, color='blue')
    axes[0].set_ylabel('Action MSE')
    axes[0].set_title('Overall Action Error (Delta)')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(frames, results['position_errors'], linewidth=1.5, color='green')
    axes[1].set_ylabel('Position MSE')
    axes[1].set_title('Position Error (dx, dy, dz - Delta)')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(frames, results['rotation_errors'], linewidth=1.5, color='red')
    axes[2].set_ylabel('Rotation MSE')
    axes[2].set_title('Rotation Error (drx, dry, drz - Delta)')
    axes[2].grid(True, alpha=0.3)

    if 'trajectory_errors' in results:
        axes[3].plot(frames, results['trajectory_errors'], linewidth=1.5, color='purple')
        axes[3].set_xlabel('Frame')
        axes[3].set_ylabel('Trajectory Error (mm)')
        axes[3].set_title('Accumulated Position Error (Euclidean Distance)')
        axes[3].grid(True, alpha=0.3)
    else:
        axes[2].set_xlabel('Frame')

    plt.tight_layout()
    plt.savefig(output_dir / 'error_over_time.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Per-dimension comparison
    max_frames_to_plot = min(300, len(frames))
    gt_actions = np.array(results['gt_actions'][:max_frames_to_plot])
    pred_actions = np.array(results['pred_actions'][:max_frames_to_plot])
    frames_subset = frames[:max_frames_to_plot]

    dim_names = ['dx', 'dy', 'dz', 'drx', 'dry', 'drz']
    min_dim = min(gt_actions.shape[1], pred_actions.shape[1], 6)

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(f'Ground Truth vs Predicted Actions (First {max_frames_to_plot} Frames, Normalized)', fontsize=16)

    for i in range(min_dim):
        row = i // 2
        col = i % 2

        axes[row, col].plot(frames_subset, gt_actions[:, i], label='Ground Truth', linewidth=2, alpha=0.7)
        axes[row, col].plot(frames_subset, pred_actions[:, i], label='Predicted', linewidth=2, alpha=0.7, linestyle='--')
        axes[row, col].set_xlabel('Frame')
        axes[row, col].set_ylabel(dim_names[i])
        axes[row, col].set_title(f'{dim_names[i]} Comparison')
        # axes[row, col].set_ylim(-1, 1)
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'action_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 3: 3D Trajectory (GT vs Predicted)
    if 'ee_poses' in results and len(results['ee_poses']) > 0:
        logger.info("Creating 3D trajectory plot...")

        # Get ground truth EE poses (actual positions)
        gt_poses = np.array(results['ee_poses'])
        gt_x = gt_poses[:, 0]
        gt_y = gt_poses[:, 1]
        gt_z = gt_poses[:, 2]

        # Get predicted trajectory (already computed in evaluate_episode)
        pred_trajectory = np.array(results['pred_trajectory'])
        pred_x = pred_trajectory[:, 0]
        pred_y = pred_trajectory[:, 1]
        pred_z = pred_trajectory[:, 2]

        # Create 3D plot
        fig = plt.figure(figsize=(16, 12))

        # Plot 1: Full trajectory
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        ax1.plot(gt_x, gt_y, gt_z, 'b-', linewidth=2, alpha=0.7, label='Ground Truth')
        ax1.plot(pred_x, pred_y, pred_z, 'r--', linewidth=2, alpha=0.7, label='Predicted')
        ax1.scatter(gt_x[0], gt_y[0], gt_z[0], c='green', s=100, marker='o', label='Start')
        ax1.scatter(gt_x[-1], gt_y[-1], gt_z[-1], c='red', s=100, marker='x', label='End')
        ax1.set_xlabel('X (mm)', fontsize=10)
        ax1.set_ylabel('Y (mm)', fontsize=10)
        ax1.set_zlabel('Z (mm)', fontsize=10)
        ax1.set_title('3D Trajectory - Full View', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Top view (XY plane)
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(gt_x, gt_y, 'b-', linewidth=2, alpha=0.7, label='Ground Truth')
        ax2.plot(pred_x, pred_y, 'r--', linewidth=2, alpha=0.7, label='Predicted')
        ax2.scatter(gt_x[0], gt_y[0], c='green', s=100, marker='o', label='Start')
        ax2.scatter(gt_x[-1], gt_y[-1], c='red', s=100, marker='x', label='End')
        ax2.set_xlabel('X (mm)', fontsize=10)
        ax2.set_ylabel('Y (mm)', fontsize=10)
        ax2.set_title('Top View (XY Plane)', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')

        # Plot 3: Side view (XZ plane)
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(gt_x, gt_z, 'b-', linewidth=2, alpha=0.7, label='Ground Truth')
        ax3.plot(pred_x, pred_z, 'r--', linewidth=2, alpha=0.7, label='Predicted')
        ax3.scatter(gt_x[0], gt_z[0], c='green', s=100, marker='o', label='Start')
        ax3.scatter(gt_x[-1], gt_z[-1], c='red', s=100, marker='x', label='End')
        ax3.set_xlabel('X (mm)', fontsize=10)
        ax3.set_ylabel('Z (mm)', fontsize=10)
        ax3.set_title('Side View (XZ Plane)', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.axis('equal')

        # Plot 4: Front view (YZ plane)
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.plot(gt_y, gt_z, 'b-', linewidth=2, alpha=0.7, label='Ground Truth')
        ax4.plot(pred_y, pred_z, 'r--', linewidth=2, alpha=0.7, label='Predicted')
        ax4.scatter(gt_y[0], gt_z[0], c='green', s=100, marker='o', label='Start')
        ax4.scatter(gt_y[-1], gt_z[-1], c='red', s=100, marker='x', label='End')
        ax4.set_xlabel('Y (mm)', fontsize=10)
        ax4.set_ylabel('Z (mm)', fontsize=10)
        ax4.set_title('Front View (YZ Plane)', fontsize=12)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axis('equal')

        plt.suptitle('End-Effector Trajectory Comparison (GT vs Predicted)', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_dir / 'trajectory_3d.png', dpi=150, bbox_inches='tight')
        plt.close()

        # Use precomputed trajectory errors
        if 'trajectory_errors' in results and len(results['trajectory_errors']) > 0:
            trajectory_errors = np.array(results['trajectory_errors'])
            logger.info(f"3D Trajectory Statistics:")
            logger.info(f"  Mean position error: {trajectory_errors.mean():.4f} mm")
            logger.info(f"  Max position error: {trajectory_errors.max():.4f} mm")
            logger.info(f"  Final position error: {trajectory_errors[-1]:.4f} mm")

    logger.info(f"Plots saved to {output_dir}")


def print_summary(results):
    """Print summary statistics."""
    action_errors = np.array(results['action_errors'])
    position_errors = np.array(results['position_errors'])
    rotation_errors = np.array(results['rotation_errors'])
    trajectory_errors = np.array(results['trajectory_errors']) if 'trajectory_errors' in results else None

    print("\n" + "="*80)
    print("EPISODE EVALUATION SUMMARY (WITH NORMALIZATION)")
    print("="*80)
    print(f"Total frames evaluated: {len(results['frame_idx'])}")
    print()
    print("Action MSE (Delta Error):")
    print(f"  Mean: {action_errors.mean():.6f}")
    print(f"  Std:  {action_errors.std():.6f}")
    print(f"  Min:  {action_errors.min():.6f}")
    print(f"  Max:  {action_errors.max():.6f}")
    print()
    print("Position MSE (dx, dy, dz - Delta Error):")
    print(f"  Mean: {position_errors.mean():.6f}")
    print(f"  Std:  {position_errors.std():.6f}")
    print(f"  Min:  {position_errors.min():.6f}")
    print(f"  Max:  {position_errors.max():.6f}")
    print()
    print("Rotation MSE (drx, dry, drz - Delta Error):")
    print(f"  Mean: {rotation_errors.mean():.6f}")
    print(f"  Std:  {rotation_errors.std():.6f}")
    print(f"  Min:  {rotation_errors.min():.6f}")
    print(f"  Max:  {rotation_errors.max():.6f}")
    print()
    if trajectory_errors is not None and len(trajectory_errors) > 0:
        print("Trajectory Error (Accumulated Position Error in mm):")
        print(f"  Mean:  {trajectory_errors.mean():.4f}")
        print(f"  Std:   {trajectory_errors.std():.4f}")
        print(f"  Min:   {trajectory_errors.min():.4f}")
        print(f"  Max:   {trajectory_errors.max():.4f}")
        print(f"  Final: {trajectory_errors[-1]:.4f}")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate episode with normalization")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt)"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config YAML file"
    )

    parser.add_argument(
        "--episode",
        type=str,
        required=True,
        help="Path to HDF5 episode file"
    )

    parser.add_argument(
        "--stats",
        type=str,
        required=True,
        help="Path to dataset statistics YAML file"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/episode_eval_normalized",
        help="Output directory for results"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cpu/cuda)"
    )

    args = parser.parse_args()

    # Setup
    checkpoint_path = Path(args.checkpoint)
    config_path = Path(args.config)
    episode_path = Path(args.episode)
    stats_path = Path(args.stats)
    output_dir = Path(args.output_dir)

    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)

    if not episode_path.exists():
        logger.error(f"Episode not found: {episode_path}")
        sys.exit(1)

    if not stats_path.exists():
        logger.error(f"Statistics file not found: {stats_path}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Initialize evaluator
    logger.info("Initializing SmolVLA evaluator...")
    evaluator = SmolVLAEvaluator(
        config_path=str(config_path),
        stats_path=str(stats_path),
        checkpoint_path=str(checkpoint_path),
        device=device
    )

    # Load episode data
    episode_data = load_episode_data(str(episode_path))

    # Evaluate episode
    results = evaluate_episode(
        evaluator=evaluator,
        episode_data=episode_data
    )

    # Print summary
    print_summary(results)

    # Create DataFrame
    logger.info("Creating detailed comparison DataFrame...")
    df = create_comparison_dataframe(results)

    # Save DataFrame
    csv_path = output_dir / 'frame_by_frame_comparison.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"Detailed comparison saved to: {csv_path}")

    # Save summary statistics
    summary = {
        'num_frames': len(results['frame_idx']),
        'action_mse_mean': float(np.mean(results['action_errors'])),
        'action_mse_std': float(np.std(results['action_errors'])),
        'position_mse_mean': float(np.mean(results['position_errors'])),
        'position_mse_std': float(np.std(results['position_errors'])),
        'rotation_mse_mean': float(np.mean(results['rotation_errors'])),
        'rotation_mse_std': float(np.std(results['rotation_errors'])),
    }

    # Add trajectory error statistics if available
    if 'trajectory_errors' in results and len(results['trajectory_errors']) > 0:
        trajectory_errors = np.array(results['trajectory_errors'])
        summary['trajectory_error_mean'] = float(trajectory_errors.mean())
        summary['trajectory_error_std'] = float(trajectory_errors.std())
        summary['trajectory_error_min'] = float(trajectory_errors.min())
        summary['trajectory_error_max'] = float(trajectory_errors.max())
        summary['trajectory_error_final'] = float(trajectory_errors[-1])

    summary_path = output_dir / 'summary.yaml'
    with open(summary_path, 'w') as f:
        yaml.dump(summary, f, default_flow_style=False)
    logger.info(f"Summary saved to: {summary_path}")

    # Create plots
    logger.info("Creating visualization plots...")
    try:
        plot_results(results, output_dir)
    except Exception as e:
        logger.warning(f"Failed to create plots: {e}")

    logger.info(f"\nAll results saved to: {output_dir}")
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
