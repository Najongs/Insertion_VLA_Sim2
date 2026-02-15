#!/usr/bin/env python
"""
Evaluate a trained NORA model on a single episode.

This script loads a NORA (Qwen2.5-VL + FAST tokenizer) checkpoint and evaluates
it frame-by-frame on an HDF5 episode, comparing predicted actions to ground truth.

Uses manual FAST token decoding with Mean/Std unnormalization matching the
approach in run_nora_inference_sim.py.

Usage:
python3 evaluate_episode_nora.py \
    --checkpoint /path/to/checkpoint_step_5000.pt \
    --config /path/to/train_config_nora.yaml \
    --stats /path/to/dataset_stats.yaml \
    --episode /path/to/episode.h5 \
    --output_dir /path/to/output
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
import torch
from PIL import Image
import io

# Add Nora to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
NORA_DIR = os.path.join(PROJECT_ROOT, "TRAIN/Nora")
sys.path.append(NORA_DIR)

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FAST tokenizer action token range
ACTION_TOKEN_MIN = 151665
ACTION_TOKEN_MAX = 153712


def load_config(path):
    """Load YAML config file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


class NoraEvaluator:
    """NORA evaluator using Qwen2.5-VL + manual FAST token decoding."""

    def __init__(self, config_path, stats_path, checkpoint_path, device):
        self.config = load_config(config_path)
        self.device = device

        logger.info(f"Initializing NORA Evaluator (Device: {device})...")

        # 1. Load processor
        base_model = "declare-lab/nora"
        logger.info(f"Loading processor from: {base_model}")
        self.processor = AutoProcessor.from_pretrained(base_model)
        self.processor.tokenizer.padding_side = 'left'

        # 2. Load model
        logger.info(f"Loading model from: {base_model}")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        )

        # 3. Load checkpoint weights
        if checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

            if "trainable_state_dict" in checkpoint:
                # New format: only trainable params (lm_head / embed_tokens)
                state_dict = checkpoint["trainable_state_dict"]
                missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
                logger.info(f"  Loaded trainable params: {len(state_dict)} keys, missing={len(missing)}, unexpected={len(unexpected)}")
            elif "model_state_dict" in checkpoint:
                # Old format: full model
                state_dict = checkpoint["model_state_dict"]
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                self.model.load_state_dict(state_dict, strict=True)
                logger.info(f"  Loaded full model state dict")
            else:
                logger.warning(f"  Unknown checkpoint format. Keys: {list(checkpoint.keys())}")

            logger.info(f"  Checkpoint step: {checkpoint.get('step', 'unknown')}")
        else:
            logger.warning("No checkpoint found. Using base model weights.")

        self.model.to(device)
        self.model.eval()

        # 4. Load normalization stats (only if training used normalization)
        self.action_mean = None
        self.action_std = None
        dataset_cfg = self.config.get("dataset", {})
        train_norm_path = dataset_cfg.get("normalization_stats_path", None)

        if train_norm_path is not None:
            # Training used normalization → eval must unnormalize
            if os.path.exists(stats_path):
                stats = load_config(stats_path)
                self.action_mean = np.array(stats['action']['mean'])
                self.action_std = np.array(stats['action']['std'])
                logger.info(f"Normalization stats loaded from: {stats_path}")
                logger.info(f"  Action mean: {self.action_mean}")
                logger.info(f"  Action std: {self.action_std}")
            else:
                logger.warning(f"Training used normalization but stats file not found: {stats_path}")
        else:
            logger.info("Training did not use normalization → skipping unnormalization in eval")

        # 5. Get task instruction from config
        dataset_cfg = self.config.get("dataset", {})
        self.instruction = dataset_cfg.get("task_instruction",
            self.config.get("task_instruction", "Insert the needle into the target point"))

        logger.info(f"  Task instruction: {self.instruction}")

    @torch.inference_mode()
    def step(self, image):
        """
        Predict action from a single camera image using NORA.

        Args:
            image: PIL Image or numpy array [H, W, 3] RGB (camera1)

        Returns:
            numpy array [6] (predicted action)
        """
        # 1. Convert to PIL if needed
        if isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            pil_img = image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # 2. Build single-image message (matching training format)
        content = [
            {
                "type": "image",
                "image": pil_img,
                "resized_height": 224,
                "resized_width": 224,
            },
            {"type": "text", "text": self.instruction},
        ]

        messages = [{"role": "user", "content": content}]

        # 3. Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 4. Generate
        generated_ids = self.model.generate(
            **inputs,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            max_new_tokens=20,
        )

        # 5. Extract action tokens from generated output
        prompt_len = inputs['input_ids'].shape[1]
        new_tokens = generated_ids[0][prompt_len:]

        mask = (ACTION_TOKEN_MIN <= new_tokens) & (new_tokens <= ACTION_TOKEN_MAX)
        action_indices = torch.where(mask)[0]

        if len(action_indices) == 0:
            logger.warning("No action tokens found in generated output!")
            return np.zeros(6)

        raw_token_ids = new_tokens[action_indices]

        # Take last 7 tokens (or pad if fewer)
        action_dim = 7
        if len(raw_token_ids) >= action_dim:
            target_tokens = raw_token_ids[-action_dim:]
        else:
            padding = torch.full(
                (action_dim - len(raw_token_ids),),
                ACTION_TOKEN_MIN,
                device=raw_token_ids.device
            )
            target_tokens = torch.cat([raw_token_ids, padding])

        # 6. Manual FAST token decoding: token -> [-1, 1]
        total_bins = ACTION_TOKEN_MAX - ACTION_TOKEN_MIN  # 2047
        offsets = (target_tokens - ACTION_TOKEN_MIN).float()
        output_action = (offsets / total_bins) * 2.0 - 1.0
        output_action = output_action.cpu().numpy()

        # 7. Unnormalize: [-1, 1] -> original scale using Mean/Std
        pose_action = output_action[:6]
        if self.action_mean is not None and self.action_std is not None:
            unnorm_action = pose_action * self.action_std + self.action_mean
        else:
            unnorm_action = pose_action

        return unnorm_action


def load_episode_data(h5_path: str):
    """Load all data from HDF5 episode file."""
    logger.info(f"Loading episode from: {h5_path}")

    # Only load camera1 (side_camera) matching training
    camera_mapping = {
        'camera1': 'side_camera',
    }

    with h5py.File(h5_path, 'r') as f:
        images = {}
        for cam_key, h5_cam_name in camera_mapping.items():
            cam_data = f['observations']['images'][h5_cam_name][:]
            images[cam_key] = cam_data
            logger.info(f"Loaded {cam_key} ({h5_cam_name}): {len(cam_data)} frames")

        actions = f['action'][:]
        ee_pose = f['observations']['ee_pose'][:]

        timestamps = f['timestamp'][:] if 'timestamp' in f else np.arange(len(actions))

        num_frames = len(actions)
        logger.info(f"Loaded {num_frames} frames")
        logger.info(f"Action shape: {actions.shape}")
        logger.info(f"EE pose shape: {ee_pose.shape}")

    return {
        'images': images,
        'actions': actions,
        'ee_pose': ee_pose,
        'timestamps': timestamps,
        'num_frames': num_frames,
    }


def decode_image(img_bytes):
    """Decode JPEG bytes to PIL Image."""
    return Image.open(io.BytesIO(img_bytes))


def evaluate_episode(evaluator, episode_data):
    """Evaluate policy on entire episode, frame by frame."""
    num_frames = episode_data['num_frames']

    results = {
        'frame_idx': [],
        'gt_actions': [],
        'pred_actions': [],
        'action_errors': [],
        'position_errors': [],
        'rotation_errors': [],
        'ee_poses': [],
        'pred_trajectory': [],
        'trajectory_errors': [],
    }

    logger.info(f"Evaluating {num_frames} frames...")

    pred_position = None

    for frame_idx in range(num_frames):
        if frame_idx % 50 == 0:
            logger.info(f"Processing frame {frame_idx}/{num_frames}")

        try:
            gt_action = episode_data['actions'][frame_idx]

            # Decode camera1 image only (matching training)
            img_bytes = episode_data['images']['camera1'][frame_idx]
            camera1_img = decode_image(img_bytes)

            state_6d = episode_data['ee_pose'][frame_idx]

            # Predict action (camera1 only, NORA is vision-only)
            pred_action = evaluator.step(camera1_img)

            # Handle dimension mismatch
            min_dim = min(len(pred_action), len(gt_action))
            pred_action_cmp = pred_action[:min_dim]
            gt_action_cmp = gt_action[:min_dim]

            # Compute errors
            action_error = np.mean((pred_action_cmp - gt_action_cmp) ** 2)
            position_error = np.mean((pred_action_cmp[:3] - gt_action_cmp[:3]) ** 2)
            rotation_error = np.mean((pred_action_cmp[3:6] - gt_action_cmp[3:6]) ** 2)

            # Trajectory error
            if pred_position is None:
                pred_position = state_6d[:3].copy()
            else:
                pred_position = pred_position + pred_action[:3]

            gt_position = state_6d[:3]
            trajectory_error = np.sqrt(np.sum((gt_position - pred_position) ** 2))

            # Store results
            results['frame_idx'].append(frame_idx)
            results['gt_actions'].append(gt_action)
            results['pred_actions'].append(pred_action)
            results['action_errors'].append(action_error)
            results['position_errors'].append(position_error)
            results['rotation_errors'].append(rotation_error)
            results['ee_poses'].append(state_6d)
            results['pred_trajectory'].append(pred_position.copy())
            results['trajectory_errors'].append(trajectory_error)

        except Exception as e:
            logger.warning(f"Failed to process frame {frame_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    logger.info("Evaluation complete!")

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

        min_dim = min(len(gt_action), len(pred_action))
        dim_names = ['dx', 'dy', 'dz', 'drx', 'dry', 'drz', 'gripper']

        for d in range(min_dim):
            dim_name = dim_names[d] if d < len(dim_names) else f'dim{d}'
            row[f'GT_{dim_name}'] = gt_action[d]
            row[f'Pred_{dim_name}'] = pred_action[d] if d < len(pred_action) else 0.0
            row[f'Error_{dim_name}'] = (pred_action[d] - gt_action[d]) ** 2 if d < len(pred_action) else 0.0

        data.append(row)

    return pd.DataFrame(data)


def plot_results(results, output_dir: Path):
    """Create visualization plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = results['frame_idx']

    # Plot 1: Error over time
    num_plots = 4 if 'trajectory_errors' in results else 3
    fig, axes = plt.subplots(num_plots, 1, figsize=(14, 3*num_plots+1))
    fig.suptitle('NORA Prediction Errors Over Episode', fontsize=16)

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
    fig.suptitle(f'NORA: Ground Truth vs Predicted Actions (First {max_frames_to_plot} Frames)', fontsize=16)

    for i in range(min_dim):
        row = i // 2
        col = i % 2

        axes[row, col].plot(frames_subset, gt_actions[:, i], label='Ground Truth', linewidth=2, alpha=0.7)
        axes[row, col].plot(frames_subset, pred_actions[:, i], label='Predicted', linewidth=2, alpha=0.7, linestyle='--')
        axes[row, col].set_xlabel('Frame')
        axes[row, col].set_ylabel(dim_names[i])
        axes[row, col].set_title(f'{dim_names[i]} Comparison')
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'action_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 3: 3D Trajectory
    if 'ee_poses' in results and len(results['ee_poses']) > 0:
        logger.info("Creating 3D trajectory plot...")

        gt_poses = np.array(results['ee_poses'])
        gt_x, gt_y, gt_z = gt_poses[:, 0], gt_poses[:, 1], gt_poses[:, 2]

        pred_trajectory = np.array(results['pred_trajectory'])
        pred_x, pred_y, pred_z = pred_trajectory[:, 0], pred_trajectory[:, 1], pred_trajectory[:, 2]

        fig = plt.figure(figsize=(16, 12))

        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        ax1.plot(gt_x, gt_y, gt_z, 'b-', linewidth=2, alpha=0.7, label='Ground Truth')
        ax1.plot(pred_x, pred_y, pred_z, 'r--', linewidth=2, alpha=0.7, label='Predicted')
        ax1.scatter(gt_x[0], gt_y[0], gt_z[0], c='green', s=100, marker='o', label='Start')
        ax1.scatter(gt_x[-1], gt_y[-1], gt_z[-1], c='red', s=100, marker='x', label='End')
        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Y (mm)')
        ax1.set_zlabel('Z (mm)')
        ax1.set_title('3D Trajectory - Full View')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(gt_x, gt_y, 'b-', linewidth=2, alpha=0.7, label='Ground Truth')
        ax2.plot(pred_x, pred_y, 'r--', linewidth=2, alpha=0.7, label='Predicted')
        ax2.scatter(gt_x[0], gt_y[0], c='green', s=100, marker='o', label='Start')
        ax2.scatter(gt_x[-1], gt_y[-1], c='red', s=100, marker='x', label='End')
        ax2.set_xlabel('X (mm)')
        ax2.set_ylabel('Y (mm)')
        ax2.set_title('Top View (XY Plane)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')

        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(gt_x, gt_z, 'b-', linewidth=2, alpha=0.7, label='Ground Truth')
        ax3.plot(pred_x, pred_z, 'r--', linewidth=2, alpha=0.7, label='Predicted')
        ax3.scatter(gt_x[0], gt_z[0], c='green', s=100, marker='o', label='Start')
        ax3.scatter(gt_x[-1], gt_z[-1], c='red', s=100, marker='x', label='End')
        ax3.set_xlabel('X (mm)')
        ax3.set_ylabel('Z (mm)')
        ax3.set_title('Side View (XZ Plane)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.axis('equal')

        ax4 = fig.add_subplot(2, 2, 4)
        ax4.plot(gt_y, gt_z, 'b-', linewidth=2, alpha=0.7, label='Ground Truth')
        ax4.plot(pred_y, pred_z, 'r--', linewidth=2, alpha=0.7, label='Predicted')
        ax4.scatter(gt_y[0], gt_z[0], c='green', s=100, marker='o', label='Start')
        ax4.scatter(gt_y[-1], gt_z[-1], c='red', s=100, marker='x', label='End')
        ax4.set_xlabel('Y (mm)')
        ax4.set_ylabel('Z (mm)')
        ax4.set_title('Front View (YZ Plane)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axis('equal')

        plt.suptitle('NORA: End-Effector Trajectory Comparison (GT vs Predicted)', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_dir / 'trajectory_3d.png', dpi=150, bbox_inches='tight')
        plt.close()

    logger.info(f"Plots saved to {output_dir}")


def print_summary(results):
    """Print summary statistics."""
    action_errors = np.array(results['action_errors'])
    position_errors = np.array(results['position_errors'])
    rotation_errors = np.array(results['rotation_errors'])
    trajectory_errors = np.array(results['trajectory_errors']) if 'trajectory_errors' in results else None

    print("\n" + "="*80)
    print("NORA EPISODE EVALUATION SUMMARY")
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
    parser = argparse.ArgumentParser(description="Evaluate NORA model on episode")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to NORA checkpoint (.pt)")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to NORA training config YAML file")
    parser.add_argument("--episode", type=str, required=True,
                        help="Path to HDF5 episode file")
    parser.add_argument("--stats", type=str, required=True,
                        help="Path to dataset statistics YAML file")
    parser.add_argument("--output_dir", type=str, default="outputs/nora_eval",
                        help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cpu/cuda)")

    args = parser.parse_args()

    # Setup
    checkpoint_path = Path(args.checkpoint)
    config_path = Path(args.config)
    episode_path = Path(args.episode)
    stats_path = Path(args.stats)
    output_dir = Path(args.output_dir)

    for name, path in [("Checkpoint", checkpoint_path), ("Config", config_path),
                        ("Episode", episode_path), ("Stats", stats_path)]:
        if not path.exists():
            logger.error(f"{name} not found: {path}")
            sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Initialize evaluator
    logger.info("Initializing NORA evaluator...")
    evaluator = NoraEvaluator(
        config_path=str(config_path),
        stats_path=str(stats_path),
        checkpoint_path=str(checkpoint_path),
        device=device
    )

    # Load episode data
    episode_data = load_episode_data(str(episode_path))

    # Evaluate
    results = evaluate_episode(evaluator=evaluator, episode_data=episode_data)

    # Print summary
    print_summary(results)

    # Save results
    logger.info("Creating detailed comparison DataFrame...")
    df = create_comparison_dataframe(results)

    csv_path = output_dir / 'frame_by_frame_comparison.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"Detailed comparison saved to: {csv_path}")

    # Save summary
    summary = {
        'model': 'NORA',
        'num_frames': len(results['frame_idx']),
        'action_mse_mean': float(np.mean(results['action_errors'])),
        'action_mse_std': float(np.std(results['action_errors'])),
        'position_mse_mean': float(np.mean(results['position_errors'])),
        'position_mse_std': float(np.std(results['position_errors'])),
        'rotation_mse_mean': float(np.mean(results['rotation_errors'])),
        'rotation_mse_std': float(np.std(results['rotation_errors'])),
    }

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
