#!/usr/bin/env python
"""
Evaluate a trained SmolVLA model on a single episode using the official LeRobot evaluation pipeline APIs.

This script uses `PreTrainedPolicy.from_pretrained` or `make_policy` logic to instantiate
the policy and preprocessors exactly as they were trained, eliminating hardcoded normalization
logic and model configuration instantiation.

Usage:
python3 /data/public/NAS/Insertion_VLA_Sim2/Eval/evaluate_episode_lerobot.py \
    --pretrained_model /data/public/NAS/Insertion_VLA_Sim2/TRAIN/SmolVLA/outputs/train/smolvla_lerobot_dataset/checkpoints/100000/pretrained_model \
    --episode /data/public/NAS/Insertion_VLA_Sim2/Dataset/all_h5/episode_20260213_213912.h5 \
    --output_dir /data/public/NAS/Insertion_VLA_Sim2/Eval/outputs/evaluation
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
from typing import Dict, List, Optional, Any
import torch
from PIL import Image
import io
import cv2
from transformers import AutoTokenizer

# Add SmolVLA to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SMOLVLA_DIR = os.path.join(PROJECT_ROOT, "TRAIN/SmolVLA")
sys.path.append(os.path.join(SMOLVLA_DIR, "lerobot/src"))
sys.path.append(SMOLVLA_DIR)

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.utils import get_safe_torch_device
from lerobot.processor.core import TransitionKey

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SmolVLAEvaluator:
    """SmolVLA evaluator using official LeRobot PreTrained loading."""

    def __init__(self, pretrained_path: str, device: torch.device):
        self.pretrained_path = pretrained_path
        self.device = device

        logger.info(f"🚀 Initializing SmolVLA Policy from {pretrained_path} (Device: {device})...")

        # Load Policy and Config using LeRobot
        self.policy = SmolVLAPolicy.from_pretrained(pretrained_path)
        self.policy.to(self.device)
        self.policy.eval()
        self.policy.reset()

        # Load Pre/Post-processors automatically constructed from train_config / datasets
        # Note: overriding device processor to ensure everything lands correctly on eval device.
        preprocessor_overrides = {
            "device_processor": {"device": str(self.device)},
        }
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=pretrained_path,
            preprocessor_overrides=preprocessor_overrides,
        )

        # Load Tokenizer for language tokens
        vlm_model_name = self.policy.config.vlm_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(vlm_model_name)
        
        # NOTE: Using default task instruction. You can parse this from config if needed.
        self.instruction = "Insert the peg into the socket."
        tokens = self.tokenizer(
            self.instruction,
            return_tensors="pt",
            padding="max_length",
            max_length=self.policy.config.tokenizer_max_length,
            truncation=True
        )
        self.lang_tokens = tokens["input_ids"].to(self.device)
        self.lang_mask = tokens["attention_mask"].to(self.device)

        logger.info(f"  Task instruction: {self.instruction}")

        logger.info(f"  n_obs_steps: {self.policy.config.n_obs_steps}")
        logger.info(f"  chunk_size: {self.policy.config.chunk_size}")
        logger.info(f"  n_action_steps: {self.policy.config.n_action_steps}")
        
    @torch.no_grad()
    def step(self, images: Dict[str, np.ndarray], state_6d: np.ndarray, sensor_val: float):
        """
        Predict action from observation.

        Args:
            images: dictionary of PIL Images or numpy arrays mapping camera names to their visual data.
            state_6d: numpy array [6] (ee_pose)
            sensor_val: float (0.0 or 1.0) # Used for appending if needed or not.

        Returns:
            numpy array [6] (predicted action)
        """
        observation = {}

        # Format Images
        # Note: the dataset uses descriptive names (side_camera, tool_camera, top_camera),
        # Assuming the policy was trained with `observation.images.side_camera` etc. as indicated by `train_config.json`.
        for cam_name, img in images.items():
            obs_key = f"observation.images.{cam_name}"
            
            
            # NOTE: the preprocessor assumes inputs are in natively expected scale.
            # Usually cv2 outputs [H, W, C] in [0, 255]. LeRobot expects them as torch tensor [C, H, W] normalized to [0, 1].
            # Ensure images are torch tensors of shape [batch, channels, height, width] and [0,1] floating point.
            if isinstance(img, np.ndarray):
                img_rgb = img if img.shape[-1] == 3 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            elif isinstance(img, Image.Image):
                img_array = np.array(img)
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
            else:
                raise TypeError(f"Unsupported image type: {type(img)}")
            
            # Add batch dimension required for preprocessor (since we are processing frame by frame)
            observation[obs_key] = img_tensor.unsqueeze(0).to(self.device)

        # 2. State (Robot State + Sensor Appended if model requires it)
        state_key = "observation.state"
        state_dim = self.policy.config.input_features[state_key].shape[0]

        if state_dim == 7:
            # Append Sensor Data (0 or 1)
            state_tensor = torch.tensor(np.append(state_6d, sensor_val), dtype=torch.float32).to(self.device)
        else:
            state_tensor = torch.from_numpy(state_6d).float().to(self.device)

        # Add batch dimension
        observation[state_key] = state_tensor.unsqueeze(0)

        # 3. Add task instruction for Tokenizer Processor
        # LeRobot natively extracts the lang task string from the observation dictionary itself 
        # using the 'task' key natively prior to applying the tokenization inside the pipeline.
        observation["task"] = [self.instruction]

        # 4. Apply standard LeRobot preprocessors
        # This will handle `MEAN_STD` normalization, `[512,512]` padding automatically, 
        # AND tokenization since we feed the raw text.
        
        print(f"DEBUG: Before prep: {observation.keys()}")
        observation = self.preprocessor(observation)
        print(f"DEBUG: After prep: {observation.keys()}")
        
        # 5. Execute Policy
        out_action = self.policy.select_action(observation)
        
        # 6. Apply postprocessors (Un-normalizes actions)
        # Note: LeRobot's `select_action` returns a tensor directly.
        # `postprocessor` is configured with `to_transition=policy_action_to_transition` natively, 
        # so it expects the raw `PolicyAction` and returns a `PolicyAction` (`torch.Tensor`).
        out_action = self.postprocessor(out_action)

        # Squeeze batch dimension, transfer to numpy, ensure 1D array
        unnorm_action = out_action.squeeze(0).cpu().numpy()
        if unnorm_action.ndim > 1:
            unnorm_action = unnorm_action.flatten()

        return unnorm_action[:6]


def load_episode_data(h5_path: str, expected_cameras: List[str]):
    """Load all data from HDF5 episode file based on expected cameras from config."""
    logger.info(f"Loading episode from: {h5_path}")

    with h5py.File(h5_path, 'r') as f:
        # Load images (all available cameras in the HDF5 group)
        images = {}
        if 'images' in f['observations']:
            for cam_name in f['observations']['images']:
                images[cam_name] = f['observations']['images'][cam_name][:]
        else:
            logger.warning("No 'images' group found in HDF5.")

        # Load actions
        actions = f['action'][:]

        # Load state (ee_pose)
        ee_pose = f['observations']['ee_pose'][:]

        # Load sensor data if available (otherwise set to 0)
        if 'sensor' in f['observations']:
            sensor_node = f['observations']['sensor']
            if isinstance(sensor_node, h5py.Group):
                # Try specific subgroup datasets
                if 'aline' in sensor_node:
                    sensor_data = sensor_node['aline'][:]
                    logger.info("Sensor data ('aline') found in episode")
                elif 'force' in sensor_node:
                    sensor_data = sensor_node['force'][:]
                    logger.info("Sensor data ('force') found in episode")
                else:
                    sensor_data = np.zeros(len(ee_pose), dtype=np.float32)
                    logger.info("Sensor group found but no expected fields, using zeros")
            else:
                sensor_data = sensor_node[:]
                logger.info("Sensor array data found in episode")
        else:
            sensor_data = np.zeros(len(ee_pose), dtype=np.float32)
            logger.info("No sensor data found, using zeros")

        num_frames = len(actions)
        logger.info(f"Loaded {num_frames} frames")

    return {
        'images': images,
        'actions': actions,
        'ee_pose': ee_pose,
        'sensor_data': sensor_data,
        'num_frames': num_frames,
    }


def decode_image(img_bytes):
    """Decode JPEG bytes to PIL Image."""
    return Image.open(io.BytesIO(img_bytes))


def evaluate_episode(evaluator: SmolVLAEvaluator, episode_data: Dict[str, Any]):
    """
    Evaluate policy on entire episode, frame by frame.
    """
    num_frames = episode_data['num_frames']

    # Storage for results
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

    # Initialize predicted trajectory from first GT position
    pred_position = None
    
    # Identify available camera names directly from loaded episode data
    cam_names = list(episode_data['images'].keys())

    for frame_idx in range(num_frames):
        if frame_idx % 50 == 0:
            logger.info(f"Processing frame {frame_idx}/{num_frames}")

        try:
            gt_action = episode_data['actions'][frame_idx]

            images = {}
            for cam_name in cam_names:
                # Rename mapping for Eye_trocar datasets to match the simulation policy
                mapped_name = cam_name
                if cam_name == "camera1": mapped_name = "side_camera"
                elif cam_name == "camera2": mapped_name = "tool_camera"
                elif cam_name == "camera3": mapped_name = "top_camera"
                
                img_bytes = episode_data['images'][cam_name][frame_idx]
                images[mapped_name] = decode_image(img_bytes)

            state_6d = episode_data['ee_pose'][frame_idx]
            sensor_val = episode_data['sensor_data'][frame_idx]

            pred_action = evaluator.step(images, state_6d, sensor_val)

            # Handle dimension mismatch (if action dim > 6)
            pred_action_cmp = pred_action[:6]
            gt_action_cmp = gt_action[:6]

            action_error = np.mean((pred_action_cmp - gt_action_cmp) ** 2)
            position_error = np.mean((pred_action_cmp[:3] - gt_action_cmp[:3]) ** 2)
            rotation_error = np.mean((pred_action_cmp[3:6] - gt_action_cmp[3:6]) ** 2)

            if pred_position is None:
                pred_position = state_6d[:3].copy()
            else:
                pred_position = pred_position + pred_action[:3]

            gt_position = state_6d[:3]
            trajectory_error = np.sqrt(np.sum((gt_position - pred_position) ** 2))

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

    # Error Plot
    num_plots = 4
    fig, axes = plt.subplots(num_plots, 1, figsize=(14, 3*num_plots+1))
    fig.suptitle('Prediction Errors Over Episode (LeRobot Standard Inference)', fontsize=16)

    axes[0].plot(frames, results['action_errors'], color='blue')
    axes[0].set_ylabel('Action MSE')
    axes[0].set_title('Overall Action Error')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(frames, results['position_errors'], color='green')
    axes[1].set_ylabel('Position MSE')
    axes[1].set_title('Position Error')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(frames, results['rotation_errors'], color='red')
    axes[2].set_ylabel('Rotation MSE')
    axes[2].set_title('Rotation Error')
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(frames, results['trajectory_errors'], color='purple')
    axes[3].set_xlabel('Frame')
    axes[3].set_ylabel('Trajectory Error (mm)')
    axes[3].set_title('Accumulated Position Error')
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'error_over_time.png', dpi=150)
    plt.close()

    # Action Comparison Plot
    max_frames_to_plot = min(300, len(frames))
    gt_actions = np.array(results['gt_actions'][:max_frames_to_plot])
    pred_actions = np.array(results['pred_actions'][:max_frames_to_plot])
    frames_subset = frames[:max_frames_to_plot]

    dim_names = ['dx', 'dy', 'dz', 'drx', 'dry', 'drz']
    min_dim = min(gt_actions.shape[1], pred_actions.shape[1], 6)

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(f'Ground Truth vs Predicted Actions (First {max_frames_to_plot} Frames, LeRobot Standard)', fontsize=16)

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

    # Trajectory Plot
    if 'ee_poses' in results and len(results['ee_poses']) > 0:
        gt_poses = np.array(results['ee_poses'])
        pred_trajectory = np.array(results['pred_trajectory'])
        
        fig = plt.figure(figsize=(16, 12))

        # Full 3D View
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        ax1.plot(gt_poses[:, 0], gt_poses[:, 1], gt_poses[:, 2], 'b-', label='Ground Truth')
        ax1.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], pred_trajectory[:, 2], 'r--', label='Predicted')
        ax1.set_title('3D Trajectory - Full View')
        ax1.legend()

        # Top view (XY)
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(gt_poses[:, 0], gt_poses[:, 1], 'b-', label='Ground Truth')
        ax2.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], 'r--', label='Predicted')
        ax2.set_title('Top View (XY)')
        ax2.legend()
        ax2.axis('equal')

        # XZ
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(gt_poses[:, 0], gt_poses[:, 2], 'b-', label='GT')
        ax3.plot(pred_trajectory[:, 0], pred_trajectory[:, 2], 'r--', label='Predicted')
        ax3.set_title('Side View (XZ)')
        ax3.legend()
        ax3.axis('equal')

        # YZ
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.plot(gt_poses[:, 1], gt_poses[:, 2], 'b-', label='GT')
        ax4.plot(pred_trajectory[:, 1], pred_trajectory[:, 2], 'r--', label='Predicted')
        ax4.set_title('Front View (YZ)')
        ax4.legend()
        ax4.axis('equal')

        plt.suptitle('End-Effector Trajectory Comparison (GT vs Predicted)')
        plt.tight_layout()
        plt.savefig(output_dir / 'trajectory_3d.png', dpi=150)
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Evaluate episode with Standard LeRobot APIs")

    parser.add_argument("--pretrained_model", type=str, required=True, help="Path to pretrained model dir")
    parser.add_argument("--episode", type=str, required=True, help="Path to HDF5 episode file")
    parser.add_argument("--output_dir", type=str, default="outputs/episode_eval_lerobot", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cpu/cuda)")

    args = parser.parse_args()

    # Paths
    pretrained_dir = Path(args.pretrained_model)
    episode_path = Path(args.episode)
    output_dir = Path(args.output_dir)

    if not pretrained_dir.is_dir() or not (pretrained_dir / "config.json").exists():
        logger.error(f"Valid pretrained model not found at {pretrained_dir}.")
        sys.exit(1)

    if not episode_path.exists():
        logger.error(f"Episode not found: {episode_path}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize evaluator
    device = get_safe_torch_device(args.device, log=True)
    evaluator = SmolVLAEvaluator(str(pretrained_dir), device)
    
    # Identify expected cameras based on policy config inputs
    expected_cameras = []
    for input_key in evaluator.policy.config.input_features.keys():
        if input_key.startswith("observation.images."):
            expected_cameras.append(input_key.replace("observation.images.", ""))

    # Load episode data
    episode_data = load_episode_data(str(episode_path), expected_cameras)

    # Evaluate episode
    results = evaluate_episode(evaluator, episode_data)

    # Save details
    df = create_comparison_dataframe(results)
    csv_path = output_dir / 'frame_by_frame_comparison.csv'
    df.to_csv(csv_path, index=False)

    try:
        plot_results(results, output_dir)
    except Exception as e:
        logger.warning(f"Failed to create plots: {e}")

    logger.info(f"\nAll results saved to: {output_dir}")
    logger.info("Evaluation complete!")

if __name__ == "__main__":
    main()
