#!/usr/bin/env python
"""
DDP Usage:
    torchrun --nproc_per_node=5 train_smolvla_sim.py --config train_config_smolvla_sim.yaml
"""

import argparse
import logging
import os
import sys
import time
import gc
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from contextlib import nullcontext
import yaml

# Memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import GradScaler
from tqdm import tqdm
import numpy as np

# W&B for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not installed. Install with: pip install wandb")

# LeRobot imports
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.utils import init_logging
from lerobot.policies.utils import get_device_from_parameters

# HDF5 VLA imports
from hdf5_lerobot_adapter import create_hdf5_lerobot_dataset, hdf5_lerobot_collate_fn

# Normalization imports
from normalization_utils import Normalizer, load_stats

# Initialize logging
init_logging()
logger = logging.getLogger(__name__)


def setup_distributed():
    """Initialize distributed training environment (following lerobot_train.py)."""
    if torch.cuda.is_available() and int(os.environ.get("WORLD_SIZE", 1)) > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        logger.info(f"[DDP INIT] rank={rank}/{world_size}, local_rank={local_rank}")
        return rank, world_size, local_rank, True
    else:
        return 0, 1, 0, False


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_main_process():
    """Check if current process is main process (rank 0)."""
    return not dist.is_initialized() or dist.get_rank() == 0


def setup_wandb(config: Dict, resume: bool = False):
    """
    Initialize Weights & Biases for experiment tracking.

    Args:
        config: Full training configuration
        resume: Whether to resume from previous run

    Returns:
        wandb run object or None if wandb not available
    """
    if not WANDB_AVAILABLE:
        logger.warning("W&B not available, skipping W&B initialization")
        return None

    if not is_main_process():
        return None

    wandb_cfg = config.get("wandb", {})

    # Check if W&B is enabled
    if not wandb_cfg.get("enable", False):
        logger.info("W&B disabled in config")
        return None

    # W&B initialization
    project = wandb_cfg.get("project", "smolvla-sim-training")
    entity = wandb_cfg.get("entity", None)
    name = wandb_cfg.get("name", None)
    tags = wandb_cfg.get("tags", [])
    notes = wandb_cfg.get("notes", "")

    # Auto-generate name if not provided
    if name is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        name = f"smolvla_sim_{timestamp}"

    logger.info(f"Initializing W&B: project={project}, name={name}")

    try:
        run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            tags=tags,
            notes=notes,
            config=config,
            resume="allow" if resume else False,
        )

        logger.info(f"W&B initialized successfully: {run.url}")
        return run

    except Exception as e:
        logger.error(f"Failed to initialize W&B: {e}")
        return None


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def split_train_val_episodes(hdf5_files: List[Path], val_episodes: int = 10, seed: int = 1000) -> Tuple[List[Path], List[Path], List[int]]:
    """
    Split HDF5 episodes into train and validation sets.

    Args:
        hdf5_files: List of HDF5 file paths
        val_episodes: Number of episodes to use for validation
        seed: Random seed for reproducibility

    Returns:
        train_files, val_files, val_indices (indices of validation episodes)
    """
    import random
    random.seed(seed)

    # Randomly select validation episodes
    all_indices = list(range(len(hdf5_files)))
    val_indices = sorted(random.sample(all_indices, min(val_episodes, len(hdf5_files))))
    train_indices = [i for i in all_indices if i not in val_indices]

    train_files = [hdf5_files[i] for i in train_indices]
    val_files = [hdf5_files[i] for i in val_indices]

    return train_files, val_files, val_indices


def create_dataset_from_config(config: Dict, val_split: bool = False) -> Tuple[torch.utils.data.Dataset, Optional[torch.utils.data.Dataset], Optional[List[int]]]:
    """
    Create HDF5 LeRobot dataset from configuration with optional validation split.

    Args:
        config: Configuration dictionary
        val_split: Whether to create validation split

    Returns:
        train_dataset, val_dataset (None if val_split=False), val_episode_indices (None if val_split=False)
    """
    dataset_cfg = config["dataset"]
    policy_cfg = config["policy"]

    # Build full HDF5 file paths from sim dataset
    root_dir = Path(dataset_cfg["root_dir"])
    hdf5_files = []

    # Recursively search for all .h5 files in sim folders
    hdf5_files = sorted(list(root_dir.rglob("*.h5")) + list(root_dir.rglob("*.hdf5")))

    if len(hdf5_files) == 0:
        raise FileNotFoundError(f"No HDF5 files found in {root_dir}")

    if is_main_process():
        logger.info(f"Found {len(hdf5_files)} simulation HDF5 files")
        logger.info(f"Simulation data from: {root_dir}")

    # Split into train/val if requested
    val_dataset = None
    val_episode_indices = None

    if val_split:
        val_episodes = config.get("validation", {}).get("num_episodes", 10)
        seed = config.get("seed", 1000)
        train_files, val_files, val_episode_indices = split_train_val_episodes(hdf5_files, val_episodes, seed)

        if is_main_process():
            logger.info("=" * 80)
            logger.info("Simulation Train/Validation Split")
            logger.info("=" * 80)
            logger.info(f"Total episodes: {len(hdf5_files)}")
            logger.info(f"Train episodes: {len(train_files)}")
            logger.info(f"Val episodes: {len(val_files)}")
            logger.info(f"Val episode indices: {val_episode_indices}")
            logger.info("=" * 80)
    else:
        train_files = hdf5_files

    # Create tokenizer for language conditioning
    from transformers import AutoTokenizer
    tokenizer_max_length = policy_cfg.get("tokenizer_max_length", 48)
    vlm_model_name = policy_cfg.get("vlm_model_name", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(vlm_model_name)
    if is_main_process():
        logger.info(f"Created tokenizer from {vlm_model_name} (max_length={tokenizer_max_length})")

    # Camera mapping for simulation data (adjust if needed)
    # Simulation uses: side_camera, tool_camera, top_camera
    # LeRobot expects: camera1, camera2, camera3
    camera_mapping = {
        "side_camera": "camera1",
        "tool_camera": "camera2",
        "top_camera": "camera3",
    }

    # Get task instructions from config
    task_instruction = dataset_cfg.get("task_instruction", "Move the needle to the eye phantom and insert it through the trocar opening")
    task_instruction1 = dataset_cfg.get("task_instruction1", None)
    task_instruction2 = dataset_cfg.get("task_instruction2", None)

    # If task_instruction1 or task_instruction2 not provided, use task_instruction
    if task_instruction1 is None:
        task_instruction1 = task_instruction
    if task_instruction2 is None:
        task_instruction2 = task_instruction

    # Get camera selection from config
    selected_cameras = dataset_cfg.get("cameras", None)
    if selected_cameras is not None:
        if is_main_process():
            logger.info(f"Using selected cameras: {selected_cameras}")
    else:
        if is_main_process():
            logger.info("Using all available cameras")

    # Create train dataset
    train_dataset = create_hdf5_lerobot_dataset(
        hdf5_paths=train_files,
        horizon=dataset_cfg.get("horizon", 50),
        n_obs_steps=policy_cfg.get("n_obs_steps", 1),
        squeeze_n_obs_steps=(policy_cfg.get("n_obs_steps", 1) == 1),
        use_qpos=dataset_cfg.get("use_qpos", False),
        use_ee_pose=dataset_cfg.get("use_ee_pose", True),
        use_sensor=dataset_cfg.get("use_sensor", False),
        use_ee_pose_delta_as_action=dataset_cfg.get("use_ee_pose_delta_as_action", False),
        task_instruction=task_instruction,
        task_instruction1=task_instruction1,
        task_instruction2=task_instruction2,
        cameras=selected_cameras,
        tokenizer=tokenizer,
        tokenizer_max_length=tokenizer_max_length,
        # NO augmentation for sim data - DR was already applied during collection
        augment=False,
        augment_brightness=0.0,
        augment_contrast=0.0,
        augment_saturation=0.0,
        augment_hue=0.0,
        augment_noise=0.0,
    )

    if is_main_process():
        logger.info(f"Train dataset created: {len(train_dataset)} samples")
        logger.info("Note: No data augmentation applied (DR already done during sim data collection)")

    # Create validation dataset if split is requested
    if val_split:
        val_dataset = create_hdf5_lerobot_dataset(
            hdf5_paths=val_files,
            horizon=dataset_cfg.get("horizon", 50),
            n_obs_steps=policy_cfg.get("n_obs_steps", 1),
            squeeze_n_obs_steps=(policy_cfg.get("n_obs_steps", 1) == 1),
            use_qpos=dataset_cfg.get("use_qpos", False),
            use_ee_pose=dataset_cfg.get("use_ee_pose", True),
            use_sensor=dataset_cfg.get("use_sensor", False),
            use_ee_pose_delta_as_action=dataset_cfg.get("use_ee_pose_delta_as_action", False),
            task_instruction=task_instruction,
            task_instruction1=task_instruction1,
            task_instruction2=task_instruction2,
            cameras=selected_cameras,
            tokenizer=tokenizer,
            tokenizer_max_length=tokenizer_max_length,
            augment=False,
            augment_brightness=0.0,
            augment_contrast=0.0,
            augment_saturation=0.0,
            augment_hue=0.0,
            augment_noise=0.0,
        )

        if is_main_process():
            logger.info(f"Validation dataset created: {len(val_dataset)} samples")

    return train_dataset, val_dataset, val_episode_indices


def create_policy_from_config(config: Dict, device: torch.device) -> SmolVLAPolicy:
    """Create SmolVLA policy from configuration."""
    policy_cfg = config["policy"]

    if is_main_process():
        logger.info("=" * 80)
        logger.info("Creating SmolVLA Policy for Simulation")
        logger.info("=" * 80)
        logger.info(f"vlm_model_name: {policy_cfg.get('vlm_model_name', 'HuggingFaceTB/SmolVLM2-500M-Video-Instruct')}")
        logger.info(f"n_obs_steps: {policy_cfg.get('n_obs_steps', 1)}")
        logger.info(f"chunk_size: {policy_cfg.get('chunk_size', 50)}")
        logger.info(f"freeze_vision_encoder: {policy_cfg.get('freeze_vision_encoder', True)}")
        logger.info(f"train_expert_only: {policy_cfg.get('train_expert_only', True)}")

    # Determine state dimension from config
    dataset_cfg = config["dataset"]
    use_ee_pose = dataset_cfg.get("use_ee_pose", True)
    use_qpos = dataset_cfg.get("use_qpos", False)
    use_sensor = dataset_cfg.get("use_sensor", False)

    # State dimension: 6 for both ee_pose and qpos
    # ee_pose = [x, y, z, rx, ry, rz] (position + euler angles)
    # qpos = [q1, q2, q3, q4, q5, q6] (joint angles)
    state_dim = 0
    if use_ee_pose: state_dim += 6
    if use_qpos: state_dim += 6
    if use_sensor: state_dim += 1

    # Fallback default if nothing selected
    if state_dim == 0: state_dim = 6

    # Get selected cameras from config
    selected_cameras = dataset_cfg.get("cameras", None)
    if selected_cameras is None:
        # Default: use all 3 cameras
        selected_cameras = ["camera1", "camera2", "camera3"]

    if is_main_process():
        logger.info(f"State dimension: {state_dim} (use_ee_pose={use_ee_pose}, use_qpos={use_qpos}, use_sensor={use_sensor})")
        logger.info(f"Selected cameras for policy: {selected_cameras}")

    # Build input_features based on selected cameras
    input_features = {}
    for cam in selected_cameras:
        input_features[f"observation.images.{cam}"] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 512, 512))

    # Add state feature
    input_features["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(state_dim,))

    # Create SmolVLAConfig
    smolvla_config = SmolVLAConfig(
        vlm_model_name=policy_cfg.get("vlm_model_name", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"),
        n_obs_steps=policy_cfg.get("n_obs_steps", 1),
        chunk_size=policy_cfg.get("chunk_size", 50),
        n_action_steps=policy_cfg.get("n_action_steps", 50),
        max_state_dim=policy_cfg.get("max_state_dim", 32),
        max_action_dim=policy_cfg.get("max_action_dim", 32),
        num_steps=policy_cfg.get("num_steps", 10),
        freeze_vision_encoder=policy_cfg.get("freeze_vision_encoder", True),
        train_expert_only=policy_cfg.get("train_expert_only", True),
        train_state_proj=policy_cfg.get("train_state_proj", True),
        tokenizer_max_length=policy_cfg.get("tokenizer_max_length", 48),
        resize_imgs_with_padding=tuple(policy_cfg.get("resize_imgs_with_padding", [512, 512])),
        input_features=input_features,
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
        },
    )

    # Create policy
    policy = SmolVLAPolicy(smolvla_config)
    policy.to(device)

    if is_main_process():
        logger.info(f"Policy created and moved to {device}")
        logger.info("=" * 80)

    return policy


def compute_temporal_consistency_loss(actions: torch.Tensor, lambda_temporal: float = 0.1) -> torch.Tensor:
    """
    Compute temporal consistency loss to encourage smooth action trajectories.
    """
    if actions.shape[1] <= 1:
        return torch.tensor(0.0, device=actions.device)

    action_diffs = actions[:, 1:, :] - actions[:, :-1, :]
    temporal_loss = torch.mean(torch.square(action_diffs))

    return lambda_temporal * temporal_loss


def create_optimizer_from_config(policy: nn.Module, config: Dict) -> AdamW:
    """Create AdamW optimizer."""
    optimizer_cfg = config["optimizer"]

    trainable_params = [p for p in policy.parameters() if p.requires_grad]

    if is_main_process():
        total_params = sum(p.numel() for p in policy.parameters())
        trainable_params_count = sum(p.numel() for p in trainable_params)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params_count:,}")
        logger.info(f"Trainable %: {100 * trainable_params_count / total_params:.2f}%")

    optimizer = AdamW(
        trainable_params,
        lr=optimizer_cfg.get("lr", 1e-4),
        betas=tuple(optimizer_cfg.get("betas", [0.9, 0.95])),
        eps=optimizer_cfg.get("eps", 1e-8),
        weight_decay=optimizer_cfg.get("weight_decay", 1e-10),
    )

    return optimizer


def create_scheduler_from_config(optimizer: AdamW, config: Dict) -> LambdaLR:
    """Create learning rate scheduler with warmup and cosine decay."""
    scheduler_cfg = config["scheduler"]

    num_warmup_steps = scheduler_cfg.get("num_warmup_steps", 1000)
    num_decay_steps = scheduler_cfg.get("num_decay_steps", 30000)
    peak_lr = scheduler_cfg.get("peak_lr", 1e-4)
    decay_lr = scheduler_cfg.get("decay_lr", 2.5e-6)

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        else:
            progress = float(current_step - num_warmup_steps) / float(max(1, num_decay_steps - num_warmup_steps))
            progress = min(progress, 1.0)
            cosine_decay = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)))
            return (decay_lr / peak_lr) + (1.0 - decay_lr / peak_lr) * cosine_decay

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler


def normalize_batch(batch: dict, normalizer: Optional[Normalizer], device: torch.device) -> dict:
    """Apply normalization to state and action in batch."""
    if normalizer is None:
        return batch

    normalized_batch = {}
    for key, value in batch.items():
        # Move value to device first
        val_device = value.to(device) if isinstance(value, torch.Tensor) else value

        if key == "observation.state":
            # Check for dimension mismatch (e.g. 7D input vs 6D stats)
            # This happens when we add sensor data (1D) to robot state (6D)
            # Use tensor_stats instead of stats to ensure we have tensors with shape
            stats_dim = normalizer.tensor_stats[key]["mean"].shape[0]
            input_dim = val_device.shape[-1]

            if input_dim > stats_dim:
                # Split: [batch, 6] normalized, [batch, 1] raw
                state_to_norm = val_device[..., :stats_dim]
                sensor_data = val_device[..., stats_dim:]
                
                norm_state = normalizer.normalize(state_to_norm, key)
                normalized_batch[key] = torch.cat([norm_state, sensor_data], dim=-1)
            else:
                normalized_batch[key] = normalizer.normalize(val_device, key)
        elif key == "action":
            # Normalize action
            normalized_batch[key] = normalizer.normalize(val_device, key)
        else:
            # Keep other keys as-is
            normalized_batch[key] = val_device

    return normalized_batch


@torch.no_grad()
def evaluate_validation(
    policy: nn.Module,
    val_dataloader: DataLoader,
    normalizer: Optional[Normalizer] = None,
    use_amp: bool = False,
    max_batches: Optional[int] = None,
    is_distributed: bool = False,
) -> Dict[str, float]:
    """
    Evaluate policy on validation set.

    Args:
        policy: Policy model
        val_dataloader: Validation dataloader
        normalizer: Optional normalizer for state and action
        use_amp: Use automatic mixed precision
        max_batches: Maximum number of batches to evaluate (None = all)
        is_distributed: Whether running in distributed mode

    Returns:
        Dictionary with validation metrics
    """
    device = get_device_from_parameters(policy)
    policy.eval()

    total_loss = 0.0
    total_action_loss = 0.0
    total_metrics = {}
    num_batches = 0

    for batch_idx, batch in enumerate(val_dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # Apply normalization
        batch = normalize_batch(batch, normalizer, device)

        # Forward pass
        with torch.autocast(device_type=device.type, enabled=use_amp, dtype=torch.float16):
            loss, loss_dict = policy.forward(batch)
            action_loss = loss

        total_loss += loss.item()
        total_action_loss += action_loss.item()
        
        # Accumulate all metrics from loss_dict
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                if v.numel() > 1:
                    v = v.mean().item()
                else:
                    v = v.item()
            total_metrics[k] = total_metrics.get(k, 0.0) + v
            
        num_batches += 1

    # Distributed Aggregation
    if is_distributed:
        # Create tensor for [total_loss, total_action_loss, num_batches]
        # and one for each extra metric
        
        # 1. Base metrics
        base_metrics = torch.tensor([total_loss, total_action_loss, float(num_batches)], device=device)
        dist.all_reduce(base_metrics, op=dist.ReduceOp.SUM)
        
        total_loss = base_metrics[0].item()
        total_action_loss = base_metrics[1].item()
        total_batches_all = base_metrics[2].item()
        
        # 2. Detailed metrics (assuming same keys on all ranks)
        # Sort keys to ensure consistent order
        metric_keys = sorted(total_metrics.keys())
        if metric_keys:
            metric_vals = [total_metrics[k] for k in metric_keys]
            metric_tensor = torch.tensor(metric_vals, device=device)
            dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
            
            for i, k in enumerate(metric_keys):
                total_metrics[k] = metric_tensor[i].item()
                
        # Use aggregated batch count for averaging
        avg_divisor = max(total_batches_all, 1)
    else:
        avg_divisor = max(num_batches, 1)

    # Average metrics
    avg_loss = total_loss / avg_divisor
    avg_action_loss = total_action_loss / avg_divisor
    
    results = {
        "val/loss": avg_loss,
        "val/action_loss": avg_action_loss,
    }
    
    # Add averaged detailed metrics
    for k, v in total_metrics.items():
        results[f"val/{k}"] = v / avg_divisor

    return results

def update_policy(
    policy: nn.Module,
    batch: dict,
    optimizer: AdamW,
    grad_scaler: GradScaler,
    grad_clip_norm: float,
    lr_scheduler: LambdaLR,
    lambda_temporal: float,
    normalizer: Optional[Normalizer] = None,
    use_amp: bool = False,
    gradient_accumulation_steps: int = 1,
    accumulation_step: int = 0,
):
    """
    Performs a single training step with gradient accumulation support.
    """
    device = get_device_from_parameters(policy)
    policy.train()

    # Apply normalization to batch
    batch = normalize_batch(batch, normalizer, device)

    # Scale loss by accumulation steps to maintain gradient magnitude
    loss_scale = 1.0 / gradient_accumulation_steps

    # Use float16 explicitly for AMP
    with torch.autocast(device_type=device.type, enabled=use_amp, dtype=torch.float16):
        # Forward pass - SmolVLA returns (loss, loss_dict) tuple
        loss, loss_dict = policy.forward(batch)
        action_loss = loss

        # Compute temporal consistency loss on ground truth actions
        if "action" in batch and lambda_temporal > 0:
            gt_actions = batch["action"]
            temporal_loss = compute_temporal_consistency_loss(gt_actions, lambda_temporal)
            loss = action_loss + temporal_loss
        else:
            temporal_loss = torch.tensor(0.0, device=device)

        # Scale loss for gradient accumulation
        loss = loss * loss_scale

    # Backward pass
    if use_amp:
        grad_scaler.scale(loss).backward()
    else:
        loss.backward()

    # Only update weights after accumulating gradients
    is_update_step = (accumulation_step + 1) % gradient_accumulation_steps == 0

    if is_update_step:
        if use_amp:
            grad_scaler.unscale_(optimizer)

        # Clip gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(),
            grad_clip_norm,
            error_if_nonfinite=False,
        )

        # Update weights
        if use_amp:
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)

        if lr_scheduler is not None:
            lr_scheduler.step()
    else:
        grad_norm = 0.0

    # Return unscaled loss for logging
    return loss.item() / loss_scale, action_loss.item(), temporal_loss.item(), grad_norm if isinstance(grad_norm, float) else grad_norm.item()


def train(
    policy: nn.Module,
    dataloader: DataLoader,
    optimizer: AdamW,
    scheduler: LambdaLR,
    grad_scaler: GradScaler,
    config: Dict,
    rank: int,
    world_size: int,
    normalizer: Optional[Normalizer] = None,
    wandb_run=None,
    resume_checkpoint: Optional[str] = None,
    reset_scheduler: bool = False,
    val_dataloader: Optional[DataLoader] = None,
    val_episode_indices: Optional[List[int]] = None,
    hdf5_files: Optional[List[Path]] = None,
):
    """Main training loop (adapted for simulation data)."""
    training_cfg = config["training"]
    optimizer_cfg = config["optimizer"]
    validation_cfg = config.get("validation", {})

    total_steps = training_cfg.get("steps", 30000)
    log_freq = training_cfg.get("log_freq", 100)
    save_freq = training_cfg.get("save_freq", 5000)
    val_freq = validation_cfg.get("val_freq", 1000)
    grad_clip_norm = optimizer_cfg.get("grad_clip_norm", 10.0)
    lambda_temporal = training_cfg.get("lambda_temporal", 0.1)
    gradient_accumulation_steps = training_cfg.get("gradient_accumulation_steps", 1)
    use_amp = training_cfg.get("use_amp", False)

    output_dir = Path(config["output_dir"])
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Loss spike detection settings
    loss_spike_threshold = training_cfg.get("loss_spike_threshold", 3.0)
    loss_spike_log_path = output_dir / "loss_spikes.txt"
    loss_spike_csv_path = output_dir / "loss_spikes.csv"
    loss_history = []
    loss_spike_records = []

    # Save validation episode information
    if is_main_process() and val_episode_indices is not None and hdf5_files is not None:
        val_info_path = output_dir / "validation_episodes.txt"
        with open(val_info_path, 'w') as f:
            f.write("Validation Episodes (Simulation Data)\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Total episodes: {len(hdf5_files)}\n")
            f.write(f"Validation episodes: {len(val_episode_indices)}\n")
            f.write(f"Validation episode indices: {val_episode_indices}\n\n")
            f.write("Validation episode paths:\n")
            for idx in val_episode_indices:
                f.write(f"  [{idx}] {hdf5_files[idx]}\n")
        logger.info(f"Saved validation episode info to: {val_info_path}")

    # Initialize loss spike log files
    if is_main_process():
        with open(loss_spike_log_path, 'w') as f:
            f.write("Loss Spikes Log (Simulation Training)\n")
            f.write("=" * 80 + "\n")
            f.write(f"Threshold: {loss_spike_threshold}x moving average\n")
            f.write("=" * 80 + "\n\n")

        with open(loss_spike_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "step", "loss", "action_loss", "temporal_loss", "moving_avg",
                "ratio", "episode_indices", "frame_indices", "episode_paths"
            ])

        logger.info(f"Loss spike detection enabled (threshold: {loss_spike_threshold}x moving avg)")
        logger.info(f"  Text log: {loss_spike_log_path}")
        logger.info(f"  CSV log: {loss_spike_csv_path}")

    policy.train()
    step = 0
    running_loss = 0.0
    running_action_loss = 0.0
    running_temporal_loss = 0.0
    running_grad_norm = 0.0

    # Load checkpoint if resuming (same as train_smolvla.py)
    if resume_checkpoint is not None:
        if is_main_process():
            logger.info("=" * 80)
            logger.info(f"Resuming from checkpoint: {resume_checkpoint}")
            logger.info("=" * 80)

        checkpoint = torch.load(resume_checkpoint, map_location='cpu', weights_only=False)

        if 'policy_state_dict' in checkpoint:
            state_dict = checkpoint['policy_state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            raise KeyError("Checkpoint does not contain 'policy_state_dict' or 'model_state_dict'")

        # Handle DDP module prefix mismatch
        is_model_ddp = isinstance(policy, DDP)
        has_module_prefix = any(k.startswith('module.') for k in state_dict.keys())

        if is_model_ddp and not has_module_prefix:
            state_dict = {f'module.{k}': v for k, v in state_dict.items()}
            if is_main_process():
                logger.info("Added 'module.' prefix to checkpoint state_dict for DDP compatibility")
        elif not is_model_ddp and has_module_prefix:
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            if is_main_process():
                logger.info("Removed 'module.' prefix from checkpoint state_dict")

        policy.load_state_dict(state_dict)
        if is_main_process():
            logger.info("Loaded policy state_dict")

        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if is_main_process():
                logger.info("Loaded optimizer state_dict")
        except ValueError as e:
            if "doesn't match the size of optimizer's group" in str(e):
                if is_main_process():
                    logger.warning("Optimizer parameter groups don't match checkpoint")
                    logger.warning("Loading only matching parameter states")
            else:
                raise

        if reset_scheduler:
            if is_main_process():
                logger.info("Scheduler reset requested - will use fresh warmup schedule")
        else:
            if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                if is_main_process():
                    logger.info("Loaded scheduler state_dict")
                    current_lr = scheduler.get_last_lr()[0]
                    logger.info(f"  Current LR: {current_lr:.2e}")

        if use_amp and 'grad_scaler_state_dict' in checkpoint:
            grad_scaler.load_state_dict(checkpoint['grad_scaler_state_dict'])
            if is_main_process():
                logger.info("Loaded grad_scaler state_dict")

        step = checkpoint['step']
        if is_main_process():
            logger.info(f"Resuming from step {step}")
            logger.info("=" * 80)

        if dist.is_initialized():
            if is_main_process():
                logger.info("Synchronizing all ranks after checkpoint loading...")
            dist.barrier()
            if is_main_process():
                logger.info("All ranks synchronized")

    if is_main_process():
        batch_size = training_cfg.get('batch_size', 8)
        effective_batch = batch_size * world_size * gradient_accumulation_steps
        logger.info("=" * 80)
        logger.info("Starting SmolVLA Simulation Training")
        logger.info("=" * 80)
        logger.info(f"Total steps: {total_steps}")
        logger.info(f"Batch size per GPU: {batch_size}")
        logger.info(f"Number of GPUs: {world_size}")
        logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
        logger.info(f"Effective batch size: {effective_batch}")
        logger.info(f"Gradient clip norm: {grad_clip_norm}")
        logger.info(f"Temporal consistency weight (λ): {lambda_temporal}")
        logger.info(f"Mixed Precision (AMP): {use_amp}")
        logger.info("NOTE: Training on simulation data with DR applied")
        if wandb_run is not None:
            logger.info(f"W&B tracking: {wandb_run.url}")
        logger.info("=" * 80)

    start_time = time.time()

    def get_dataloader_iterator():
        """Create a fresh dataloader iterator to prevent memory buildup."""
        while True:
            for batch in dataloader:
                yield batch

    dl_iter = get_dataloader_iterator()

    if is_main_process():
        pbar = tqdm(total=total_steps, desc="Training", unit="step",
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    else:
        pbar = None

    for step in range(total_steps):
        accumulation_step = step % gradient_accumulation_steps

        batch = next(dl_iter)

        # Extract metadata for loss spike detection
        batch_episode_indices = batch.get("episode_index", [])
        batch_frame_indices = batch.get("frame_index", [])
        batch_timestamps = batch.get("timestamp", [])

        if isinstance(batch_episode_indices, torch.Tensor):
            batch_episode_indices = batch_episode_indices.cpu().tolist()
        if isinstance(batch_frame_indices, torch.Tensor):
            batch_frame_indices = batch_frame_indices.cpu().tolist()
        if isinstance(batch_timestamps, torch.Tensor):
            batch_timestamps = batch_timestamps.cpu().tolist()

        # Move batch to device
        batch = {k: v.to(get_device_from_parameters(policy)) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # Update policy
        loss, action_loss, temporal_loss, grad_norm = update_policy(
            policy,
            batch,
            optimizer,
            grad_scaler,
            grad_clip_norm,
            scheduler,
            lambda_temporal,
            normalizer=normalizer,
            use_amp=use_amp,
            gradient_accumulation_steps=gradient_accumulation_steps,
            accumulation_step=accumulation_step,
        )

        running_loss += loss
        running_action_loss += action_loss
        running_temporal_loss += temporal_loss
        if grad_norm > 0:
            running_grad_norm += grad_norm

        # Loss spike detection
        if is_main_process():
            loss_history.append(loss)
            if len(loss_history) > 100:
                loss_history.pop(0)

            if len(loss_history) >= 50:
                moving_avg = sum(loss_history) / len(loss_history)

                if loss > loss_spike_threshold * moving_avg:
                    ratio = loss / moving_avg
                    unique_episodes = sorted(set(batch_episode_indices))

                    episode_paths = []
                    if hdf5_files is not None:
                        for ep_idx in unique_episodes:
                            if ep_idx < len(hdf5_files):
                                episode_paths.append(f"[{ep_idx}] {hdf5_files[ep_idx].name}")

                    with open(loss_spike_log_path, 'a') as f:
                        f.write(f"\n{'='*80}\n")
                        f.write(f"LOSS SPIKE at Step {step + 1}\n")
                        f.write(f"{'='*80}\n")
                        f.write(f"Loss: {loss:.6f} (moving avg: {moving_avg:.6f}, ratio: {ratio:.2f}x)\n")
                        f.write(f"Action Loss: {action_loss:.6f}\n")
                        f.write(f"Temporal Loss: {temporal_loss:.6f}\n")
                        f.write(f"\nBatch Information:\n")
                        f.write(f"  Batch size: {len(batch_episode_indices)}\n")
                        f.write(f"  Episode indices: {batch_episode_indices}\n")
                        f.write(f"  Frame indices: {batch_frame_indices}\n")
                        if episode_paths:
                            f.write(f"\nEpisode Paths:\n")
                            for ep_path in episode_paths:
                                f.write(f"  {ep_path}\n")
                        f.write("\n")

                    with open(loss_spike_csv_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            step + 1,
                            f"{loss:.6f}",
                            f"{action_loss:.6f}",
                            f"{temporal_loss:.6f}",
                            f"{moving_avg:.6f}",
                            f"{ratio:.2f}",
                            str(batch_episode_indices),
                            str(batch_frame_indices),
                            "; ".join(episode_paths) if episode_paths else "N/A"
                        ])

                    spike_record = {
                        "step": step + 1,
                        "loss": float(loss),
                        "action_loss": float(action_loss),
                        "temporal_loss": float(temporal_loss),
                        "moving_avg": float(moving_avg),
                        "ratio": float(ratio),
                        "episode_indices": str(unique_episodes),
                        "num_episodes": len(unique_episodes),
                        "batch_size": len(batch_episode_indices),
                    }
                    loss_spike_records.append(spike_record)

                    spike_msg = (
                        f"⚠️  LOSS SPIKE at step {step + 1}: {loss:.4f} "
                        f"(avg: {moving_avg:.4f}, {ratio:.1f}x) | "
                        f"Episodes: {unique_episodes}"
                    )
                    pbar.write(spike_msg) if pbar is not None else logger.warning(spike_msg)

                    if wandb_run is not None:
                        wandb.log({
                            "spike/loss": float(loss),
                            "spike/moving_avg": float(moving_avg),
                            "spike/ratio": float(ratio),
                            "spike/step": int(step + 1),
                        })

        # Memory cleanup
        del batch
        if (step + 1) % 50 == 0:
            torch.cuda.empty_cache()
            gc.collect()

        # Update progress bar
        if pbar is not None:
            pbar.update(1)
            if (step + 1) % 10 == 0:
                current_lr = scheduler.get_last_lr()[0]
                pbar.set_postfix({
                    'loss': f'{running_loss / min(step % log_freq + 1, log_freq):.4f}',
                    'lr': f'{current_lr:.2e}'
                })

        # Logging
        if (step + 1) % log_freq == 0 and is_main_process():
            avg_loss = running_loss / log_freq
            avg_action_loss = running_action_loss / log_freq
            avg_temporal_loss = running_temporal_loss / log_freq
            avg_grad_norm = running_grad_norm / log_freq
            elapsed_time = time.time() - start_time
            steps_per_sec = log_freq / elapsed_time
            lr = scheduler.get_last_lr()[0]

            # Memory monitoring
            if PSUTIL_AVAILABLE:
                process = psutil.Process()
                ram_gb = process.memory_info().rss / (1024 ** 3)
                ram_percent = process.memory_percent()
            else:
                ram_gb = 0.0
                ram_percent = 0.0

            if torch.cuda.is_available():
                cuda_mem_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                cuda_mem_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            else:
                cuda_mem_allocated = 0.0
                cuda_mem_reserved = 0.0

            progress_pct = (step + 1) / total_steps * 100

            log_msg = (
                f"[{progress_pct:5.1f}%] Step {step + 1}/{total_steps} | "
                f"Loss: {avg_loss:.4f} (Action: {avg_action_loss:.4f}, Temporal: {avg_temporal_loss:.4f}) | "
                f"GradNorm: {avg_grad_norm:.3f} | "
                f"LR: {lr:.2e} | "
                f"{steps_per_sec:.2f} steps/sec | "
                f"VRAM: {cuda_mem_allocated:.1f}GB"
            )

            pbar.write(log_msg)

            # W&B logging
            if wandb_run is not None:
                log_dict = {
                    "train/loss": float(avg_loss),
                    "train/action_loss": float(avg_action_loss),
                    "train/temporal_loss": float(avg_temporal_loss),
                    "train/grad_norm": float(avg_grad_norm),
                    "train/learning_rate": float(lr),
                    "train/steps_per_sec": float(steps_per_sec),
                    "system/ram_gb": float(ram_gb),
                    "system/ram_percent": float(ram_percent),
                    "system/vram_allocated_gb": float(cuda_mem_allocated),
                    "system/vram_reserved_gb": float(cuda_mem_reserved),
                    "step": int(step + 1),
                }
                wandb.log(log_dict)
                del log_dict

            running_loss = 0.0
            running_action_loss = 0.0
            running_temporal_loss = 0.0
            running_grad_norm = 0.0
            start_time = time.time()

        # Validation
        if val_dataloader is not None and (step + 1) % val_freq == 0:
            if is_main_process():
                logger.info(f"Running validation at step {step + 1}...")

            val_metrics = evaluate_validation(
                policy,
                val_dataloader,
                normalizer=normalizer,
                use_amp=use_amp,
                max_batches=None,
                is_distributed=dist.is_initialized(),
            )
            
            # Logging (Main Process Only)
            if is_main_process():
                val_msg = f"[VAL] Step {step + 1} | Loss: {val_metrics['val/loss']:.4f} | Action Loss: {val_metrics['val/action_loss']:.4f}"
                pbar.write(val_msg) if pbar is not None else logger.info(val_msg)

                if wandb_run is not None:
                    # Log all metrics returned by evaluate_validation
                    val_log_dict = val_metrics.copy()
                    val_log_dict["step"] = int(step + 1)
                    wandb.log(val_log_dict)
                    del val_log_dict

            policy.train()

        # Checkpoint saving
        if (step + 1) % save_freq == 0 and is_main_process():
            logger.info(f"Saving checkpoint at step {step + 1}...")

            policy_to_save = policy.module if isinstance(policy, DDP) else policy

            checkpoint_path = checkpoints_dir / f"checkpoint_step_{step + 1}.pt"

            checkpoint_dict = {
                "step": step + 1,
                "epoch": 0,
                "policy_state_dict": policy_to_save.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": config,
            }

            if normalizer is not None:
                checkpoint_dict["normalization_stats"] = normalizer.stats

            torch.save(checkpoint_dict, checkpoint_path)
            logger.info(f"Saved checkpoint to: {checkpoint_path}")

    # Close progress bar
    if pbar is not None:
        pbar.close()

    # Upload loss spike table to W&B
    if is_main_process() and wandb_run is not None and len(loss_spike_records) > 0:
        logger.info(f"Uploading {len(loss_spike_records)} loss spike records to W&B...")
        try:
            import pandas as pd
            spike_df = pd.DataFrame(loss_spike_records)
            spike_table = wandb.Table(dataframe=spike_df)
            wandb.log({"loss_spikes_table": spike_table})
            logger.info("Loss spike table uploaded to W&B successfully")
        except ImportError:
            logger.warning("pandas not available, skipping W&B table upload")
        except Exception as e:
            logger.warning(f"Failed to upload loss spike table to W&B: {e}")

    # Save final checkpoint
    if is_main_process():
        logger.info("=" * 80)
        logger.info("Saving final model...")
        logger.info("=" * 80)

        policy_to_save = policy.module if isinstance(policy, DDP) else policy

        # Compute normalization statistics
        logger.info("Computing action normalization statistics from dataset...")
        all_actions = []
        sample_count = 0
        max_samples = 10000

        for batch in dataloader:
            if "action" in batch:
                actions = batch["action"]
                if actions.ndim == 3:
                    actions = actions.reshape(-1, actions.shape[-1])
                elif actions.ndim == 2:
                    pass
                all_actions.append(actions.cpu().numpy())
                sample_count += actions.shape[0]

            if sample_count >= max_samples:
                break

        if len(all_actions) > 0:
            all_actions = np.concatenate(all_actions, axis=0)
            action_min = all_actions.min(axis=0)
            action_max = all_actions.max(axis=0)
            action_mean = all_actions.mean(axis=0)
            action_std = all_actions.std(axis=0)

            logger.info(f"Action statistics computed from {len(all_actions)} samples:")
            logger.info(f"  Min: {action_min}")
            logger.info(f"  Max: {action_max}")
            logger.info(f"  Mean: {action_mean}")
            logger.info(f"  Std: {action_std}")
        else:
            logger.warning("No actions found, using default normalization")
            action_dim = 6
            action_min = np.array([-1.0] * action_dim)
            action_max = np.array([1.0] * action_dim)
            action_mean = np.array([0.0] * action_dim)
            action_std = np.array([1.0] * action_dim)

        # Save final checkpoint
        final_checkpoint_path = checkpoints_dir / "checkpoint_latest.pt"

        checkpoint_dict = {
            "step": total_steps,
            "epoch": 0,
            "policy_state_dict": policy_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": config,
            "action_min": action_min,
            "action_max": action_max,
            "action_mean": action_mean,
            "action_std": action_std,
        }

        if normalizer is not None:
            checkpoint_dict["normalization_stats"] = normalizer.stats

        torch.save(checkpoint_dict, final_checkpoint_path)
        logger.info(f"Saved final checkpoint (.pt) to: {final_checkpoint_path}")

        # Also save HuggingFace format
        hf_checkpoint_path = checkpoints_dir / "final_hf"
        policy_to_save.save_pretrained(hf_checkpoint_path)
        logger.info(f"Saved HuggingFace format to: {hf_checkpoint_path}")

        logger.info("=" * 80)
        logger.info("Simulation Training completed successfully!")
        logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Train SmolVLA Policy on Simulation HDF5 Data")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--steps", type=int, default=None, help="Override training steps")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--reset_scheduler", action="store_true", help="Reset scheduler when resuming")
    args = parser.parse_args()

    # Setup distributed training
    rank, world_size, local_rank, is_distributed = setup_distributed()

    # Load config
    config = load_config(args.config)

    # Override config with command line arguments
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.steps is not None:
        config["training"]["steps"] = args.steps
    if args.lr is not None:
        config["optimizer"]["lr"] = args.lr
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir

    # Set random seed
    seed = config.get("seed", 1000)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Get device
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")

    if is_main_process():
        logger.info(f"Using device: {device}")

    # Check if validation is enabled
    validation_enabled = config.get("validation", {}).get("enable", False)

    # Create dataset with validation split if enabled
    if validation_enabled:
        train_dataset, val_dataset, val_episode_indices = create_dataset_from_config(config, val_split=True)

        dataset_cfg = config["dataset"]
        root_dir = Path(dataset_cfg["root_dir"])
        all_hdf5_files = sorted(list(root_dir.rglob("*.h5")) + list(root_dir.rglob("*.hdf5")))
    else:
        train_dataset, val_dataset, val_episode_indices = create_dataset_from_config(config, val_split=False)
        all_hdf5_files = None

    # Create train dataloader with distributed sampler
    train_sampler = DistributedSampler(train_dataset, shuffle=False) if is_distributed else None

    num_workers = config["training"].get("num_workers", 0)

    if is_main_process():
        if num_workers == 0:
            logger.info("DataLoader: num_workers=0 (loading in main process)")
        else:
            logger.info(f"DataLoader: num_workers={num_workers}")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=config["training"].get("pin_memory", True),
        sampler=train_sampler,
        collate_fn=hdf5_lerobot_collate_fn,
        drop_last=False,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=(num_workers > 0),
    )

    # Create validation dataloader if validation is enabled
    val_dataloader = None
    if validation_enabled and val_dataset is not None:
        val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False) if is_distributed else None
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=config["training"].get("pin_memory", True),
            collate_fn=hdf5_lerobot_collate_fn,
            drop_last=False,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=(num_workers > 0),
            sampler=val_sampler,
        )
        if is_main_process():
            logger.info(f"Validation dataloader created (Distributed: {val_sampler is not None})")

    dataloader = train_dataloader

    # Create normalizer if normalization is enabled
    normalizer = None
    if config.get("normalization", {}).get("enable", False):
        stats_file = config["normalization"].get("stats_file", "dataset_stats_sim.yaml")
        stats_path = Path(__file__).parent / stats_file

        if stats_path.exists():
            if is_main_process():
                logger.info("=" * 80)
                logger.info("Loading Normalization Statistics (Simulation)")
                logger.info("=" * 80)
                logger.info(f"Stats file: {stats_path}")

            stats = load_stats(str(stats_path))
            normalizer = Normalizer(stats).to(device)

            if is_main_process():
                logger.info("Normalizer created successfully")
                logger.info(f"  Action mean: {stats['action']['mean'][:3]}...")
                logger.info(f"  Action std:  {stats['action']['std'][:3]}...")
                logger.info(f"  State mean:  {stats['observation.state']['mean'][:3]}...")
                logger.info(f"  State std:   {stats['observation.state']['std'][:3]}...")
                logger.info("=" * 80)
        else:
            if is_main_process():
                logger.warning(f"Normalization enabled but stats file not found: {stats_path}")
                logger.warning("Training without normalization!")

    # Create policy
    policy = create_policy_from_config(config, device)

    # Wrap policy with DDP if using distributed training
    if is_distributed:
        policy = DDP(
            policy,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
        if is_main_process():
            logger.info("Policy wrapped with DistributedDataParallel")

    # Create optimizer and scheduler
    optimizer = create_optimizer_from_config(policy, config)
    scheduler = create_scheduler_from_config(optimizer, config)

    # Create GradScaler for Mixed Precision Training
    use_amp = config["training"].get("use_amp", False)
    grad_scaler = GradScaler(device.type, enabled=use_amp)

    # Setup W&B (main process only)
    wandb_run = None
    if is_main_process():
        wandb_run = setup_wandb(config)

    # Train
    try:
        train(
            policy=policy,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            grad_scaler=grad_scaler,
            config=config,
            rank=rank,
            world_size=world_size,
            normalizer=normalizer,
            wandb_run=wandb_run,
            resume_checkpoint=args.resume,
            reset_scheduler=args.reset_scheduler,
            val_dataloader=val_dataloader,
            val_episode_indices=val_episode_indices,
            hdf5_files=all_hdf5_files,
        )
    except KeyboardInterrupt:
        if is_main_process():
            logger.info("\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise
    finally:
        if is_main_process() and wandb_run is not None:
            wandb.finish()
        cleanup_distributed()


if __name__ == "__main__":
    main()
