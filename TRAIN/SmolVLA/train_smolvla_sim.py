#!/usr/bin/env python
"""
SmolVLA Training script for Simulation Data - Following LeRobot Training Methodology

This script follows the exact training approach from lerobot_train.py:
- Uses make_pre_post_processors with dataset.meta.stats for normalization
- Follows SmolVLA paper settings (100k steps, batch=64, warmup=100, cosine LR)
- Uses LeRobot's NormalizerProcessorStep for state/action normalization

Usage:
    python train_smolvla_sim.py --config train_config_smolvla_sim.yaml

DDP Usage:
    torchrun --nproc_per_node=5 train_smolvla_sim.py --config train_config_smolvla_sim.yaml
"""

import argparse
import logging
import os
import sys
import time
import gc
from pathlib import Path
from typing import Dict, List, Optional
import yaml

# Memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

import torch
import torch.nn as nn
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

# LeRobot imports
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.configs.types import FeatureType, PolicyFeature, NormalizationMode
from lerobot.utils.utils import init_logging
from lerobot.policies.utils import get_device_from_parameters

# HDF5 VLA imports
from hdf5_lerobot_adapter import create_hdf5_lerobot_dataset, hdf5_lerobot_collate_fn

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
    """Initialize Weights & Biases for experiment tracking."""
    if not WANDB_AVAILABLE or not is_main_process():
        return None

    wandb_cfg = config.get("wandb", {})
    if not wandb_cfg.get("enable", False):
        logger.info("W&B disabled in config")
        return None

    try:
        run = wandb.init(
            project=wandb_cfg.get("project", "smolvla-sim-training"),
            entity=wandb_cfg.get("entity", None),
            name=wandb_cfg.get("name", None),
            tags=wandb_cfg.get("tags", []),
            notes=wandb_cfg.get("notes", ""),
            config=config,
            resume="allow" if resume else False,
        )
        logger.info(f"W&B initialized: {run.url}")
        return run
    except Exception as e:
        logger.error(f"Failed to initialize W&B: {e}")
        return None


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dataset_meta_with_stats(hdf5_files: List[Path], config: Dict) -> object:
    """
    Create dataset metadata with statistics for normalization.

    This mimics LeRobotDataset.meta structure with stats for NormalizerProcessorStep.
    Following the LeRobot approach: dataset.meta.stats is used by make_pre_post_processors.
    """
    dataset_cfg = config["dataset"]
    policy_cfg = config["policy"]

    # State/action dimensions
    state_dim = 6  # ee_pose or qpos
    action_dim = 6

    # Create meta object
    class DatasetMeta:
        pass

    meta = DatasetMeta()

    # Load stats from YAML if normalization is enabled (CRITICAL for proper training!)
    normalization_cfg = config.get("normalization", {})
    if normalization_cfg.get("enable", False):
        stats_file = normalization_cfg.get("stats_file", "dataset_stats.yaml")
        stats_path = Path(__file__).parent / stats_file

        if stats_path.exists():
            if is_main_process():
                logger.info("=" * 80)
                logger.info("Loading Dataset Statistics")
                logger.info("=" * 80)
                logger.info(f"Stats file: {stats_path}")

            import yaml
            with open(stats_path, 'r') as f:
                stats_dict = yaml.safe_load(f)

            # Convert stats to torch tensors (required by NormalizerProcessorStep)
            meta.stats = {}
            for key, value in stats_dict.items():
                meta.stats[key] = {}
                for stat_name, stat_value in value.items():
                    if isinstance(stat_value, list):
                        meta.stats[key][stat_name] = torch.tensor(stat_value, dtype=torch.float32)
                    elif isinstance(stat_value, (int, float)):
                        meta.stats[key][stat_name] = torch.tensor([stat_value], dtype=torch.float32)
                    else:
                        meta.stats[key][stat_name] = torch.tensor(stat_value, dtype=torch.float32)

            if is_main_process():
                logger.info("Statistics loaded successfully")
                logger.info(f"  action mean: {meta.stats['action']['mean'][:3].tolist()}...")
                logger.info(f"  action std:  {meta.stats['action']['std'][:3].tolist()}...")
                if 'observation.state' in meta.stats:
                    logger.info(f"  state mean:  {meta.stats['observation.state']['mean'][:3].tolist()}...")
                    logger.info(f"  state std:   {meta.stats['observation.state']['std'][:3].tolist()}...")
                logger.info("=" * 80)
        else:
            # No stats file - use placeholder (mean=0, std=1)
            # This is NOT recommended - compute real stats with compute_dataset_stats.py!
            if is_main_process():
                logger.warning("=" * 80)
                logger.warning("No normalization statistics found!")
                logger.warning(f"Expected file: {stats_path}")
                logger.warning("Using placeholder stats (mean=0, std=1)")
                logger.warning("For best results, run: python compute_dataset_stats.py")
                logger.warning("=" * 80)

            meta.stats = {
                "action": {
                    "mean": torch.zeros(action_dim, dtype=torch.float32),
                    "std": torch.ones(action_dim, dtype=torch.float32),
                    "min": torch.full((action_dim,), -1.0, dtype=torch.float32),
                    "max": torch.full((action_dim,), 1.0, dtype=torch.float32),
                },
                "observation.state": {
                    "mean": torch.zeros(state_dim, dtype=torch.float32),
                    "std": torch.ones(state_dim, dtype=torch.float32),
                    "min": torch.full((state_dim,), -1.0, dtype=torch.float32),
                    "max": torch.full((state_dim,), 1.0, dtype=torch.float32),
                },
            }
    else:
        # Normalization disabled - use identity (no normalization)
        meta.stats = None
        if is_main_process():
            logger.info("Normalization disabled - training without normalization")

    # Camera keys
    meta.camera_keys = ["observation.images.camera1", "observation.images.camera2", "observation.images.camera3"]

    # FPS (for delta_timestamps if needed)
    meta.fps = 10

    return meta


def create_dataset_from_config(config: Dict) -> torch.utils.data.Dataset:
    """Create HDF5 LeRobot dataset from configuration."""
    dataset_cfg = config["dataset"]
    policy_cfg = config["policy"]

    # Build full HDF5 file paths
    root_dir = Path(dataset_cfg["root_dir"])
    hdf5_files = sorted(list(root_dir.rglob("*.h5")) + list(root_dir.rglob("*.hdf5")))

    if len(hdf5_files) == 0:
        raise FileNotFoundError(f"No HDF5 files found in {root_dir}")

    if is_main_process():
        logger.info(f"Found {len(hdf5_files)} HDF5 files")

    # Create tokenizer
    from transformers import AutoTokenizer
    tokenizer_max_length = policy_cfg.get("tokenizer_max_length", 48)
    vlm_model_name = policy_cfg.get("vlm_model_name", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(vlm_model_name)

    # Create dataset
    dataset = create_hdf5_lerobot_dataset(
        hdf5_paths=hdf5_files,
        horizon=dataset_cfg.get("horizon", 50),
        n_obs_steps=policy_cfg.get("n_obs_steps", 1),
        squeeze_n_obs_steps=(policy_cfg.get("n_obs_steps", 1) == 1),
        use_qpos=dataset_cfg.get("use_qpos", False),
        use_ee_pose=dataset_cfg.get("use_ee_pose", True),
        use_ee_pose_delta_as_action=dataset_cfg.get("use_ee_pose_delta_as_action", False),
        task_instruction=dataset_cfg.get("task_instruction", "Insert needle into target"),
        tokenizer=tokenizer,
        tokenizer_max_length=tokenizer_max_length,
        augment=dataset_cfg.get("augment", False),
        augment_brightness=dataset_cfg.get("augment_brightness", 0.0),
        augment_contrast=dataset_cfg.get("augment_contrast", 0.0),
        augment_saturation=dataset_cfg.get("augment_saturation", 0.0),
        augment_hue=dataset_cfg.get("augment_hue", 0.0),
        augment_noise=dataset_cfg.get("augment_noise", 0.0),
    )

    # Add meta with stats to dataset (for make_pre_post_processors)
    dataset.meta = create_dataset_meta_with_stats(hdf5_files, config)

    if is_main_process():
        logger.info(f"Dataset created: {len(dataset)} samples")

    return dataset


def create_policy_from_config(config: Dict, device: torch.device) -> SmolVLAPolicy:
    """Create SmolVLA policy from configuration following SmolVLA paper settings."""
    policy_cfg = config["policy"]
    dataset_cfg = config["dataset"]

    state_dim = 6
    action_dim = 6

    if is_main_process():
        logger.info("=" * 80)
        logger.info("Creating SmolVLA Policy")
        logger.info("=" * 80)
        logger.info(f"VLM: {policy_cfg.get('vlm_model_name')}")
        logger.info(f"n_obs_steps: {policy_cfg.get('n_obs_steps', 1)}")
        logger.info(f"chunk_size: {policy_cfg.get('chunk_size', 50)}")
        logger.info(f"freeze_vision_encoder: {policy_cfg.get('freeze_vision_encoder', True)}")
        logger.info(f"train_expert_only: {policy_cfg.get('train_expert_only', True)}")

    # SmolVLA normalization mapping (from paper)
    # VISUAL: IDENTITY (no normalization for images)
    # STATE: MEAN_STD (normalize state with mean/std)
    # ACTION: MEAN_STD (normalize action with mean/std)
    normalization_mapping = {
        FeatureType.VISUAL: NormalizationMode.IDENTITY,
        FeatureType.STATE: NormalizationMode.MEAN_STD,
        FeatureType.ACTION: NormalizationMode.MEAN_STD,
    }

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
        normalization_mapping=normalization_mapping,  # CRITICAL: Use SmolVLA normalization settings
        input_features={
            "observation.images.camera1": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 512, 512)),
            "observation.images.camera2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 512, 512)),
            "observation.images.camera3": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 512, 512)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,)),
        },
    )

    # Create policy
    policy = SmolVLAPolicy(smolvla_config)
    policy.to(device)

    if is_main_process():
        total_params = sum(p.numel() for p in policy.parameters())
        trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
        logger.info("=" * 80)

    return policy


def create_optimizer_from_config(policy: nn.Module, config: Dict) -> AdamW:
    """Create AdamW optimizer following SmolVLA paper settings."""
    optimizer_cfg = config["optimizer"]

    trainable_params = [p for p in policy.parameters() if p.requires_grad]

    optimizer = AdamW(
        trainable_params,
        lr=optimizer_cfg.get("lr", 1e-4),
        betas=tuple(optimizer_cfg.get("betas", [0.9, 0.95])),  # Paper: β1=0.9, β2=0.95
        eps=optimizer_cfg.get("eps", 1e-8),
        weight_decay=optimizer_cfg.get("weight_decay", 1e-6),
    )

    return optimizer


def create_scheduler_from_config(optimizer: AdamW, config: Dict) -> LambdaLR:
    """Create learning rate scheduler with warmup and cosine decay (SmolVLA paper)."""
    scheduler_cfg = config["scheduler"]

    num_warmup_steps = scheduler_cfg.get("num_warmup_steps", 100)  # Paper: 100
    num_decay_steps = scheduler_cfg.get("num_decay_steps", 100000)
    peak_lr = scheduler_cfg.get("peak_lr", 1e-4)  # Paper: 1e-4
    decay_lr = scheduler_cfg.get("decay_lr", 2.5e-6)  # Paper: 2.5e-6

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


def train(
    policy: nn.Module,
    dataloader: DataLoader,
    optimizer: AdamW,
    scheduler: LambdaLR,
    grad_scaler: GradScaler,
    preprocessor,
    postprocessor,
    config: Dict,
    rank: int,
    world_size: int,
    wandb_run=None,
):
    """Main training loop following LeRobot methodology."""
    training_cfg = config["training"]
    optimizer_cfg = config["optimizer"]

    total_steps = training_cfg.get("steps", 100000)
    log_freq = training_cfg.get("log_freq", 100)
    save_freq = training_cfg.get("save_freq", 10000)
    grad_clip_norm = optimizer_cfg.get("grad_clip_norm", 10.0)
    gradient_accumulation_steps = training_cfg.get("gradient_accumulation_steps", 1)
    use_amp = training_cfg.get("use_amp", False)

    output_dir = Path(config["output_dir"])
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    if is_main_process():
        batch_size = training_cfg.get('batch_size', 64)
        effective_batch = batch_size * world_size * gradient_accumulation_steps
        logger.info("=" * 80)
        logger.info("Starting SmolVLA Training (Following LeRobot Methodology)")
        logger.info("=" * 80)
        logger.info(f"Total steps: {total_steps}")
        logger.info(f"Batch size per GPU: {batch_size}")
        logger.info(f"Number of GPUs: {world_size}")
        logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
        logger.info(f"Effective batch size: {effective_batch}")
        logger.info(f"Gradient clip norm: {grad_clip_norm}")
        logger.info(f"Mixed Precision (AMP): {use_amp}")
        logger.info("=" * 80)

    policy.train()
    step = 0
    running_loss = 0.0
    start_time = time.time()

    def get_dataloader_iterator():
        """Create a fresh dataloader iterator."""
        while True:
            for batch in dataloader:
                yield batch

    dl_iter = get_dataloader_iterator()

    if is_main_process():
        pbar = tqdm(total=total_steps, desc="Training", unit="step")
    else:
        pbar = None

    for step in range(total_steps):
        accumulation_step = step % gradient_accumulation_steps

        batch = next(dl_iter)

        # CRITICAL: Apply preprocessor (includes normalization via NormalizerProcessorStep)
        # This is the LeRobot way - preprocessor handles normalization automatically
        batch = preprocessor(batch)

        # Scale loss by accumulation steps
        loss_scale = 1.0 / gradient_accumulation_steps

        # Forward pass with AMP
        with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu',
                           enabled=use_amp, dtype=torch.float16):
            loss, loss_dict = policy.forward(batch)
            loss = loss * loss_scale

        # Backward pass
        if use_amp:
            grad_scaler.scale(loss).backward()
        else:
            loss.backward()

        # Update weights after accumulation
        is_update_step = (accumulation_step + 1) % gradient_accumulation_steps == 0

        if is_update_step:
            if use_amp:
                grad_scaler.unscale_(optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(),
                grad_clip_norm,
                error_if_nonfinite=False,
            )

            if use_amp:
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            if scheduler is not None:
                scheduler.step()

        running_loss += loss.item() / loss_scale

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
            elapsed_time = time.time() - start_time
            steps_per_sec = log_freq / elapsed_time
            lr = scheduler.get_last_lr()[0]

            log_msg = (
                f"[{(step + 1) / total_steps * 100:5.1f}%] Step {step + 1}/{total_steps} | "
                f"Loss: {avg_loss:.4f} | LR: {lr:.2e} | {steps_per_sec:.2f} steps/sec"
            )
            pbar.write(log_msg)

            if wandb_run is not None:
                wandb.log({
                    "train/loss": float(avg_loss),
                    "train/learning_rate": float(lr),
                    "train/steps_per_sec": float(steps_per_sec),
                    "step": int(step + 1),
                })

            running_loss = 0.0
            start_time = time.time()

        # Checkpoint saving
        if (step + 1) % save_freq == 0 and is_main_process():
            policy_to_save = policy.module if isinstance(policy, DDP) else policy
            checkpoint_path = checkpoints_dir / f"checkpoint_step_{step + 1}.pt"

            torch.save({
                "step": step + 1,
                "policy_state_dict": policy_to_save.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": config,
            }, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

    if pbar is not None:
        pbar.close()

    # Save final checkpoint
    if is_main_process():
        policy_to_save = policy.module if isinstance(policy, DDP) else policy
        final_checkpoint_path = checkpoints_dir / "checkpoint_latest.pt"

        torch.save({
            "step": total_steps,
            "policy_state_dict": policy_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": config,
        }, final_checkpoint_path)

        # Also save HuggingFace format
        hf_checkpoint_path = checkpoints_dir / "final_hf"
        policy_to_save.save_pretrained(hf_checkpoint_path)

        logger.info("=" * 80)
        logger.info("Training completed successfully!")
        logger.info(f"Final checkpoint: {final_checkpoint_path}")
        logger.info(f"HuggingFace format: {hf_checkpoint_path}")
        logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Train SmolVLA Policy (LeRobot Methodology)")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--steps", type=int, default=None, help="Override training steps")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
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

    # Create dataset (with meta.stats)
    dataset = create_dataset_from_config(config)

    # Create train dataloader
    train_sampler = DistributedSampler(dataset, shuffle=False) if is_distributed else None
    num_workers = config["training"].get("num_workers", 0)

    dataloader = DataLoader(
        dataset,
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

    # Create policy
    policy = create_policy_from_config(config, device)

    # CRITICAL: Create preprocessor and postprocessor (LeRobot way)
    # This creates NormalizerProcessorStep with dataset.meta.stats
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        dataset_stats=dataset.meta.stats,  # Pass dataset stats for normalization
    )

    if is_main_process():
        logger.info("=" * 80)
        logger.info("Preprocessor/Postprocessor created")
        logger.info("Using LeRobot's NormalizerProcessorStep for normalization")
        logger.info("=" * 80)

    # Wrap policy with DDP if using distributed training
    if is_distributed:
        policy = DDP(
            policy,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
        if is_main_process():
            logger.info("Policy wrapped with DistributedDataParallel")

    # Create optimizer and scheduler (SmolVLA paper settings)
    optimizer = create_optimizer_from_config(policy, config)
    scheduler = create_scheduler_from_config(optimizer, config)

    # Create GradScaler for Mixed Precision Training
    use_amp = config["training"].get("use_amp", False)
    grad_scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu', enabled=use_amp)

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
            preprocessor=preprocessor,  # Pass preprocessor (handles normalization)
            postprocessor=postprocessor,
            config=config,
            rank=rank,
            world_size=world_size,
            wandb_run=wandb_run,
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
