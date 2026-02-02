"""
NORA VLA Training Script with Custom HDF5 Dataset
==================================================

This script trains the NORA VLA model using custom HDF5 datasets.
It integrates the HDF5LeRobotDataset adapter with NORA's training pipeline.

Based on: nora/training/train.py
Dataset: hdf5_lerobot_adapter.py

Usage:
    python train_nora.py --config train_config_nora.yaml
"""

import os
import sys
import json
import logging
import argparse
import gc
import psutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from glob import glob

# Set NCCL timeout to 30 minutes (1800 seconds) to prevent timeout during validation/checkpointing
os.environ['NCCL_TIMEOUT'] = '1800'
os.environ['NCCL_BLOCKING_WAIT'] = '1'  # Make NCCL operations blocking for better error messages

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

from accelerate import Accelerator
from accelerate.logging import get_logger
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from transformers import get_scheduler
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import wandb
import yaml

# Import custom dataset adapter
from hdf5_lerobot_adapter import create_hdf5_lerobot_dataset, hdf5_lerobot_collate_fn

# Import normalization utilities
from normalization_utils import Normalizer, load_stats

# Initialize logger - will work both with and without Accelerate
def _get_safe_logger():
    """Get logger that works with or without Accelerate initialization."""
    try:
        return get_logger(__name__)
    except RuntimeError:
        # Fallback to standard Python logging if Accelerate is not initialized
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        return logging.getLogger(__name__)

logger = _get_safe_logger()


# --- Memory Monitoring ---
def get_memory_stats():
    """Get current memory usage statistics."""
    process = psutil.Process()
    ram_mb = process.memory_info().rss / 1024 / 1024  # MB
    ram_percent = process.memory_percent()

    gpu_mb = 0
    gpu_percent = 0
    if torch.cuda.is_available():
        gpu_mb = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        gpu_reserved_mb = torch.cuda.memory_reserved() / 1024 / 1024  # MB
        gpu_percent = (gpu_mb / (torch.cuda.get_device_properties(0).total_memory / 1024 / 1024)) * 100

    return {
        "ram_mb": ram_mb,
        "ram_percent": ram_percent,
        "gpu_mb": gpu_mb,
        "gpu_reserved_mb": gpu_reserved_mb if torch.cuda.is_available() else 0,
        "gpu_percent": gpu_percent,
    }


# --- 1. Configuration ---
class NoraTrainingConfig:
    """Training configuration for NORA VLA with custom HDF5 datasets."""

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration from dictionary (loaded from YAML).

        Args:
            config_dict: Configuration dictionary from YAML file
        """
        if config_dict is None:
            config_dict = {}

        # Training parameters
        self.per_device_batch_size = config_dict.get('per_device_batch_size', 8)
        self.learning_rate = config_dict.get('learning_rate', 5e-5)
        self.gradient_accumulation_steps = config_dict.get('gradient_accumulation_steps', 2)
        self.num_warmup_steps = config_dict.get('num_warmup_steps', 1000)
        self.max_train_steps = config_dict.get('max_train_steps', 100000)
        self.gradient_clipping = config_dict.get('gradient_clipping', None)

        # Paths
        self.output_dir = config_dict.get('output_dir', '/home/najo/NAS/VLA/Insertion_VLA_Sim/outputs/nora')
        self.resume_from_checkpoint = config_dict.get('resume_from_checkpoint', '')
        self.load_model_weights = config_dict.get('load_model_weights', None)

        # Dataset configuration
        dataset_cfg = config_dict.get('dataset', {})
        self.data_root_dir = dataset_cfg.get('root_dir', '/home/najo/NAS/VLA/Insertion_VLA_Sim/Sim/collected_data_sim_6d_clean/collected_data_merged')
        self.horizon = dataset_cfg.get('horizon', 1)
        self.use_qpos = dataset_cfg.get('use_qpos', False)
        self.use_ee_pose = dataset_cfg.get('use_ee_pose', True)
        self.use_ee_pose_delta_as_action = dataset_cfg.get('use_ee_pose_delta_as_action', False)
        self.task_instruction = dataset_cfg.get('task_instruction', 'Insert the needle into the target point')
        self.camera_dropout_prob = dataset_cfg.get('camera_dropout_prob', 0.0)
        self.min_cameras = dataset_cfg.get('min_cameras', 1)

        # Normalization configuration
        self.normalization_stats_path = dataset_cfg.get('normalization_stats_path', None)

        # Data augmentation
        self.augment = dataset_cfg.get('augment', False)
        self.augment_brightness = dataset_cfg.get('augment_brightness', 0.2)
        self.augment_contrast = dataset_cfg.get('augment_contrast', 0.2)
        self.augment_saturation = dataset_cfg.get('augment_saturation', 0.2)
        self.augment_hue = dataset_cfg.get('augment_hue', 0.05)
        self.augment_noise = dataset_cfg.get('augment_noise', 0.02)

        # Image resolution for NORA (Qwen2.5-VL uses dynamic resolution)
        self.resize_resolution = tuple(dataset_cfg.get('resize_resolution', [224, 224]))

        # Validation settings
        self.num_val_episodes = dataset_cfg.get('num_val_episodes', 0)
        self.val_frequency = config_dict.get('val_frequency', 1000)  # Validation every N steps

        # Logging and checkpointing
        self.wandb_project_name = config_dict.get('wandb_project_name', 'NORA VLA Training')
        self.checkpoint_save_frequency = config_dict.get('checkpoint_save_frequency', 5000)
        self.logging_frequency = config_dict.get('logging_frequency', 100)

        # DataLoader settings
        self.num_workers = config_dict.get('num_workers', 4)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'NoraTrainingConfig':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)


# --- 2. Data Loading ---
def find_hdf5_episodes(root_dir: str) -> List[str]:
    """
    Find all HDF5 episode files in the dataset directory.

    Args:
        root_dir: Root directory containing HDF5 files

    Returns:
        List of HDF5 file paths
    """
    root_path = Path(root_dir)

    # Search for HDF5 files
    h5_files = list(root_path.glob('**/*.h5'))
    h5_files = sorted([str(f) for f in h5_files])

    try:
        logger.info(f"Found {len(h5_files)} HDF5 episode files in {root_dir}")
        if len(h5_files) == 0:
            logger.warning(f"No HDF5 files found in {root_dir}")
            logger.info(f"Make sure your dataset directory contains .h5 files")
    except RuntimeError:
        # Accelerate not initialized, use print
        print(f"   ✓ Found {len(h5_files)} HDF5 files")
        if len(h5_files) == 0:
            print(f"   ✗ No HDF5 files found in {root_dir}")
            print(f"   Make sure your dataset directory contains .h5 files")

    return h5_files


def create_datasets(config: NoraTrainingConfig) -> tuple[Dataset, Optional[Dataset]]:
    """
    Create train and validation HDF5 datasets.

    Args:
        config: Training configuration

    Returns:
        Tuple of (train_dataset, val_dataset)
        val_dataset will be None if num_val_episodes is 0
    """
    # Load normalization statistics if provided
    normalizer = None
    if config.normalization_stats_path:
        try:
            stats = load_stats(config.normalization_stats_path)
            normalizer = Normalizer(stats)
            print(f"   ✓ Loaded normalization statistics from {config.normalization_stats_path}")
        except Exception as e:
            print(f"   ✗ Failed to load normalization statistics: {e}")
            print(f"   Proceeding without normalization...")

    # Find all HDF5 episodes
    hdf5_paths = find_hdf5_episodes(config.data_root_dir)

    if len(hdf5_paths) == 0:
        raise ValueError(f"No HDF5 files found in {config.data_root_dir}")

    # Split into train and validation
    num_val = config.num_val_episodes
    if num_val > 0:
        if num_val >= len(hdf5_paths):
            raise ValueError(f"num_val_episodes ({num_val}) must be less than total episodes ({len(hdf5_paths)})")

        # Use last N episodes for validation
        train_paths = hdf5_paths[:-num_val]
        val_paths = hdf5_paths[-num_val:]

        try:
            print(f"   Split: {len(train_paths)} train episodes, {len(val_paths)} val episodes")
        except RuntimeError:
            pass
    else:
        train_paths = hdf5_paths
        val_paths = []

    # Create training dataset
    train_dataset = create_hdf5_lerobot_dataset(
        hdf5_paths=train_paths,
        horizon=config.horizon,
        n_obs_steps=1,  # NORA uses single observation
        use_qpos=config.use_qpos,
        use_ee_pose=config.use_ee_pose,
        use_ee_pose_delta_as_action=config.use_ee_pose_delta_as_action,
        task_instruction=config.task_instruction,
        camera_dropout_prob=config.camera_dropout_prob,
        min_cameras=config.min_cameras,
        augment=config.augment,
        augment_brightness=config.augment_brightness,
        augment_contrast=config.augment_contrast,
        augment_saturation=config.augment_saturation,
        augment_hue=config.augment_hue,
        augment_noise=config.augment_noise,
        squeeze_n_obs_steps=True,  # Squeeze temporal dimension for NORA
        tokenizer=None,  # Will be handled by NORA processor
        tokenizer_max_length=48,
        normalizer=normalizer,
    )

    # Create validation dataset (no augmentation)
    val_dataset = None
    if len(val_paths) > 0:
        val_dataset = create_hdf5_lerobot_dataset(
            hdf5_paths=val_paths,
            horizon=config.horizon,
            n_obs_steps=1,
            use_qpos=config.use_qpos,
            use_ee_pose=config.use_ee_pose,
            use_ee_pose_delta_as_action=config.use_ee_pose_delta_as_action,
            task_instruction=config.task_instruction,
            camera_dropout_prob=0.0,  # No dropout for validation
            min_cameras=1,
            augment=False,  # No augmentation for validation
            squeeze_n_obs_steps=True,
            tokenizer=None,
            tokenizer_max_length=48,
            normalizer=normalizer,
        )

    return train_dataset, val_dataset


# --- 3. Data Processing for NORA ---
def map_fast_token_to_vlm_action(tokens: List[str]) -> str:
    """
    Maps FAST action tokens to VLM action format.
    Action token 0 is mapped to <robot_action_0>, etc.
    """
    return ''.join([f"<robot_action_{token}>" for token in tokens])


def process_example_for_nora(example: Dict[str, Any], processor: AutoProcessor, fast_tokenizer: AutoProcessor) -> Dict[str, Any]:
    """
    Process a single example from HDF5 dataset for NORA training.

    Args:
        example: Sample from HDF5LeRobotDataset with keys:
                 - observation.images.camera1/2/3: (C, H, W) tensors
                 - observation.state: (state_dim,) tensor
                 - action: (action_dim,) or (horizon, action_dim) tensor
                 - task: task instruction string
        processor: NORA processor (Qwen2.5-VL)
        fast_tokenizer: FAST tokenizer for actions

    Returns:
        Processed messages in NORA format
    """
    # Get action and tokenize with FAST
    action = example['action'].numpy()
    if len(action.shape) == 1:
        action = action[np.newaxis, :]  # (action_dim,) -> (1, action_dim)

    # Tokenize action with FAST
    fast_tokens = fast_tokenizer(action)
    vlm_action = map_fast_token_to_vlm_action(fast_tokens[0])

    # Get task instruction
    lang = example['task']

    # Get image - NORA uses camera1 by default (you can modify to use multiple cameras)
    # The HDF5 adapter provides images as (C, H, W) tensors in [0, 1] range
    pixel_values = example['observation.images.camera1']  # (C, H, W)

    # Convert to PIL Image format expected by Qwen processor
    # (C, H, W) -> (H, W, C) and scale to [0, 255]
    img_np = pixel_values.permute(1, 2, 0).numpy()  # (H, W, C)
    img_np = (img_np * 255).astype(np.uint8)
    pil_image = Image.fromarray(img_np)  # Convert numpy to PIL Image

    # Create messages in NORA format
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},  # Use PIL Image instead of numpy
                {"type": "text", "text": lang},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": vlm_action},
            ],
        },
    ]

    return messages


def collate_fn_nora(examples: List[Dict[str, Any]], processor: AutoProcessor, fast_tokenizer: AutoProcessor):
    """
    Collate function for NORA training with HDF5 datasets.

    Args:
        examples: List of samples from HDF5LeRobotDataset
        processor: NORA processor
        fast_tokenizer: FAST tokenizer

    Returns:
        Batched inputs for NORA model
    """
    # Process each example
    messages = [process_example_for_nora(example, processor, fast_tokenizer) for example in examples]

    # Apply chat template
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    # Process vision info
    image_inputs, video_inputs = process_vision_info(messages)

    # Create batch
    batch_input = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # Create labels (mask everything except action tokens)
    action_token_min = 151665
    action_token_max = 153712
    labels = batch_input['input_ids'].clone()

    for i in range(labels.size(0)):
        seq = labels[i]
        # Create mask for action tokens
        mask_seq = (seq >= action_token_min) & (seq <= action_token_max)
        nonzero_indices = torch.nonzero(mask_seq, as_tuple=False)

        if nonzero_indices.numel() > 0:
            first_action_index = nonzero_indices[0].item()
            # Mask out everything before first action token
            seq[:first_action_index] = -100
        else:
            # No action tokens found, mask entire sequence
            seq[:] = -100

    # Mask padding tokens
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch_input['labels'] = labels

    return batch_input


# --- 4. Model Initialization ---
def load_model_and_processor(config: NoraTrainingConfig, accelerator: Accelerator):
    """
    Load NORA model and processors.

    Returns:
        (model, processor, fast_tokenizer)
    """
    # Load NORA processor and model
    processor = AutoProcessor.from_pretrained('declare-lab/nora')
    processor.tokenizer.padding_side = 'left'

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        'declare-lab/nora',
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"  # Use PyTorch's SDPA instead of flash_attention_2 (GLIBC compatibility)
    )

    # Freeze Qwen VLM (vision encoder + language model), only train action decoder (lm_head)
    accelerator.print("Freezing Qwen VLM parameters (visual encoder + language model)...")

    # Freeze vision encoder
    if hasattr(model, 'visual'):
        for param in model.visual.parameters():
            param.requires_grad = False
        accelerator.print("  ✓ Vision encoder frozen")

    # Freeze language model backbone
    if hasattr(model, 'model'):
        for param in model.model.parameters():
            param.requires_grad = False
        accelerator.print("  ✓ Language model frozen")

    # Keep lm_head trainable (action decoder)
    if hasattr(model, 'lm_head'):
        for param in model.lm_head.parameters():
            param.requires_grad = True
        accelerator.print("  ✓ LM head (action decoder) trainable")

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    accelerator.print(f"  Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    # Load FAST tokenizer for actions
    fast_tokenizer = AutoProcessor.from_pretrained(
        "physical-intelligence/fast", trust_remote_code=True
    )

    # Load pretrained weights if specified
    if config.load_model_weights:
        tensors = {}
        from safetensors import safe_open
        with safe_open(config.load_model_weights, framework="pt") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        model.load_state_dict(tensors, strict=False)
        accelerator.print("Pretrained weights loaded.")

    return model, processor, fast_tokenizer


# --- 5. Training and Validation ---
def run_validation(model, val_dataloader, accelerator):
    """
    Run validation and return average loss.

    Args:
        model: The model to validate
        val_dataloader: Validation dataloader
        accelerator: Accelerator instance

    Returns:
        Average validation loss
    """
    model.eval()
    total_val_loss = 0.0
    num_val_batches = 0

    # Ensure all processes start validation at the same time
    accelerator.wait_for_everyone()

    with torch.no_grad():
        for batch in val_dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            total_val_loss += loss.detach().float().item()  # Convert to Python float immediately
            num_val_batches += 1

            # Clean up batch and outputs immediately
            del batch, outputs, loss

    # Average across batches on this process
    avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0.0

    # Use reduce for averaging across processes
    avg_val_loss_tensor = torch.tensor(avg_val_loss, device=accelerator.device)
    avg_val_loss_tensor = accelerator.reduce(avg_val_loss_tensor, reduction="mean")

    model.train()

    # Clean up validation memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return avg_val_loss_tensor.item()


def train(config: NoraTrainingConfig):
    """Main training loop for NORA with custom HDF5 datasets."""

    # Initialize accelerator
    accelerator = Accelerator(gradient_accumulation_steps=config.gradient_accumulation_steps)
    accelerator.dataloader_config.dispatch_batches = False
    logger.info(accelerator.state, main_process_only=False)

    # Initialize W&B
    if accelerator.is_main_process:
        wandb.init(project=config.wandb_project_name, config=vars(config))

    # Load model and processors
    model, processor, fast_tokenizer = load_model_and_processor(config, accelerator)

    # Create datasets (train and validation)
    with accelerator.main_process_first():
        train_dataset, val_dataset = create_datasets(config)

    # Create train DataLoader with custom collate function
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.per_device_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=lambda examples: collate_fn_nora(examples, processor, fast_tokenizer),
        pin_memory=False,  # FIXED: Disabled to reduce memory pressure with Accelerate
        persistent_workers=False,  # FIXED: Disabled to force worker cleanup and prevent memory leaks
        prefetch_factor=1 if config.num_workers > 0 else None,  # FIXED: Reduced from 2 to 1
    )

    # Create validation DataLoader if validation set exists
    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.per_device_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=lambda examples: collate_fn_nora(examples, processor, fast_tokenizer),
            pin_memory=False,  # FIXED: Disabled to reduce memory pressure with Accelerate
            persistent_workers=False,  # FIXED: Disabled to force worker cleanup and prevent memory leaks
            prefetch_factor=1 if config.num_workers > 0 else None,  # FIXED: Reduced from 2 to 1
        )

    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=1e-8,
        eps=1e-8,
    )

    # Initialize learning rate scheduler
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=config.num_warmup_steps,
        num_training_steps=config.max_train_steps
    )

    # Prepare everything with Accelerator
    if val_dataloader is not None:
        model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, val_dataloader
        )
    else:
        model, optimizer, train_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader
        )

    # Resume from checkpoint if provided
    if config.resume_from_checkpoint:
        accelerator.load_state(config.resume_from_checkpoint)
        accelerator.print(f"Resumed from checkpoint: {config.resume_from_checkpoint}")

    # Training info
    total_batch_size = config.per_device_batch_size * accelerator.num_processes * config.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num train examples = {len(train_dataset)}")
    if val_dataset is not None:
        logger.info(f"  Num val examples = {len(val_dataset)}")
        logger.info(f"  Validation frequency = every {config.val_frequency} steps")
    logger.info(f"  Num steps = {config.max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {config.per_device_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {config.max_train_steps}")

    # Training loop
    completed_steps = 0
    progress_bar = tqdm(
        range(config.max_train_steps),
        disable=not accelerator.is_local_main_process,
        desc="Training",
        dynamic_ncols=True
    )
    total_loss = 0.0  # Use Python float to avoid tensor accumulation
    recent_losses = []  # Track recent losses for smoothed display
    smoothing_window = 10  # Average over last N steps

    while completed_steps < config.max_train_steps:
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                outputs = model(**batch)
                loss = outputs.loss
                loss_value = loss.detach().float().item()  # Convert to Python float immediately
                total_loss += loss_value
                accelerator.backward(loss)

                # Clean up outputs immediately after backward
                del outputs

                # Gradient clipping
                if config.gradient_clipping is not None:
                    accelerator.clip_grad_norm_(model.parameters(), config.gradient_clipping)

                if accelerator.sync_gradients:
                    completed_steps += 1

                    # Update recent losses for smoothed display
                    recent_losses.append(loss_value)
                    if len(recent_losses) > smoothing_window:
                        recent_losses.pop(0)
                    avg_recent_loss = sum(recent_losses) / len(recent_losses)

                    # Update progress bar with current metrics
                    progress_bar.update(1)
                    progress_bar.set_postfix({
                        'loss': f'{loss_value:.4f}',
                        'avg_loss': f'{avg_recent_loss:.4f}',
                        'lr': f'{lr_scheduler.get_last_lr()[0]:.2e}'
                    })

                    # Run validation if enabled (MUST be outside main_process block!)
                    if val_dataloader is not None and completed_steps % config.val_frequency == 0:
                        # Ensure all processes are ready before validation
                        accelerator.wait_for_everyone()
                        val_loss = run_validation(model, val_dataloader, accelerator)
                        if accelerator.is_main_process:
                            logger.info(f"Step {completed_steps}, Val Loss: {val_loss:.4f}")
                            # Log validation loss to wandb immediately
                            wandb.log({"val_loss": val_loss}, step=completed_steps)

                    # Logging
                    if completed_steps % config.logging_frequency == 0:
                        if accelerator.is_main_process:
                            # Calculate gradient norm
                            total_norm = 0.0
                            for p in model.parameters():
                                if p.grad is not None:
                                    total_norm += p.grad.data.norm(2).item() ** 2
                            total_norm = total_norm ** 0.5

                            # Get memory statistics
                            mem_stats = get_memory_stats()

                            lr = lr_scheduler.get_last_lr()[0]
                            logger.info(
                                f"Step {completed_steps}, Loss: {loss_value:.4f}, Grad Norm: {total_norm:.4f}, LR: {lr:.6f}, "
                                f"RAM: {mem_stats['ram_mb']:.0f}MB ({mem_stats['ram_percent']:.1f}%), "
                                f"GPU: {mem_stats['gpu_mb']:.0f}MB ({mem_stats['gpu_percent']:.1f}%)"
                            )

                            log_dict = {
                                "train_loss": loss_value,
                                "grad_norm": total_norm,
                                "learning_rate": lr,
                                "memory/ram_mb": mem_stats['ram_mb'],
                                "memory/ram_percent": mem_stats['ram_percent'],
                                "memory/gpu_mb": mem_stats['gpu_mb'],
                                "memory/gpu_reserved_mb": mem_stats['gpu_reserved_mb'],
                                "memory/gpu_percent": mem_stats['gpu_percent'],
                            }

                            wandb.log(log_dict, step=completed_steps)

                    # Checkpointing
                    if completed_steps % config.checkpoint_save_frequency == 0 and completed_steps > 0:
                        # Ensure all processes are ready before checkpointing
                        accelerator.wait_for_everyone()

                        # 1. Save standard Accelerate state (for resuming)
                        checkpoint_dir = os.path.join(config.output_dir, f"steps_{completed_steps}")
                        accelerator.save_state(checkpoint_dir)

                        # 2. Save PyTorch checkpoint (single file, lightweight) - inspired by train_smolvla_sim.py
                        if accelerator.is_main_process:
                            checkpoint_path = os.path.join(config.output_dir, f"checkpoint_step_{completed_steps}.pt")
                            unwrapped_model = accelerator.unwrap_model(model)

                            torch.save({
                                "step": completed_steps,
                                "model_state_dict": unwrapped_model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "scheduler_state_dict": lr_scheduler.state_dict(),
                                "config": vars(config),
                            }, checkpoint_path)

                        if accelerator.is_main_process:
                            avg_loss = total_loss / config.checkpoint_save_frequency
                            summary_data = {"steps": completed_steps, "train_loss": avg_loss}

                            with open(os.path.join(config.output_dir, "summary.jsonl"), "a") as f:
                                f.write(json.dumps(summary_data) + "\n")

                            logger.info(f"Checkpoint saved at step {completed_steps}, Avg Loss: {avg_loss:.4f}")
                            total_loss = 0.0

                        # Clean up after checkpointing
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                optimizer.step()
                lr_scheduler.step()

                # Clean up batch and loss after optimizer step
                del batch, loss

                # Memory cleanup after gradient accumulation
                if accelerator.sync_gradients:
                    # More aggressive cleanup every step (since sync_gradients means real update happened)
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            if completed_steps >= config.max_train_steps:
                break

    # Save final checkpoint
    final_checkpoint_dir = os.path.join(config.output_dir, f"steps_{completed_steps}")
    accelerator.save_state(final_checkpoint_dir)

    if accelerator.is_main_process:
        logger.info(f"Training finished. Final checkpoint saved at {final_checkpoint_dir}")
        wandb.finish()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train NORA VLA with custom HDF5 datasets")
    parser.add_argument(
        "--config",
        type=str,
        default="train_config_nora.yaml",
        help="Path to training configuration YAML file"
    )
    args = parser.parse_args()

    # Load configuration
    config = NoraTrainingConfig.from_yaml(args.config)

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Run training
    train(config)


if __name__ == "__main__":
    main()
