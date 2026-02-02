"""
Normalization utilities for SmolVLA training and inference.

This module provides normalization functions compatible with LeRobot's
MEAN_STD normalization approach.
"""

import torch
import numpy as np
from typing import Dict, Any, Union


class Normalizer:
    """
    Normalizes and unnormalizes data using mean/std (z-score) normalization.

    Compatible with LeRobot's NormalizerProcessorStep with MEAN_STD mode.
    """

    def __init__(self, stats: Dict[str, Dict[str, Any]], eps: float = 1e-8):
        """
        Args:
            stats: Dictionary with keys like 'action', 'observation.state'
                   Each value is a dict with 'mean' and 'std' arrays/lists
            eps: Small epsilon to prevent division by zero
        """
        self.stats = stats
        self.eps = eps

        # Convert to tensors for efficiency
        self.tensor_stats = {}
        for key, stat_dict in stats.items():
            self.tensor_stats[key] = {
                'mean': torch.tensor(stat_dict['mean'], dtype=torch.float32),
                'std': torch.tensor(stat_dict['std'], dtype=torch.float32),
            }

    def to(self, device: Union[str, torch.device]):
        """Move normalizer to device."""
        for key in self.tensor_stats:
            for stat_name in ['mean', 'std']:
                self.tensor_stats[key][stat_name] = self.tensor_stats[key][stat_name].to(device)
        return self

    def normalize(self, data: torch.Tensor, key: str) -> torch.Tensor:
        """
        Normalize data: (x - mean) / std

        Args:
            data: Input tensor to normalize
            key: Key in stats dict (e.g., 'action', 'observation.state')

        Returns:
            Normalized tensor
        """
        if key not in self.tensor_stats:
            return data

        mean = self.tensor_stats[key]['mean'].to(data.device)
        std = self.tensor_stats[key]['std'].to(data.device)

        # Prevent division by zero
        std_safe = torch.clamp(std, min=self.eps)

        # Add better error message if shape mismatch
        try:
            result = (data - mean) / std_safe
        except RuntimeError as e:
            error_msg = (
                f"\nNormalization shape mismatch for key='{key}':\n"
                f"  data.shape: {data.shape}\n"
                f"  mean.shape: {mean.shape}\n"
                f"  std.shape: {std.shape}\n"
                f"  Original error: {e}\n"
            )
            raise RuntimeError(error_msg) from e

        return result

    def unnormalize(self, data: torch.Tensor, key: str) -> torch.Tensor:
        """
        Unnormalize data: x * std + mean

        Args:
            data: Normalized tensor
            key: Key in stats dict (e.g., 'action', 'observation.state')

        Returns:
            Unnormalized tensor (original scale)
        """
        if key not in self.tensor_stats:
            return data

        mean = self.tensor_stats[key]['mean'].to(data.device)
        std = self.tensor_stats[key]['std'].to(data.device)

        return data * std + mean


def compute_dataset_stats(
    dataset_root: str,
    max_episodes: int = None,
    use_ee_pose_delta: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute normalization statistics from HDF5 dataset.

    Args:
        dataset_root: Path to dataset root directory
        max_episodes: Maximum number of episodes to load (None = all)
        use_ee_pose_delta: If True, compute actions from ee_pose delta.

    Returns:
        Dictionary with statistics for 'action' and 'observation.state'
    """
    import h5py
    from pathlib import Path
    from tqdm import tqdm

    dataset_path = Path(dataset_root)
    episode_files = sorted(dataset_path.rglob("*.h5"))

    if max_episodes is not None:
        episode_files = episode_files[:max_episodes]

    print(f"Computing statistics from {len(episode_files)} episodes...")
    if use_ee_pose_delta:
        print("INFO: Using EE pose delta as action for statistics computation.")

    all_actions = []
    all_states = []

    for ep_path in tqdm(episode_files, desc="Loading episodes"):
        try:
            with h5py.File(ep_path, 'r') as f:
                ee_pose = f['observations']['ee_pose'][:]

                if use_ee_pose_delta:
                    # Calculate action from the delta of ee_pose
                    if len(ee_pose) > 1:
                        actions = ee_pose[1:] - ee_pose[:-1]
                        all_actions.append(actions)
                else:
                    # Use the recorded action
                    actions = f['action'][:]
                    all_actions.append(actions)

                all_states.append(ee_pose)
        except Exception as e:
            print(f"Warning: Failed to load {ep_path.name}: {e}")
            continue

    if len(all_actions) == 0:
        raise ValueError("No episodes loaded successfully")

    all_actions = np.concatenate(all_actions, axis=0)
    all_states = np.concatenate(all_states, axis=0)

    # Compute statistics
    action_mean = all_actions.mean(axis=0)
    action_std = all_actions.std(axis=0)
    state_mean = all_states.mean(axis=0)
    state_std = all_states.std(axis=0)

    # Prevent zero std (for dimensions with no variance)
    action_std = np.where(action_std < 1e-6, 1.0, action_std)
    state_std = np.where(state_std < 1e-6, 1.0, state_std)

    stats = {
        'action': {
            'mean': action_mean.tolist(),
            'std': action_std.tolist(),
            'min': all_actions.min(axis=0).tolist(),
            'max': all_actions.max(axis=0).tolist(),
        },
        'observation.state': {
            'mean': state_mean.tolist(),
            'std': state_std.tolist(),
            'min': all_states.min(axis=0).tolist(),
            'max': all_states.max(axis=0).tolist(),
        }
    }

    print("\n" + "="*80)
    print("COMPUTED STATISTICS")
    print("="*80)
    print(f"Action mean: {action_mean}")
    print(f"Action std:  {action_std}")
    print(f"State mean:  {state_mean}")
    print(f"State std:   {state_std}")
    print("="*80 + "\n")

    return stats


def save_stats(stats: Dict[str, Dict[str, Any]], save_path: str):
    """Save statistics to YAML file."""
    import yaml
    from pathlib import Path

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w') as f:
        yaml.dump(stats, f, default_flow_style=False)

    print(f"Statistics saved to: {save_path}")


def load_stats(stats_path: str) -> Dict[str, Dict[str, Any]]:
    """Load statistics from YAML file."""
    import yaml
    from pathlib import Path

    stats_path = Path(stats_path)
    if not stats_path.exists():
        raise FileNotFoundError(f"Statistics file not found: {stats_path}")

    with open(stats_path, 'r') as f:
        stats = yaml.safe_load(f)

    return stats
