#!/usr/bin/env python3
"""
Compare reproduced trajectories (GT vs Predicted) from HDF5 files.

This script visualizes the difference between ground truth and predicted
trajectories reproduced in simulation.

Usage:
python3 compare_reproductions.py \
    --gt reproduced_gt.h5 \
    --pred reproduced_pred.h5 \
    --output comparison_plot.png
"""

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path


def load_trajectory(h5_path):
    """Load trajectory data from HDF5 file."""
    with h5py.File(h5_path, 'r') as f:
        ee_pose = f['observations/ee_pose'][:]
        qpos = f['observations/qpos'][:]
        actions = f['action'][:]

    return {
        'ee_pose': ee_pose,
        'qpos': qpos,
        'actions': actions,
        'x': ee_pose[:, 0],
        'y': ee_pose[:, 1],
        'z': ee_pose[:, 2],
        'rx': ee_pose[:, 3],
        'ry': ee_pose[:, 4],
        'rz': ee_pose[:, 5],
    }


def plot_comparison(gt_data, pred_data, output_path):
    """Create comparison plots."""

    # Create figure
    fig = plt.figure(figsize=(20, 12))

    # 3D Trajectory
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.plot(gt_data['x'], gt_data['y'], gt_data['z'],
             'b-', linewidth=2, alpha=0.7, label='Ground Truth')
    ax1.plot(pred_data['x'], pred_data['y'], pred_data['z'],
             'r--', linewidth=2, alpha=0.7, label='Predicted')
    ax1.scatter(gt_data['x'][0], gt_data['y'][0], gt_data['z'][0],
                c='green', s=100, marker='o', label='Start')
    ax1.scatter(gt_data['x'][-1], gt_data['y'][-1], gt_data['z'][-1],
                c='red', s=100, marker='x', label='End')
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.set_title('3D Trajectory Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Top view (XY)
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(gt_data['x'], gt_data['y'], 'b-', linewidth=2, alpha=0.7, label='GT')
    ax2.plot(pred_data['x'], pred_data['y'], 'r--', linewidth=2, alpha=0.7, label='Pred')
    ax2.scatter(gt_data['x'][0], gt_data['y'][0], c='green', s=100, marker='o')
    ax2.scatter(gt_data['x'][-1], gt_data['y'][-1], c='red', s=100, marker='x')
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_title('Top View (XY)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')

    # Side view (XZ)
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(gt_data['x'], gt_data['z'], 'b-', linewidth=2, alpha=0.7, label='GT')
    ax3.plot(pred_data['x'], pred_data['z'], 'r--', linewidth=2, alpha=0.7, label='Pred')
    ax3.scatter(gt_data['x'][0], gt_data['z'][0], c='green', s=100, marker='o')
    ax3.scatter(gt_data['x'][-1], gt_data['z'][-1], c='red', s=100, marker='x')
    ax3.set_xlabel('X (mm)')
    ax3.set_ylabel('Z (mm)')
    ax3.set_title('Side View (XZ)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')

    # Position error over time
    ax4 = fig.add_subplot(2, 3, 4)
    pos_error = np.sqrt(
        (gt_data['x'] - pred_data['x'])**2 +
        (gt_data['y'] - pred_data['y'])**2 +
        (gt_data['z'] - pred_data['z'])**2
    )
    frames = np.arange(len(pos_error))
    ax4.plot(frames, pos_error, 'r-', linewidth=2)
    ax4.set_xlabel('Frame')
    ax4.set_ylabel('Position Error (mm)')
    ax4.set_title('Position Error Over Time')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=pos_error.mean(), color='b', linestyle='--',
                label=f'Mean: {pos_error.mean():.4f} mm')
    ax4.legend()

    # Per-dimension position error
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(frames, np.abs(gt_data['x'] - pred_data['x']),
             label='X error', linewidth=1.5)
    ax5.plot(frames, np.abs(gt_data['y'] - pred_data['y']),
             label='Y error', linewidth=1.5)
    ax5.plot(frames, np.abs(gt_data['z'] - pred_data['z']),
             label='Z error', linewidth=1.5)
    ax5.set_xlabel('Frame')
    ax5.set_ylabel('Absolute Error (mm)')
    ax5.set_title('Position Error by Dimension')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Rotation error
    ax6 = fig.add_subplot(2, 3, 6)
    rx_error = np.abs(gt_data['rx'] - pred_data['rx'])
    ry_error = np.abs(gt_data['ry'] - pred_data['ry'])
    rz_error = np.abs(gt_data['rz'] - pred_data['rz'])
    ax6.plot(frames, rx_error, label='RX error', linewidth=1.5)
    ax6.plot(frames, ry_error, label='RY error', linewidth=1.5)
    ax6.plot(frames, rz_error, label='RZ error', linewidth=1.5)
    ax6.set_xlabel('Frame')
    ax6.set_ylabel('Absolute Error (rad)')
    ax6.set_title('Rotation Error by Dimension')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.suptitle('Trajectory Comparison: Ground Truth vs Predicted', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Calculate and return statistics
    stats = {
        'position_error_mean': pos_error.mean(),
        'position_error_max': pos_error.max(),
        'position_error_final': pos_error[-1],
        'x_error_mean': np.abs(gt_data['x'] - pred_data['x']).mean(),
        'y_error_mean': np.abs(gt_data['y'] - pred_data['y']).mean(),
        'z_error_mean': np.abs(gt_data['z'] - pred_data['z']).mean(),
        'rx_error_mean': np.abs(gt_data['rx'] - pred_data['rx']).mean(),
        'ry_error_mean': np.abs(gt_data['ry'] - pred_data['ry']).mean(),
        'rz_error_mean': np.abs(gt_data['rz'] - pred_data['rz']).mean(),
    }

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Compare reproduced trajectories (GT vs Predicted)"
    )
    parser.add_argument(
        "--gt",
        type=str,
        required=True,
        help="Path to ground truth HDF5 file"
    )
    parser.add_argument(
        "--pred",
        type=str,
        required=True,
        help="Path to predicted HDF5 file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="trajectory_comparison.png",
        help="Output plot file path"
    )

    args = parser.parse_args()

    # Validate paths
    gt_path = Path(args.gt)
    pred_path = Path(args.pred)

    if not gt_path.exists():
        print(f"❌ Error: GT file not found: {gt_path}")
        return

    if not pred_path.exists():
        print(f"❌ Error: Predicted file not found: {pred_path}")
        return

    print(f"\n{'='*80}")
    print(f"Trajectory Comparison")
    print(f"{'='*80}")
    print(f"Ground Truth:  {gt_path}")
    print(f"Predicted:     {pred_path}")
    print(f"Output:        {args.output}")
    print(f"{'='*80}\n")

    # Load data
    print("📖 Loading trajectories...")
    gt_data = load_trajectory(str(gt_path))
    pred_data = load_trajectory(str(pred_path))

    print(f"   GT frames:   {len(gt_data['x'])}")
    print(f"   Pred frames: {len(pred_data['x'])}")

    # Check length match
    if len(gt_data['x']) != len(pred_data['x']):
        print("⚠️ Warning: Trajectories have different lengths!")
        min_len = min(len(gt_data['x']), len(pred_data['x']))
        print(f"   Using first {min_len} frames for comparison")
        for key in gt_data.keys():
            gt_data[key] = gt_data[key][:min_len]
            pred_data[key] = pred_data[key][:min_len]

    # Create comparison plot
    print("\n📊 Creating comparison plots...")
    stats = plot_comparison(gt_data, pred_data, args.output)

    # Print statistics
    print(f"\n{'='*80}")
    print(f"Trajectory Statistics")
    print(f"{'='*80}")
    print(f"Position Errors:")
    print(f"  Mean:      {stats['position_error_mean']:.4f} mm")
    print(f"  Max:       {stats['position_error_max']:.4f} mm")
    print(f"  Final:     {stats['position_error_final']:.4f} mm")
    print(f"\nPer-Dimension Position Errors (Mean):")
    print(f"  X:         {stats['x_error_mean']:.4f} mm")
    print(f"  Y:         {stats['y_error_mean']:.4f} mm")
    print(f"  Z:         {stats['z_error_mean']:.4f} mm")
    print(f"\nRotation Errors (Mean):")
    print(f"  RX:        {stats['rx_error_mean']:.6f} rad")
    print(f"  RY:        {stats['ry_error_mean']:.6f} rad")
    print(f"  RZ:        {stats['rz_error_mean']:.6f} rad")
    print(f"{'='*80}\n")

    print(f"✅ Comparison plot saved to: {args.output}")


if __name__ == "__main__":
    main()
