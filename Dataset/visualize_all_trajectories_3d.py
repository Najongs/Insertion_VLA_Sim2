#!/usr/bin/env python3
"""

python visualize_all_trajectories_3d.py --dataset_path "/home/najo/NAS/VLA/Insertion_VLA_Sim2/Dataset/Insert_time_diff"

Visualize all trajectories from Eye_trocar dataset in 3D space
Shows all trajectories with their start and end points marked
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import glob
from tqdm import tqdm

def load_trajectory(h5_file_path):
    """Load trajectory data from h5 file"""
    try:
        with h5py.File(h5_file_path, 'r') as f:
            # Get end-effector pose from observations
            # ee_pose contains [x, y, z, roll, pitch, yaw] in absolute coordinates
            ee_pose = f['observations']['ee_pose'][:]
            # Extract position (first 3 dimensions)
            position = ee_pose[:, :3]

            # Convert to mm if values are in meters (< 10 means likely in meters)
            if np.abs(position).max() < 10:
                position = position * 1000.0  # Convert m to mm

            return position
    except Exception as e:
        print(f"Error loading {h5_file_path}: {e}")
        return None

def visualize_all_trajectories(dataset_path, output_path=None, max_trajectories=None):
    """
    Visualize all trajectories in 3D space

    Args:
        dataset_path: Path to Eye_trocar dataset
        output_path: Path to save the visualization (optional)
        max_trajectories: Maximum number of trajectories to visualize (for testing)
    """
    # Find all h5 files
    h5_files = glob.glob(str(Path(dataset_path) / "**/*.h5"), recursive=True)

    if max_trajectories:
        h5_files = h5_files[:max_trajectories]

    print(f"Found {len(h5_files)} trajectory files")

    # Load all trajectories and group by folder (date/session)
    trajectories = []
    start_points = []
    end_points = []
    folder_labels = []  # Track which folder each trajectory belongs to

    print("Loading trajectories...")
    for h5_file in tqdm(h5_files):
        position = load_trajectory(h5_file)
        if position is not None and len(position) > 0:
            trajectories.append(position)
            start_points.append(position[0])
            end_points.append(position[-1])

            # Extract folder name hierarchy
            # For real data: date/person (e.g., "260106/1_MIN")
            # For sim data: just use single folder
            path_parts = Path(h5_file).parts

            # Check if this is real or sim data
            is_real_data = 'Eye_trocar_sim' not in h5_file

            if is_real_data:
                # Real data: find person folder only (ignore date)
                person_folder = None

                for i, part in enumerate(path_parts):
                    # Check if this is a person folder (e.g., 1_MIN, 2_JYT)
                    if '_' in part and part[0].isdigit():
                        person_folder = part
                        break

                # If person folder found, use it; otherwise treat as 1_KTY
                if person_folder:
                    folder_name = person_folder
                else:
                    # All "기타" episodes go to 1_KTY
                    folder_name = "1_KTY"
            else:
                # Sim data: just use a simple label
                folder_name = "Simulation"

            folder_labels.append(folder_name)

    print(f"Successfully loaded {len(trajectories)} trajectories")

    if len(trajectories) == 0:
        print("No valid trajectories found!")
        return

    # Convert to numpy arrays
    start_points = np.array(start_points)
    end_points = np.array(end_points)

    # Create folder to color mapping
    unique_folders = sorted(list(set(folder_labels)))
    folder_to_color = {folder: i for i, folder in enumerate(unique_folders)}

    # Print summary with colors
    print(f"\nTotal Contributions by Worker:")
    print("=" * 80)

    # Count contributions per worker
    worker_counts = {}
    for folder in unique_folders:
        worker_counts[folder] = folder_labels.count(folder)

    # Sort by contribution count (descending)
    sorted_workers = sorted(worker_counts.items(), key=lambda x: x[1], reverse=True)

    # Print worker contributions
    for worker, count in sorted_workers:
        color_idx = folder_to_color[worker]
        print(f"  👤 {worker:30s}: {count:4d} episodes (Color #{color_idx})")

    print("=" * 80)
    print(f"Total: {len(trajectories)} trajectories across {len(unique_folders)} workers")

    # Create 3D visualization with legend
    fig = plt.figure(figsize=(20, 12))

    # Main 3D plot
    ax1 = fig.add_subplot(221, projection='3d')

    # Plot all trajectories colored by folder and collect for legend
    print("Plotting trajectories...")
    plotted_folders = {}
    for i, traj in enumerate(trajectories):
        # Use colormap based on folder
        folder_name = folder_labels[i]
        folder_idx = folder_to_color[folder_name]
        color = plt.cm.tab20(folder_idx / len(unique_folders))

        # Plot trajectory
        line, = ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                alpha=0.4, linewidth=0.5, color=color)

        # Save one line per folder for legend
        if folder_name not in plotted_folders:
            plotted_folders[folder_name] = line

    # Plot start points (green) - 시작점
    start_scatter = ax1.scatter(start_points[:, 0], start_points[:, 1], start_points[:, 2],
               c='green', marker='o', s=50, alpha=0.8, edgecolors='darkgreen', linewidths=1)

    # Plot end points (red) - 끝점
    end_scatter = ax1.scatter(end_points[:, 0], end_points[:, 1], end_points[:, 2],
               c='red', marker='x', s=50, alpha=0.8, linewidths=2)

    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.set_title(f'All Trajectories 3D View (n={len(trajectories)})')

    # Create legend with folder counts (only for first plot to avoid clutter)
    legend_elements = [start_scatter, end_scatter]
    legend_labels = ['Start Points', 'End Points']

    # Add up to 10 most common folders to legend
    folder_counts = [(folder, folder_labels.count(folder)) for folder in unique_folders]
    folder_counts.sort(key=lambda x: x[1], reverse=True)
    for folder, count in folder_counts[:10]:
        if folder in plotted_folders:
            legend_elements.append(plotted_folders[folder])
            legend_labels.append(f'{folder} (n={count})')

    ax1.legend(legend_elements, legend_labels, loc='upper left', fontsize=8)
    ax1.grid(True)

    # XY projection (Top View)
    ax2 = fig.add_subplot(222)
    for i, traj in enumerate(trajectories):
        folder_idx = folder_to_color[folder_labels[i]]
        color = plt.cm.tab20(folder_idx / len(unique_folders))
        ax2.plot(traj[:, 0], traj[:, 1], alpha=0.4, linewidth=0.5, color=color)
    ax2.scatter(start_points[:, 0], start_points[:, 1], c='green', marker='o', s=30, alpha=0.8,
                label='Start', edgecolors='darkgreen', linewidths=1)
    ax2.scatter(end_points[:, 0], end_points[:, 1], c='red', marker='x', s=30, alpha=0.8,
                label='End', linewidths=2)
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_title('XY Projection (Top View)')
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')

    # XZ projection (Front View)
    ax3 = fig.add_subplot(223)
    for i, traj in enumerate(trajectories):
        folder_idx = folder_to_color[folder_labels[i]]
        color = plt.cm.tab20(folder_idx / len(unique_folders))
        ax3.plot(traj[:, 0], traj[:, 2], alpha=0.4, linewidth=0.5, color=color)
    ax3.scatter(start_points[:, 0], start_points[:, 2], c='green', marker='o', s=30, alpha=0.8,
                label='Start', edgecolors='darkgreen', linewidths=1)
    ax3.scatter(end_points[:, 0], end_points[:, 2], c='red', marker='x', s=30, alpha=0.8,
                label='End', linewidths=2)
    ax3.set_xlabel('X (mm)')
    ax3.set_ylabel('Z (mm)')
    ax3.set_title('XZ Projection (Front View)')
    ax3.legend()
    ax3.grid(True)
    ax3.axis('equal')

    # YZ projection (Side View)
    ax4 = fig.add_subplot(224)
    for i, traj in enumerate(trajectories):
        folder_idx = folder_to_color[folder_labels[i]]
        color = plt.cm.tab20(folder_idx / len(unique_folders))
        ax4.plot(traj[:, 1], traj[:, 2], alpha=0.4, linewidth=0.5, color=color)
    ax4.scatter(start_points[:, 1], start_points[:, 2], c='green', marker='o', s=30, alpha=0.8,
                label='Start', edgecolors='darkgreen', linewidths=1)
    ax4.scatter(end_points[:, 1], end_points[:, 2], c='red', marker='x', s=30, alpha=0.8,
                label='End', linewidths=2)
    ax4.set_xlabel('Y (mm)')
    ax4.set_ylabel('Z (mm)')
    ax4.set_title('YZ Projection (Side View)')
    ax4.legend()
    ax4.grid(True)
    ax4.axis('equal')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")

    plt.show()

    # Print statistics
    print("\n=== Trajectory Statistics ===")
    print(f"Total trajectories: {len(trajectories)}")
    print(f"\nStart Points Statistics:")
    print(f"  X: min={start_points[:, 0].min():.2f}, max={start_points[:, 0].max():.2f}, mean={start_points[:, 0].mean():.2f}")
    print(f"  Y: min={start_points[:, 1].min():.2f}, max={start_points[:, 1].max():.2f}, mean={start_points[:, 1].mean():.2f}")
    print(f"  Z: min={start_points[:, 2].min():.2f}, max={start_points[:, 2].max():.2f}, mean={start_points[:, 2].mean():.2f}")
    print(f"\nEnd Points Statistics:")
    print(f"  X: min={end_points[:, 0].min():.2f}, max={end_points[:, 0].max():.2f}, mean={end_points[:, 0].mean():.2f}")
    print(f"  Y: min={end_points[:, 1].min():.2f}, max={end_points[:, 1].max():.2f}, mean={end_points[:, 1].mean():.2f}")
    print(f"  Z: min={end_points[:, 2].min():.2f}, max={end_points[:, 2].max():.2f}, mean={end_points[:, 2].mean():.2f}")

    # Calculate average trajectory length
    traj_lengths = [len(traj) for traj in trajectories]
    print(f"\nTrajectory Length Statistics:")
    print(f"  Min: {min(traj_lengths)} steps")
    print(f"  Max: {max(traj_lengths)} steps")
    print(f"  Mean: {np.mean(traj_lengths):.1f} steps")
    print(f"  Median: {np.median(traj_lengths):.1f} steps")

    return trajectories, start_points, end_points

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Visualize all trajectories in 3D')
    parser.add_argument('--dataset_path', type=str,
                       default='/home/najo/NAS/VLA/dataset/New_dataset/collected_data/Eye_trocar/Eye_trocar',
                       help='Path to Eye_trocar dataset')
    parser.add_argument('--output', type=str, default='all_trajectories_3d.png',
                       help='Output image path')
    parser.add_argument('--max_trajectories', type=int, default=None,
                       help='Maximum number of trajectories to visualize (for testing)')

    args = parser.parse_args()

    visualize_all_trajectories(args.dataset_path, args.output, args.max_trajectories)

