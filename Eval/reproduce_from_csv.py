#!/usr/bin/env python3
"""
Reproduce trajectory from CSV evaluation results in MuJoCo simulation.

This script reads frame_by_frame_comparison.csv from evaluation and
reproduces the trajectory using either GT or Predicted actions.

Usage:
python3 reproduce_from_csv.py \
    --csv /path/to/frame_by_frame_comparison.csv \
    --mode pred \
    --output reproduced_from_csv.h5
"""

import os
os.environ['MUJOCO_GL'] = 'egl'

import mujoco
import numpy as np
import cv2
import h5py
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# === Configuration ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "Sim")
MODEL_PATH = os.path.join(SIM_DIR, "meca_add.xml")
IMG_WIDTH = 480
IMG_HEIGHT = 480

# Default home position (degrees)
HOME_QPOS = np.array([30, -20, 20, 0, 30, 60], dtype=np.float32)


class ReplayRecorder:
    """Records simulation replay data to HDF5 format."""

    def __init__(self, output_path):
        self.output_path = output_path
        self.buffer = []

    def add(self, frames, qpos, ee_pose, action, frame_idx, sensor_dist):
        self.buffer.append({
            "frame": frame_idx,
            "imgs": frames,
            "q": qpos,
            "p": ee_pose,
            "act": action,
            "sd": sensor_dist
        })

    def save(self):
        if not self.buffer:
            print("No data to save.")
            return

        print(f"💾 Saving reproduced data to {self.output_path}...")
        try:
            with h5py.File(self.output_path, 'w') as f:
                obs = f.create_group("observations")
                img_grp = obs.create_group("images")

                q_data = np.array([x['q'] for x in self.buffer], dtype=np.float32)
                p_data = np.array([x['p'] for x in self.buffer], dtype=np.float32)
                act_data = np.array([x['act'] for x in self.buffer], dtype=np.float32)
                frame_data = np.array([x['frame'] for x in self.buffer], dtype=np.int32)
                sensor_data = np.array([x['sd'] for x in self.buffer], dtype=np.float32)

                obs.create_dataset("qpos", data=q_data, compression="gzip")
                obs.create_dataset("ee_pose", data=p_data, compression="gzip")
                obs.create_dataset("sensor_dist", data=sensor_data, compression="gzip")
                f.create_dataset("action", data=act_data, compression="gzip")
                f.create_dataset("frame", data=frame_data, compression="gzip")

                # Save images
                first_imgs = self.buffer[0]["imgs"]
                for cam_name in first_imgs.keys():
                    jpeg_list = []
                    for step in self.buffer:
                        img = step["imgs"][cam_name]
                        success, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        if success:
                            jpeg_list.append(buf.flatten())
                        else:
                            jpeg_list.append(np.zeros(1, dtype=np.uint8))

                    dt = h5py.special_dtype(vlen=np.dtype('uint8'))
                    dset = img_grp.create_dataset(cam_name, (len(jpeg_list),), dtype=dt)
                    for i, code in enumerate(jpeg_list):
                        dset[i] = code

            print("✅ Save complete.")
            print(f"   Saved {len(self.buffer)} frames")

        except Exception as e:
            print(f"❌ Save Failed: {e}")


# === Math Helpers ===
def euler_to_rot_mat(r, p, y):
    """Convert Euler angles to rotation matrix."""
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(r), -np.sin(r)],
        [0, np.sin(r), np.cos(r)]
    ])
    Ry = np.array([
        [np.cos(p), 0, np.sin(p)],
        [0, 1, 0],
        [-np.sin(p), 0, np.cos(p)]
    ])
    Rz = np.array([
        [np.cos(y), -np.sin(y), 0],
        [np.sin(y), np.cos(y), 0],
        [0, 0, 1]
    ])
    return Rz @ Ry @ Rx


def get_ee_pose_6d_scaled(data, link6_id):
    """Get end-effector pose in mm and radians."""
    if link6_id >= 0:
        pos = data.xpos[link6_id].copy() * 1000  # Convert to mm
        mat = data.xmat[link6_id].reshape(3, 3)
        sy = np.sqrt(mat[0, 0] ** 2 + mat[1, 0] ** 2)
        if sy > 1e-6:
            r = np.arctan2(mat[2, 1], mat[2, 2])
            p = np.arctan2(-mat[2, 0], sy)
            y = np.arctan2(mat[1, 0], mat[0, 0])
        else:
            r = np.arctan2(-mat[1, 2], mat[1, 1])
            p = np.arctan2(-mat[2, 0], sy)
            y = 0.0
        return np.concatenate([pos, [r, p, y]])
    return np.zeros(6, dtype=np.float32)


def solve_ik(model, data, target_pos_m, target_rot_mat, link_id, q_init, n_motors, max_iter=500):
    """
    Solve inverse kinematics to reach target pose.

    Returns:
        q_solution: Joint angles solution
        converged: True if IK converged
    """
    data.qpos[:n_motors] = q_init
    mujoco.mj_forward(model, data)

    pos_tol = 1e-6
    rot_tol = 1e-5

    for _ in range(max_iter):
        curr_pos = data.xpos[link_id]
        curr_mat = data.xmat[link_id].reshape(3, 3)
        err_pos = target_pos_m - curr_pos
        R_err = target_rot_mat @ curr_mat.T

        quat_err = np.zeros(4)
        mujoco.mju_mat2Quat(quat_err, R_err.flatten())
        if quat_err[0] >= 1.0:
            err_rot = np.zeros(3)
        else:
            theta = 2 * np.arccos(np.clip(quat_err[0], -1, 1))
            sin_half = np.sqrt(1 - quat_err[0]**2)
            if sin_half < 1e-6:
                err_rot = np.zeros(3)
            else:
                err_rot = theta * (quat_err[1:] / sin_half)

        if np.linalg.norm(err_pos) < pos_tol and np.linalg.norm(err_rot) < rot_tol:
            return data.qpos[:n_motors].copy(), True

        jac_pos = np.zeros((3, model.nv))
        jac_rot = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jac_pos, jac_rot, link_id)
        J = np.vstack([jac_pos[:, :n_motors], jac_rot[:, :n_motors]])
        err = np.concatenate([err_pos, err_rot])
        lamb = 1e-2
        dq = J.T @ np.linalg.solve(J @ J.T + lamb**2 * np.eye(6), err)
        data.qpos[:n_motors] += dq
        mujoco.mj_forward(model, data)

    return data.qpos[:n_motors].copy(), False


def calculate_sensor_distance(model, data, tip_id, back_id, link6_id):
    """Calculate distance from needle tip to phantom surface."""
    if tip_id < 0 or back_id < 0:
        return -1.0

    p_sensor = data.site_xpos[tip_id].copy()
    p_back = data.site_xpos[back_id].copy()
    needle_dir = (p_sensor - p_back)
    needle_dir /= (np.linalg.norm(needle_dir) + 1e-10)

    dist = mujoco.mj_ray(
        model, data, p_sensor, needle_dir,
        None, 1, link6_id, np.zeros(1, dtype=np.int32)
    )

    return dist * 1000.0 if dist >= 0 else -1.0


def load_actions_from_csv(csv_path, mode='pred'):
    """
    Load actions from CSV evaluation results.

    Args:
        csv_path: Path to frame_by_frame_comparison.csv
        mode: 'gt' for ground truth, 'pred' for predicted actions

    Returns:
        actions: numpy array of shape (N, 6) with delta actions
    """
    print(f"📖 Reading CSV from {csv_path}...")
    df = pd.read_csv(csv_path)

    if mode == 'gt':
        columns = ['GT_dx', 'GT_dy', 'GT_dz', 'GT_drx', 'GT_dry', 'GT_drz']
        print("   Using Ground Truth actions")
    elif mode == 'pred':
        columns = ['Pred_dx', 'Pred_dy', 'Pred_dz', 'Pred_drx', 'Pred_dry', 'Pred_drz']
        print("   Using Predicted actions")
    else:
        raise ValueError(f"Invalid mode: {mode}. Use 'gt' or 'pred'")

    # Check if columns exist
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in CSV: {missing_cols}")

    actions = df[columns].values
    print(f"   Loaded {len(actions)} actions")
    print(f"   Action range: [{actions.min():.6f}, {actions.max():.6f}]")

    return actions


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce trajectory from CSV evaluation results"
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to frame_by_frame_comparison.csv"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="pred",
        choices=["gt", "pred"],
        help="Action mode: 'gt' (ground truth) or 'pred' (predicted)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output HDF5 file path (default: reproduced_{mode}.h5)"
    )
    parser.add_argument(
        "--start_qpos",
        type=float,
        nargs=6,
        default=None,
        help="Initial joint positions in degrees (default: home position)"
    )

    args = parser.parse_args()

    # Validate paths
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"❌ Error: CSV file not found: {csv_path}")
        return

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file not found: {MODEL_PATH}")
        return

    # Set output path
    if args.output is None:
        output_dir = csv_path.parent
        args.output = str(output_dir / f"reproduced_{args.mode}.h5")

    print(f"\n{'='*80}")
    print(f"CSV Trajectory Reproduction")
    print(f"{'='*80}")
    print(f"Input CSV:    {csv_path}")
    print(f"Action mode:  {args.mode.upper()}")
    print(f"Output file:  {args.output}")
    print(f"Model:        {MODEL_PATH}")
    print(f"{'='*80}\n")

    # Load MuJoCo model
    print(f"🔄 Loading MuJoCo model...")
    try:
        model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=IMG_HEIGHT, width=IMG_WIDTH)
        print("   ✅ Model loaded successfully")
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        return

    # Get body/site IDs
    try:
        tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "needle_tip")
        back_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "needle_back")
        link6_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "6_Link")
        n_motors = model.nu
        print(f"   Link6 ID: {link6_id}, Motors: {n_motors}")
    except Exception as e:
        print(f"⚠️ Warning: Some IDs not found: {e}")
        link6_id = -1
        tip_id = -1
        back_id = -1
        n_motors = 6

    # Load actions from CSV
    try:
        actions = load_actions_from_csv(str(csv_path), mode=args.mode)
    except Exception as e:
        print(f"❌ Error loading actions from CSV: {e}")
        return

    # Initialize recorder
    recorder = ReplayRecorder(args.output)

    # Set initial position
    if args.start_qpos is not None:
        initial_qpos = np.array(args.start_qpos, dtype=np.float32)
        print(f"   Using custom start position: {initial_qpos}")
    else:
        initial_qpos = HOME_QPOS
        print(f"   Using home position: {initial_qpos}")

    # Initialize simulation
    print("\n🚀 Starting trajectory reproduction...")
    mujoco.mj_resetData(model, data)
    data.qpos[:n_motors] = np.deg2rad(initial_qpos)
    mujoco.mj_forward(model, data)
    current_q_guess = np.deg2rad(initial_qpos)

    # Record frame 0
    current_ee_pose = get_ee_pose_6d_scaled(data, link6_id)
    sensor_dist = calculate_sensor_distance(model, data, tip_id, back_id, link6_id)

    frames = {}
    for cam_name in ["side_camera", "tool_camera", "top_camera"]:
        renderer.update_scene(data, camera=cam_name)
        frames[cam_name] = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)

    recorder.add(
        frames,
        np.rad2deg(data.qpos[:n_motors]),
        current_ee_pose,
        actions[0],
        0,
        sensor_dist
    )

    # Execute trajectory
    ik_failures = 0
    for i in tqdm(range(1, len(actions)), desc=f"Reproducing ({args.mode})"):
        # Get delta action
        delta = actions[i]

        # Current actual pose in simulation
        current_sim_ee = get_ee_pose_6d_scaled(data, link6_id)

        # Target = Current + Delta (closed-loop control)
        target_ee = current_sim_ee + delta

        target_pos_m = target_ee[:3] / 1000.0  # Convert mm to m
        target_r, target_p, target_y = target_ee[3], target_ee[4], target_ee[5]
        target_rot_mat = euler_to_rot_mat(target_r, target_p, target_y)

        # Solve IK
        q_sol, converged = solve_ik(
            model, data, target_pos_m, target_rot_mat,
            link6_id, current_q_guess, n_motors
        )

        if not converged:
            ik_failures += 1

        current_q_guess = q_sol

        # Update simulation
        data.qpos[:n_motors] = q_sol
        mujoco.mj_forward(model, data)

        # Measure current state
        current_ee_pose = get_ee_pose_6d_scaled(data, link6_id)
        sensor_dist = calculate_sensor_distance(model, data, tip_id, back_id, link6_id)

        # Capture images
        frames = {}
        for cam_name in ["side_camera", "tool_camera", "top_camera"]:
            renderer.update_scene(data, camera=cam_name)
            frames[cam_name] = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)

        # Record
        recorder.add(
            frames,
            np.rad2deg(data.qpos[:n_motors]),
            current_ee_pose,
            actions[i],
            i,
            sensor_dist
        )

    # Save results
    recorder.save()

    # Print summary
    print(f"\n{'='*80}")
    print(f"Reproduction Summary")
    print(f"{'='*80}")
    print(f"Total frames:     {len(actions)}")
    print(f"IK failures:      {ik_failures} ({ik_failures/len(actions)*100:.2f}%)")
    print(f"Output saved to:  {args.output}")
    print(f"{'='*80}\n")

    print("✅ Reproduction complete!")


if __name__ == "__main__":
    main()
