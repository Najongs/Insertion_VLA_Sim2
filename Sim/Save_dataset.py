"""
python Save_dataset.py --no-randomize-phantom-pos
"""


import os
os.environ['MUJOCO_GL'] = 'egl'

import mujoco
import numpy as np
import cv2
import time
import h5py
import datetime
import threading
import pathlib
import argparse
from collections import deque

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, total=None: x

# === Configuration ===
MODEL_PATH = "meca_add.xml"
SAVE_DIR = "collected_data_sim_clean"
MAX_EPISODES = 1   
IMG_WIDTH = 640
IMG_HEIGHT = 480
TARGET_INSERTION_DEPTH = 0.0275
TRAJ_DURATION = 15.0

# === Recorder Class (수정됨: sensor_dist 저장 로직 추가) ===
class SimRecorder:
    def __init__(self, output_dir):
        self.out = pathlib.Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.buffer = []
        self.recording = False
        self.is_saving = False
        self.save_threads = []  # 저장 스레드 추적

    def start(self):
        # 이전 저장이 진행 중이어도 새 에피소드 시작 가능
        # (각 에피소드는 독립적인 버퍼 사용)
        self.buffer = []
        self.recording = True

    def add(self, frames, qpos, ee_pose, action, timestamp, phase, sensor_dist):
        if not self.recording: return
        self.buffer.append({
            "ts": timestamp,
            "imgs": frames,
            "q": qpos,
            "p": ee_pose,
            "act": action,
            "phase": phase,
            "sd": sensor_dist  # 추가됨
        })

    def save_async(self):
        if not self.buffer: return
        data_snapshot = self.buffer
        self.buffer = []
        self.recording = False
        self.is_saving = True

        def worker(data, filename):
            try:
                with h5py.File(filename, 'w') as f:
                    obs = f.create_group("observations")
                    img_grp = obs.create_group("images")

                    q_data = np.array([x['q'] for x in data], dtype=np.float32)
                    p_data = np.array([x['p'] for x in data], dtype=np.float32)
                    act_data = np.array([x['act'] for x in data], dtype=np.float32)
                    ts_data = np.array([x['ts'] for x in data], dtype=np.float32)
                    phase_data = np.array([x['phase'] for x in data], dtype=np.int32)
                    sensor_data = np.array([x['sd'] for x in data], dtype=np.float32) # 추가됨

                    obs.create_dataset("qpos", data=q_data, compression="gzip")
                    obs.create_dataset("ee_pose", data=p_data, compression="gzip")
                    obs.create_dataset("sensor_dist", data=sensor_data, compression="gzip") # 추가됨
                    f.create_dataset("action", data=act_data, compression="gzip")
                    f.create_dataset("timestamp", data=ts_data, compression="gzip")
                    f.create_dataset("phase", data=phase_data, compression="gzip")

                    first_imgs = data[0]["imgs"]
                    for cam_name in first_imgs.keys():
                        jpeg_list = []
                        for step in data:
                            img = step["imgs"][cam_name]
                            success, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                            if success: jpeg_list.append(buf.flatten())
                            else: jpeg_list.append(np.zeros(1, dtype=np.uint8))

                        dt = h5py.special_dtype(vlen=np.dtype('uint8'))
                        dset = img_grp.create_dataset(cam_name, (len(jpeg_list),), dtype=dt)
                        for i, code in enumerate(jpeg_list): dset[i] = code
                        
            except Exception as e:
                print(f"❌ Save Failed: {e}")
            finally:
                self.is_saving = False

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = self.out / f"episode_{timestamp}.h5"
        t = threading.Thread(target=worker, args=(data_snapshot, fname))
        t.start()
        self.save_threads.append(t)  # 스레드 추적

    def discard(self):
        self.buffer = []
        self.recording = False

    def wait_for_all(self):
        """모든 저장 스레드가 완료될 때까지 대기"""
        if self.save_threads:
            print(f"\n⏳ Waiting for {len(self.save_threads)} files to finish saving...")
            for t in self.save_threads:
                t.join()
            print("✅ All files saved successfully!")
            self.save_threads = []

# === Helper Functions ===
def smooth_step(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)

def randomize_phantom_pos(model, data, phantom_id, rot_id):
    # 1. 위치 이동 (Translation)
    offset_x = np.random.uniform(-0.05, 0.0)
    offset_y = np.random.uniform(0.0, 0.03)
    offset_z = 0.0 # np.random.uniform(0.0, 0.1) 
    model.body_pos[phantom_id] = np.array([offset_x, offset_y, offset_z])
    
    # 2. 회전 (Rotation)
    random_angle_deg = np.random.uniform(-15, 15)
    new_quat = np.zeros(4)
    mujoco.mju_euler2Quat(new_quat, [0, 0, np.deg2rad(random_angle_deg)], "xyz")
    model.body_quat[rot_id] = new_quat
    print(f">>> Randomize: Pos=({offset_x:.2f}, {offset_y:.2f}), Angle={random_angle_deg:.1f} deg")
    mujoco.mj_forward(model, data)

# === Args ===
def _parse_args():
    parser = argparse.ArgumentParser(description="Record simulation dataset.")
    parser.add_argument(
        "--randomize-phantom-pos",
        dest="randomize_phantom_pos",
        action="store_true",
        default=True,
        help="Enable phantom position randomization.",
    )
    parser.add_argument(
        "--no-randomize-phantom-pos",
        dest="randomize_phantom_pos",
        action="store_false",
        help="Disable phantom position randomization.",
    )
    return parser.parse_args()

# === Main Script ===
def main():
    args = _parse_args()
    print(f"🔄 Loading Model: {MODEL_PATH}")
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=IMG_HEIGHT, width=IMG_WIDTH)
    
    try:
        tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "needle_tip")
        back_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "needle_back")
        target_entry_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "trocar_target")
        target_depth_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "trocar_depth")
        phantom_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "phantom_assembly")
        rotating_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rotating_assembly")
        link6_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "6_Link") 
        n_motors = model.nu
        dof = model.nv
    except Exception as e:
        print(f"⚠️ Warning: Some IDs not found: {e}")
        phantom_body_id = -1

    recorder = SimRecorder(SAVE_DIR)
    home_pose = np.array([0.5236, -0.3491, 0.3491, 0.0000, 0.5236, 1.0472]) # (30, -20, 20, 0, 30, 60)
    current_speed = 0.5

    def get_ee_pose_6d_scaled():
        """Get 6_link world pose (x, y, z, rx, ry, rz)."""
        if link6_id >= 0:
            pos = data.xpos[link6_id].copy() * 1000
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

    print(f"🚀 Starting Headless Collection...")
    pbar = tqdm(total=MAX_EPISODES, desc="Collecting", unit="ep")

    episode_count = 0
    while episode_count < MAX_EPISODES:
        mujoco.mj_resetData(model, data)
        data.qpos[:6] = home_pose
        if args.randomize_phantom_pos:
            randomize_phantom_pos(model, data, phantom_body_id, rotating_id)
        mujoco.mj_forward(model, data)
        
        last_ee_pose = get_ee_pose_6d_scaled()
        task_state, traj_start_time, insertion_started, accumulated_depth, align_timer, traj_initialized, hold_start_time = 1, data.time, False, 0.0, 0, False, None
        
        p_entry, p_depth = data.site_xpos[target_entry_id].copy(), data.site_xpos[target_depth_id].copy()
        start_tip, start_back = data.site_xpos[tip_id].copy(), data.site_xpos[back_id].copy()
        needle_len = np.linalg.norm(start_tip - start_back)
        
        recorder.start()
        step_count, success = 0, False

        while True:
            t_curr = data.time
            curr_tip, curr_back = data.site_xpos[tip_id].copy(), data.site_xpos[back_id].copy()
            
            # --- 1. Expert Trajectory Logic (2-State System) ---
            if task_state == 1:  # State 1: Align (정렬)
                if not traj_initialized:
                    traj_start_time, start_tip_pos, start_back_pos, traj_initialized = t_curr, curr_tip.copy(), curr_back.copy(), True
                progress = smooth_step((t_curr - traj_start_time) / TRAJ_DURATION)
                axis_dir = (p_depth - p_entry) / (np.linalg.norm(p_depth - p_entry) + 1e-10)
                goal_tip, goal_back = p_entry - (axis_dir * 0.0001), p_entry - (axis_dir * (0.0001 + needle_len))
                target_tip_pos, target_back_pos = (1 - progress) * start_tip_pos + progress * goal_tip, (1 - progress) * start_back_pos + progress * goal_back
                if progress >= 1.0:
                    if np.linalg.norm(curr_tip - goal_tip) < 0.002: align_timer += 1
                    else: align_timer = 0
                    if align_timer > 20: task_state, insertion_started = 2, False
            elif task_state == 2:  # State 2: Insert + Hold (삽입 + 대기 통합)
                if not insertion_started:
                    phase3_base_tip, insertion_started, accumulated_depth, hold_start_time = curr_tip.copy(), True, 0.0, None

                axis_dir = (p_depth - p_entry) / (np.linalg.norm(p_depth - p_entry) + 1e-10)

                # 삽입이 완료되지 않은 경우: 계속 삽입
                if accumulated_depth < TARGET_INSERTION_DEPTH:
                    accumulated_depth += 0.0000025
                    target_tip_pos = phase3_base_tip + (axis_dir * accumulated_depth)
                    target_back_pos = target_tip_pos - (axis_dir * needle_len)
                    # 목표 깊이 도달 시 대기 타이머 시작
                    if accumulated_depth >= TARGET_INSERTION_DEPTH:
                        hold_start_time = data.time
                # 삽입이 완료된 경우: 위치 고정 및 대기
                else:
                    if hold_start_time is None:
                        hold_start_time = data.time
                    # 목표를 최종 삽입 깊이로 고정
                    target_tip_pos = phase3_base_tip + (axis_dir * TARGET_INSERTION_DEPTH)
                    target_back_pos = target_tip_pos - (axis_dir * needle_len)
                    # 1초 대기 후 성공
                    if data.time - hold_start_time >= 1.0: success = True; break

            # --- 2. IK Solver ---
            err_tip, err_back = target_tip_pos - curr_tip, target_back_pos - curr_back
            tip_rot_mat = data.site_xmat[tip_id].reshape(3, 3)
            
            # 210도 오프셋
            offset_angle = np.deg2rad(180+30)
            offset_local_vec = np.array([np.cos(offset_angle), np.sin(offset_angle), 0])
            current_side_vec = tip_rot_mat @ offset_local_vec
            
            # current_side_vec = tip_rot_mat @ np.array([1, 0, 0])
            
            needle_axis_curr = (curr_tip - curr_back) / (np.linalg.norm(curr_tip - curr_back) + 1e-10)
            target_side_vec = np.cross(needle_axis_curr, np.array([0, 0, 1]))
            target_side_vec = target_side_vec / np.linalg.norm(target_side_vec) if np.linalg.norm(target_side_vec) > 1e-3 else np.array([1, 0, 0])
            err_roll = np.cross(current_side_vec, target_side_vec)
            
            jac_tip_full, jac_back = np.zeros((6, dof)), np.zeros((3, dof))
            mujoco.mj_jacSite(model, data, jac_tip_full[:3], jac_tip_full[3:], tip_id)
            mujoco.mj_jacSite(model, data, jac_back, None, back_id)
            
            J_p1, e_p1 = jac_tip_full[:3, :n_motors], (err_tip * 50.0)
            if np.linalg.norm(e_p1) > 1.0: e_p1 = e_p1 / np.linalg.norm(e_p1) * 1.0
            J_p1_pinv = np.linalg.pinv(J_p1, rcond=1e-4)
            dq_p1 = J_p1_pinv @ e_p1
            P_null_1 = np.eye(n_motors) - (J_p1_pinv @ J_p1)
            J_p2_proj = jac_back[:, :n_motors] @ P_null_1
            dq_p2 = np.linalg.pinv(J_p2_proj, rcond=1e-4) @ ((err_back * 50.0) - jac_back[:, :n_motors] @ dq_p1)
            P_null_2 = P_null_1 - (np.linalg.pinv(J_p2_proj, rcond=1e-4) @ J_p2_proj)
            J_p3_proj = jac_tip_full[3:, :n_motors] @ P_null_2
            dq_p3 = np.linalg.pinv(J_p3_proj, rcond=1e-4) @ ((err_roll * 10.0) - jac_tip_full[3:, :n_motors] @ (dq_p1 + dq_p2))
            
            data.ctrl[:n_motors] = data.qpos[:n_motors] + (dq_p1 + dq_p2 + dq_p3) * current_speed
            
            # --- 3. Sensor & Step ---
            p_sensor = data.site_xpos[tip_id].copy()
            needle_dir = (curr_tip - curr_back) / (np.linalg.norm(curr_tip - curr_back) + 1e-10)
            dist_to_surface = mujoco.mj_ray(model, data, p_sensor, needle_dir, None, 1, link6_id, np.zeros(1, dtype=np.int32))
            current_sensor_dist = dist_to_surface * 1000.0 if dist_to_surface >= 0 else -1.0

            mujoco.mj_step(model, data)
            step_count += 1
            
            # --- 4. Save ---
            if step_count % 67 == 0:
                current_qpos_deg = np.rad2deg(data.qpos[:n_motors].copy())
                current_ee_pose_mm = get_ee_pose_6d_scaled()
                delta_ee_action = current_ee_pose_mm - last_ee_pose
                
                frames = {}
                for cam_name in ["side_camera", "tool_camera", "top_camera"]:
                    renderer.update_scene(data, camera=cam_name)
                    frames[cam_name] = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)

                recorder.add(frames, current_qpos_deg, current_ee_pose_mm, delta_ee_action, data.time, task_state, current_sensor_dist)
                last_ee_pose = current_ee_pose_mm.copy()

            if data.time - traj_start_time > 50.0: break

        if success:
            recorder.save_async()
            episode_count += 1
            pbar.update(1)
        else:
            # 왜 실패했는지 출력
            reason = "Timeout" if data.time - traj_start_time > 50.0 else "Unknown"
            if task_state == 1: reason = "Failed to Align"
            elif task_state == 2 and not success: reason = "Failed to Insert/Hold"
            print(f"  ⚠️ Episode {episode_count} discarded. Reason: {reason}")
            recorder.discard()

    pbar.close()
    recorder.wait_for_all()  # 모든 저장 완료 대기
    print("\n✅ All Collections Finished!")

if __name__ == "__main__":
    main()
