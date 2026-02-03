import os
import sys
import time
import numpy as np
import cv2
import h5py
import torch
import PIL.Image
import yaml
from tqdm import tqdm

# Mujoco EGL setup
os.environ['MUJOCO_GL'] = 'egl'
import mujoco

# Add Nora to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(os.path.join(PROJECT_ROOT, "TRAIN/Nora"))

try:
    from nora.inference.nora import Nora
    from qwen_vl_utils import process_vision_info
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Please ensure nora and qwen_vl_utils are in the path.")
    sys.exit(1)

# === Configuration ===
MODEL_XML = os.path.join(SCRIPT_DIR, "meca_add.xml")
CHECKPOINT_PATH = "/home/najo/NAS/VLA/Insertion_VLA_Sim2/TRAIN/Nora/checkpoint_step_14000.pt"
STATS_PATH = os.path.join(PROJECT_ROOT, "TRAIN/SmolVLA/dataset_stats.yaml")
INSTRUCTION = "Move the needle to the eye phantom and insert it through the trocar opening"
MAX_STEPS = 1000  # Number of inference steps
IMG_WIDTH = 640
IMG_HEIGHT = 480
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "collected_data_sim_clean/nora_inference_rollout.h5")

class NoraSimInference:
    def __init__(self, checkpoint_path, stats_path):
        base_model = "declare-lab/nora"
        print(f"🚀 Initializing Nora with base model: {base_model}...")
        # Initialize Nora with base model to load processor and model structure
        self.nora = Nora(model_path=base_model)
        
        print(f"📂 Loading weights from checkpoint: {checkpoint_path}...")
        # Load weights from the .pt file
        checkpoint = torch.load(checkpoint_path, map_location=self.nora.device)
        
        # Handle different checkpoint saving formats
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
        else:
            state_dict = checkpoint
            
        # Load state dict into the model
        # Note: If the checkpoint was saved with DDP, we might need to remove 'module.' prefix
        clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.nora.model.load_state_dict(clean_state_dict, strict=True)
        print("✅ Checkpoint weights loaded successfully.")
        
        # Load custom sim stats (Mean/Std)
        print(f"📊 Loading custom stats for unnormalization: {stats_path}")
        with open(stats_path, 'r') as f:
            stats = yaml.safe_load(f)
        self.action_mean = np.array(stats['action']['mean'])
        self.action_std = np.array(stats['action']['std'])

    @torch.inference_mode()
    def predict_action(self, images, instruction):
        """
        Modified inference to support multiple images (3 cameras).
        """
        pil_images = [PIL.Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in images]
        
        # Construct multi-image message
        content = []
        for img in pil_images:
            content.append({
                "type": "image",
                "image": img,
                "resized_height": 224,
                "resized_width": 224,
            })
        content.append({"type": "text", "text": instruction})
        
        messages = [{"role": "user", "content": content}]
        
        # Apply template
        text = self.nora.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.nora.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.nora.device) for k, v in inputs.items()}
        
        # Generate with specific config to avoid warnings
        generated_ids = self.nora.model.generate(
            **inputs,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            max_new_tokens=20 # Limit new tokens
        )
        
        # 1. Only look at the GENERATED tokens (exclude prompt)
        prompt_len = inputs['input_ids'].shape[1]
        new_tokens = generated_ids[0][prompt_len:]
        
        # 2. Find action tokens in the new_tokens
        mask = (self.nora._ACTION_TOKEN_MIN <= new_tokens) & (new_tokens <= self.nora._ACTION_TOKEN_MAX)
        action_indices = torch.where(mask)[0]
        
        if len(action_indices) == 0:
            print("⚠️ No action tokens found in the generated output!")
            return np.zeros(7)

        # 3. Take the LAST 7 tokens (most likely the actual action)
        # Nora/OpenVLA usually predicts 7-dim actions per step
        raw_token_ids = new_tokens[action_indices]
        
        # Determine how many tokens to take. 
        # If we assume action_dim=7, we take the last 7.
        if len(raw_token_ids) >= 7:
            target_tokens = raw_token_ids[-7:]
        else:
            # Pad if insufficient
            padding = torch.full((7 - len(raw_token_ids),), self.nora._ACTION_TOKEN_MIN, device=raw_token_ids.device)
            target_tokens = torch.cat([raw_token_ids, padding])
            
        # Manual Decoding (Bypassing fast_tokenizer)
        # Action tokens are in range [MIN, MAX]. We map them to [-1, 1]
        # Total bins = MAX - MIN + 1
        min_token = self.nora._ACTION_TOKEN_MIN
        max_token = self.nora._ACTION_TOKEN_MAX
        total_bins = max_token - min_token # e.g. 2047
        
        # Calculate offset (0 to 2047)
        offsets = (target_tokens - min_token).float()
        
        # Map to [-1, 1]
        output_action = (offsets / total_bins) * 2.0 - 1.0
        output_action = output_action.cpu().numpy()

        # UNNORMALIZATION using Mean/Std
        # output_action is in range [-1, 1]
        
        # Take only first 6 dimensions (pose) to match sim stats
        # (Assuming the 7th dimension is gripper which we ignore for normalization)
        pose_action = output_action[:6]
        
        unnorm_action = pose_action * self.action_std + self.action_mean
        
        # Return 7 dims (6 pose + dummy 0 gripper)
        final_action = np.zeros(7)
        final_action[:6] = unnorm_action
        
        return final_action
        # output_action is in range [-1, 1], likely shape (1, 1, 7)
        flat_action = output_action.flatten()
        
        # Take only first 6 dimensions (pose) to match sim stats
        unnorm_action = flat_action[:6] * self.action_std + self.action_mean
        
        # Return 7 dims (6 pose + dummy 1 gripper for consistency if needed, but here we return pose)
        # To match previous return type, let's return a 7-dim array where 7th is 0
        final_action = np.zeros(7)
        final_action[:6] = unnorm_action
        
        return final_action

# === Math & IK Helpers ===
def euler_to_rot_mat(r, p, y):
    Rx = np.array([[1,0,0],[0,np.cos(r),-np.sin(r)],[0,np.sin(r),np.cos(r)]])
    Ry = np.array([[np.cos(p),0,np.sin(p)],[0,1,0],[-np.sin(p),0,np.cos(p)]])
    Rz = np.array([[np.cos(y),-np.sin(y),0],[np.sin(y),np.cos(y),0],[0,0,1]])
    return Rz @ Ry @ Rx

def get_ee_pose_6d(data, link6_id):
    pos = data.xpos[link6_id].copy() * 1000 # mm
    mat = data.xmat[link6_id].reshape(3, 3)
    sy = np.sqrt(mat[0, 0] ** 2 + mat[1, 0] ** 2)
    if sy > 1e-6:
        r, p, y = np.arctan2(mat[2, 1], mat[2, 2]), np.arctan2(-mat[2, 0], sy), np.arctan2(mat[1, 0], mat[0, 0])
    else:
        r, p, y = np.arctan2(-mat[1, 2], mat[1, 1]), np.arctan2(-mat[2, 0], sy), 0.0
    return np.concatenate([pos, [r, p, y]])

def solve_ik(model, data, target_pos_m, target_rot_mat, link_id, q_init, n_motors):
    data.qpos[:n_motors] = q_init
    mujoco.mj_forward(model, data)
    for _ in range(100):
        err_pos = target_pos_m - data.xpos[link_id]
        R_err = target_rot_mat @ data.xmat[link_id].reshape(3, 3).T
        quat_err = np.zeros(4)
        mujoco.mju_mat2Quat(quat_err, R_err.flatten())
        if quat_err[0] >= 1.0: err_rot = np.zeros(3)
        else:
            theta = 2 * np.arccos(np.clip(quat_err[0], -1, 1))
            sin_half = np.sqrt(1 - quat_err[0]**2)
            err_rot = theta * (quat_err[1:] / sin_half) if sin_half > 1e-6 else np.zeros(3)
        if np.linalg.norm(err_pos) < 1e-5 and np.linalg.norm(err_rot) < 1e-4: break
        jac_p, jac_r = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jac_p, jac_r, link_id)
        J = np.vstack([jac_p[:, :n_motors], jac_r[:, :n_motors]])
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), np.concatenate([err_pos, err_rot]))
        data.qpos[:n_motors] += dq
        mujoco.mj_forward(model, data)
    return data.qpos[:n_motors].copy()

def main():
    # Setup Sim
    model = mujoco.MjModel.from_xml_path(MODEL_XML)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=IMG_HEIGHT, width=IMG_WIDTH)
    link6_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "6_Link")
    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "needle_tip")
    back_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "needle_back")
    n_motors = model.nu

    # Load Nora
    nora_sim = NoraSimInference(CHECKPOINT_PATH, STATS_PATH)

    # Buffer for saving
    buffer = []

    print(f"🎬 Starting Simulation Inference for {MAX_STEPS} steps...")
    mujoco.mj_resetData(model, data)
    # Start at a neutral pose (can be adjusted)
    data.qpos[:n_motors] = 0 
    mujoco.mj_forward(model, data)

    for t in tqdm(range(MAX_STEPS), desc="Rollout"):
        # 1. Capture Images
        cam_images = []
        for cam_name in ["side_camera", "tool_camera", "top_camera"]:
            renderer.update_scene(data, camera=cam_name)
            cam_images.append(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
        
        # 2. Inference
        action = nora_sim.predict_action(cam_images, INSTRUCTION)
        delta_ee = action[:6] # [dx, dy, dz, dr, dp, dy]
        
        # 3. Apply Action via IK
        curr_ee = get_ee_pose_6d(data, link6_id)
        target_ee = curr_ee + delta_ee
        
        target_pos_m = target_ee[:3] / 1000.0
        target_rot_mat = euler_to_rot_mat(target_ee[3], target_ee[4], target_ee[5])
        
        new_q = solve_ik(model, data, target_pos_m, target_rot_mat, link6_id, data.qpos[:n_motors].copy(), n_motors)
        data.qpos[:n_motors] = new_q
        mujoco.mj_forward(model, data)

        # 4. Measure sensors
        p_sensor = data.site_xpos[tip_id].copy()
        needle_dir = (data.site_xpos[tip_id] - data.site_xpos[back_id])
        needle_dir /= (np.linalg.norm(needle_dir) + 1e-10)
        dist = mujoco.mj_ray(model, data, p_sensor, needle_dir, None, 1, link6_id, np.zeros(1, dtype=np.int32))
        sensor_dist = dist * 1000.0 if dist >= 0 else -1.0

        # 5. Record
        buffer.append({
            "imgs": cam_images,
            "qpos": np.rad2deg(data.qpos[:n_motors]).copy(),
            "ee_pose": get_ee_pose_6d(data, link6_id),
            "action": action,
            "sensor_dist": sensor_dist
        })

    # Save to HDF5
    print(f"💾 Saving to {OUTPUT_PATH}...")
    
    # Ensure directory exists
    output_dir = os.path.dirname(OUTPUT_PATH)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with h5py.File(OUTPUT_PATH, 'w') as f:
        obs = f.create_group("observations")
        img_grp = obs.create_group("images")
        obs.create_dataset("qpos", data=np.array([x['qpos'] for x in buffer]), compression="gzip")
        obs.create_dataset("ee_pose", data=np.array([x['ee_pose'] for x in buffer]), compression="gzip")
        obs.create_dataset("sensor_dist", data=np.array([x['sensor_dist'] for x in buffer]), compression="gzip")
        f.create_dataset("action", data=np.array([x['action'] for x in buffer]), compression="gzip")
        
        for i, cam_name in enumerate(["side_camera", "tool_camera", "top_camera"]):
            jpeg_list = []
            for step in buffer:
                success, buf = cv2.imencode('.jpg', step["imgs"][i], [cv2.IMWRITE_JPEG_QUALITY, 95])
                jpeg_list.append(buf.flatten())
            dt = h5py.special_dtype(vlen=np.dtype('uint8'))
            dset = img_grp.create_dataset(cam_name, (len(jpeg_list),), dtype=dt)
            for j, code in enumerate(jpeg_list): dset[j] = code
    print("✅ Rollout complete and saved.")

if __name__ == "__main__":
    main()
