import os
import sys
import yaml
import torch
import numpy as np
import cv2
import h5py
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# Mujoco EGL setup
os.environ['MUJOCO_GL'] = 'egl'
import mujoco

# Add SmolVLA to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SMOLVLA_DIR = os.path.join(PROJECT_ROOT, "TRAIN/SmolVLA")
sys.path.append(os.path.join(SMOLVLA_DIR, "lerobot/src"))
sys.path.append(SMOLVLA_DIR)

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK, OBS_STATE
from normalization_utils import Normalizer, load_stats
from transformers import AutoTokenizer

# === Configuration ===
CONFIG_PATH = os.path.join(SMOLVLA_DIR, "train_config_smolvla_sim.yaml")
STATS_PATH = os.path.join(SMOLVLA_DIR, "dataset_stats.yaml")
CHECKPOINT_PATH = "/home/najo/NAS/VLA/Insertion_VLA_Sim2/TRAIN/SmolVLA/outputs/train/smolvla/checkpoints/checkpoint_step_18000.pt"
MODEL_XML = os.path.join(SCRIPT_DIR, "meca_add.xml")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "collected_data_sim_clean/smolvla_inference_rollout.h5")
MAX_STEPS = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

class SmolVLASimInference:
    def __init__(self, config_path, stats_path, checkpoint_path=None):
        self.config = load_config(config_path)
        self.policy_cfg = self.config["policy"]
        
        print(f"🚀 Initializing SmolVLA Policy (Device: {DEVICE})...")
        
        # 1. Setup SmolVLA Config
        smolvla_config = SmolVLAConfig(
            vlm_model_name=self.policy_cfg.get("vlm_model_name", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"),
            n_obs_steps=self.policy_cfg.get("n_obs_steps", 1),
            chunk_size=self.policy_cfg.get("chunk_size", 50),
            n_action_steps=self.policy_cfg.get("n_action_steps", 50),
            max_state_dim=self.policy_cfg.get("max_state_dim", 32),
            max_action_dim=self.policy_cfg.get("max_action_dim", 32),
            num_steps=self.policy_cfg.get("num_steps", 10),
            input_features={
                "observation.images.camera1": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 512, 512)),
                "observation.images.camera2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 512, 512)),
                "observation.images.camera3": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 512, 512)),
                "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
            },
            output_features={
                "action": PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
            },
        )
        
        # 2. Create Policy
        self.policy = SmolVLAPolicy(smolvla_config)
        
        # 3. Load Checkpoint if exists
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"📂 Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            state_dict = checkpoint.get('policy_state_dict', checkpoint.get('model_state_dict', checkpoint))
            # Remove module. prefix if present
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.policy.load_state_dict(state_dict)
        else:
            print("⚠️ No checkpoint found. Using UNTRAINED model (Random Actions).")
            
        self.policy.to(DEVICE)
        self.policy.eval()
        self.policy.reset()

        # 4. Setup Tokenizer & Normalizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.policy_cfg["vlm_model_name"])
        self.normalizer = None
        if os.path.exists(stats_path):
            stats = load_stats(stats_path)
            self.normalizer = Normalizer(stats).to(DEVICE)
            print(f"📊 Normalization stats loaded from: {stats_path}")

        self.instruction = self.config["dataset"]["task_instruction"]
        tokens = self.tokenizer(
            self.instruction, 
            return_tensors="pt", 
            padding="max_length", 
            max_length=self.policy_cfg.get("tokenizer_max_length", 48),
            truncation=True
        )
        self.lang_tokens = tokens["input_ids"].to(DEVICE)
        self.lang_mask = tokens["attention_mask"].to(DEVICE)

    @torch.no_grad()
    def step(self, images, state_6d):
        # Prepare Batch
        # images: list of [H, W, 3] BGR
        # state_6d: [6]
        
        batch = {}
        # 1. Images: [1, 3, 512, 512] and range [0, 1]
        for i, img in enumerate(images):
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            batch[f"observation.images.camera{i+1}"] = img_tensor.unsqueeze(0).to(DEVICE) # (1, 3, H, W)
            
        # 2. State
        state_tensor = torch.from_numpy(state_6d).float().unsqueeze(0).to(DEVICE)
        if self.normalizer:
            state_tensor = self.normalizer.normalize(state_tensor, "observation.state")
        batch[OBS_STATE] = state_tensor
        
        # 3. Language
        batch[OBS_LANGUAGE_TOKENS] = self.lang_tokens
        batch[OBS_LANGUAGE_ATTENTION_MASK] = self.lang_mask
        
        # 4. Predict Action (Single Action from Chunk Queue)
        # policy.select_action handles the internal queueing of chunk_size(50) actions
        norm_action = self.policy.select_action(batch) # Returns (6,)
        
        # 5. Unnormalize Action
        if self.normalizer:
            unnorm_action = self.normalizer.unnormalize(norm_action.unsqueeze(0), "action").squeeze(0)
        else:
            unnorm_action = norm_action
            
        # Ensure 1D array and take first 6 elements (pose)
        if unnorm_action.ndim > 1:
            unnorm_action = unnorm_action.flatten()
            
        return unnorm_action[:6].cpu().numpy()

def get_ee_pose_6d(data, link6_id):
    pos = data.xpos[link6_id].copy() * 1000 # mm
    mat = data.xmat[link6_id].reshape(3, 3)
    sy = np.sqrt(mat[0, 0] ** 2 + mat[1, 0] ** 2)
    if sy > 1e-6:
        r, p, y = np.arctan2(mat[2, 1], mat[2, 2]), np.arctan2(-mat[2, 0], sy), np.arctan2(mat[1, 0], mat[0, 0])
    else:
        r, p, y = np.arctan2(-mat[1, 2], mat[1, 1]), np.arctan2(-mat[2, 0], sy), 0.0
    return np.concatenate([pos, [r, p, y]])

def euler_to_rot_mat(r, p, y):
    Rx = np.array([[1,0,0],[0,np.cos(r),-np.sin(r)],[0,np.sin(r),np.cos(r)]])
    Ry = np.array([[np.cos(p),0,np.sin(p)],[0,1,0],[-np.sin(p),0,np.cos(p)]])
    Rz = np.array([[np.cos(y),-np.sin(y),0],[np.sin(y),np.cos(y),0],[0,0,1]])
    return Rz @ Ry @ Rx

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
    # Use 480x480 to fit within default MuJoCo offscreen buffer. 
    # SmolVLAPolicy will resize this to 512x512 internally.
    renderer = mujoco.Renderer(model, height=480, width=480)
    link6_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "6_Link")
    n_motors = model.nu

    # Load SmolVLA
    vla = SmolVLASimInference(CONFIG_PATH, STATS_PATH, CHECKPOINT_PATH)

    # Buffer for saving
    buffer = []

    print(f"🎬 Starting SmolVLA Rollout for {MAX_STEPS} steps...")
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    for t in tqdm(range(MAX_STEPS), desc="Rollout"):
        # 1. Capture Images
        cam_images = []
        for cam_name in ["side_camera", "tool_camera", "top_camera"]:
            renderer.update_scene(data, camera=cam_name)
            cam_images.append(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
        
        # 2. Get State
        curr_state = get_ee_pose_6d(data, link6_id)
        
        # 3. Inference
        action = vla.step(cam_images, curr_state)
        
        # 4. Apply Action (Treating action as Delta EE Pose)
        delta_ee = action[:6] # [dx, dy, dz, dr, dp, dy]
        
        target_ee = curr_state + delta_ee
        target_pos_m = target_ee[:3] / 1000.0
        target_rot_mat = euler_to_rot_mat(target_ee[3], target_ee[4], target_ee[5])
        
        new_q = solve_ik(model, data, target_pos_m, target_rot_mat, link6_id, data.qpos[:n_motors].copy(), n_motors)
        data.qpos[:n_motors] = new_q
        mujoco.mj_forward(model, data)

        # 5. Record
        buffer.append({
            "imgs": cam_images,
            "qpos": np.rad2deg(data.qpos[:n_motors]).copy(),
            "ee_pose": curr_state.copy(),
            "action": action.flatten() # Ensure 1D array
        })

    # Save to HDF5
    print(f"💾 Saving to {OUTPUT_PATH}...")
    with h5py.File(OUTPUT_PATH, 'w') as f:
        obs = f.create_group("observations")
        img_grp = obs.create_group("images")
        obs.create_dataset("qpos", data=np.array([x['qpos'] for x in buffer]), compression="gzip")
        obs.create_dataset("ee_pose", data=np.array([x['ee_pose'] for x in buffer]), compression="gzip")
        # Ensure action dataset is stored as float32 and shape (N, ActionDim)
        f.create_dataset("action", data=np.array([x['action'] for x in buffer], dtype=np.float32), compression="gzip")
        
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
