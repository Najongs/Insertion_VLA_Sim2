#!/bin/bash

echo "=================================================="
echo "SmolVLA Simulation: Data Collection + Training"
echo "=================================================="
echo ""

# 1. 환경 변수 설정
export PYTHONPATH="/home/najo/NAS/VLA/Insertion_VLAv4/lerobot/src:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4"
# PyTorch 내부 쓰레드 제한 (쉘에서는 환경변수로 설정)
export OMP_NUM_THREADS=4

# 2. 경로 설정
CONFIG_FILE="/home/najo/NAS/VLA/Insertion_VLA_Sim2/TRAIN/SmolVLA/train_config_smolvla_sim.yaml"
CHECKPOINT="/home/najo/NAS/VLA/Insertion_VLA_Sim2/TRAIN/SmolVLA/outputs/train/smolvla_3cams/checkpoints/checkpoint_step_1000.pt"

# 3. 학습 실행
torchrun --nproc-per-node=5 train_smolvla_sim.py \
    --config "$CONFIG_FILE" \
    --resume "$CHECKPOINT"
    # --reset_scheduler