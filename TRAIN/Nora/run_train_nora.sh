#!/bin/bash
# NORA VLA Training Launch Script
# 3x RTX 3090 Multi-GPU Training with Accelerate
#
# Usage:
#   bash run_train_nora.sh
#   or with custom config:
#   bash run_train_nora.sh --config my_config.yaml

set -e

# Navigate to training directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default config
CONFIG="${1:-train_config_nora.yaml}"

echo "============================================"
echo "  NORA VLA Training"
echo "============================================"
echo "  Config: $CONFIG"
echo "  Working dir: $(pwd)"
echo "  GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)x $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "============================================"

# Set environment variables
export NCCL_TIMEOUT=1800
export NCCL_BLOCKING_WAIT=1
export TOKENIZERS_PARALLELISM=false

# Launch training with accelerate
accelerate launch \
    --config_file accelerator_config.yaml \
    train_nora.py \
    --config "$CONFIG"
