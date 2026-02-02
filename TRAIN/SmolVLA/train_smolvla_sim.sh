#!/bin/bash
# Training script for SmolVLA on Simulation Data with Data Collection
# Usage: bash train_smolvla_sim.sh [--skip-collection] [--episodes N] [--workers W] [--resume-from-checkpoint /path/to/checkpoint.pt]

echo "=================================================="
echo "SmolVLA Simulation: Data Collection + Training"
echo "=================================================="
echo ""

# Set environment variables
export PYTHONPATH=/home/najo/NAS/VLA/Insertion_VLAv4/lerobot/src:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4

# Parse command-line arguments
SKIP_COLLECTION=false
NUM_EPISODES=100
NUM_WORKERS=1
RESUME_CHECKPOINT=""

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --skip-collection)
            SKIP_COLLECTION=true
            ;;
        --episodes)
            NUM_EPISODES="$2"
            shift
            ;;
        --workers)
            NUM_WORKERS="$2"
            shift
            ;;
        --resume-from-checkpoint)
            RESUME_CHECKPOINT="$2"
            shift
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
    shift
done

# Configuration
CONFIG_FILE="/home/najo/NAS/VLA/Insertion_VLA_Sim2/TRAIN/SmolVLA/train_config_smolvla_sim.yaml"
NUM_GPUS=5
DATASET_PATH="/home/najo/NAS/VLA/Insertion_VLA_Sim2/Dataset"

TOTAL_EPISODES=$((NUM_WORKERS * NUM_EPISODES))

echo "Configuration:"
echo "  Config file: $CONFIG_FILE"
echo "  GPUs: $NUM_GPUS (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "  Data Collection: $([ "$SKIP_COLLECTION" = true ] && echo "SKIPPED" || echo "ENABLED")"
if [ "$SKIP_COLLECTION" = false ]; then
    echo "  Episodes to collect: $TOTAL_EPISODES ($NUM_WORKERS workers × $NUM_EPISODES episodes)"
fi
if [ -n "$RESUME_CHECKPOINT" ]; then
    echo "  Resuming from checkpoint: $RESUME_CHECKPOINT"
fi
echo ""

# ============================================================
# STEP 3: Training
# ============================================================
echo "=================================================="
echo "STEP 3: Training SmolVLA Model"
echo "=================================================="
echo ""

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "Starting training..."
echo "  Config: $CONFIG_FILE"
echo "  Dataset stats: dataset_stats_sim_6d_clean.yaml"
echo ""

# Run training with torchrun (DDP)
TRAIN_COMMAND="torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29501 \
    train_smolvla_sim.py \
    --config $CONFIG_FILE"

if [ -n "$RESUME_CHECKPOINT" ]; then
    TRAIN_COMMAND="$TRAIN_COMMAND --resume \"$RESUME_CHECKPOINT\""
fi

eval $TRAIN_COMMAND

echo ""
echo "=================================================="
echo "Training completed!"
echo "=================================================="
