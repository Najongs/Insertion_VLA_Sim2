#!/bin/bash
# SmolVLA Evaluation Script
# Evaluates a trained model on a single episode from the training dataset

# Configuration
CHECKPOINT="/home/najo/NAS/VLA/Insertion_VLA_Sim2/TRAIN/SmolVLA/outputs/train/smolvla/checkpoints/checkpoint_step_1000.pt"
CONFIG="/home/najo/NAS/VLA/Insertion_VLA_Sim2/TRAIN/SmolVLA/train_config_smolvla_sim.yaml"
STATS="/home/najo/NAS/VLA/Insertion_VLA_Sim2/TRAIN/SmolVLA/dataset_stats.yaml"
EPISODE=/home/najo/NAS/VLA/Insertion_VLA_Sim2/Dataset/Insert_never_random/collected_data_merged/worker0_episode_20260204_100003.h5
OUTPUT_DIR="/home/najo/NAS/VLA/Insertion_VLA_Sim2/Eval/outputs/evaluation_$(date +%y%m%d_%H%M%S)"
DEVICE="cuda"

# Check if files exist
echo "🔍 Checking files..."
for file in "$CHECKPOINT" "$CONFIG" "$STATS" "$EPISODE"; do
    if [ ! -f "$file" ]; then
        echo "❌ File not found: $file"
        echo "Please update the path in this script."
        exit 1
    fi
done
echo "✅ All files found!"
echo ""

# Run evaluation
python3 /home/najo/NAS/VLA/Insertion_VLA_Sim2/Eval/evaluate_episode_normalized.py \
    --checkpoint "${CHECKPOINT}" \
    --config "${CONFIG}" \
    --stats "${STATS}" \
    --episode "${EPISODE}" \
    --output_dir "${OUTPUT_DIR}" \
    --device "${DEVICE}"

echo ""
echo "✅ Evaluation complete!"
echo "📊 Results saved to: ${OUTPUT_DIR}"
echo ""
echo "Output files:"
echo "  - frame_by_frame_comparison.csv  (detailed per-frame comparison)"
echo "  - summary.yaml                   (overall statistics)"
echo "  - error_over_time.png            (error plots over time)"
echo "  - action_comparison.png          (GT vs predicted actions)"
echo "  - trajectory_3d.png              (3D trajectory visualization)"
