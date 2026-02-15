#!/bin/bash
# NORA Evaluation Script
# Evaluates a trained NORA model on a single episode from the training dataset

# Configuration
CHECKPOINT="/data/public/NAS/Insertion_VLA_Sim2/outputs/nora/checkpoint_step_5000.pt"
CONFIG="/data/public/NAS/Insertion_VLA_Sim2/TRAIN/Nora/train_config_nora.yaml"
STATS="/data/public/NAS/Insertion_VLA_Sim2/TRAIN/SmolVLA/dataset_stats.yaml"
EPISODE="/data/public/NAS/Insertion_VLA_Sim2/Dataset/all_h5/episode_20260213_213912.h5"
OUTPUT_DIR="/data/public/NAS/Insertion_VLA_Sim2/Eval/outputs/nora_evaluation_$(date +%y%m%d_%H%M%S)"
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

# Run NORA evaluation
python3 /data/public/NAS/Insertion_VLA_Sim2/Eval/evaluate_episode_nora.py \
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
