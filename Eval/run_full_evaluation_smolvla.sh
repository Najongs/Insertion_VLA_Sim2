#!/bin/bash
# Complete evaluation workflow:
# 1. Evaluate model on episode -> CSV
# 2. Reproduce Original trajectory from source qpos -> HDF5
# 3. Reproduce GT trajectory from GT delta actions -> HDF5
# 4. Reproduce Predicted trajectory -> HDF5
# 5. Compare GT vs Pred -> PNG

set -e  # Exit on error

# Configuration
CHECKPOINT="/data/public/NAS/Insertion_VLA_Sim2/TRAIN/SmolVLA/outputs/train/smolvla_lerobot_dataset/checkpoints/100000/pretrained_model"
CONFIG="/data/public/NAS/Insertion_VLA_Sim2/TRAIN/SmolVLA/train_config_smolvla_sim.yaml"
STATS="/data/public/NAS/Insertion_VLA_Sim2/TRAIN/SmolVLA/dataset_stats.yaml"
# EPISODE="/data/public/NAS/Insertion_VLA_Sim2/Dataset/dataset/New_dataset/collected_data/Eye_trocar/Eye_trocar/260119/episode_20260119_140516.h5"
EPISODE="/data/public/NAS/Insertion_VLA_Sim2/Sim/collected_data_sim_clean/episode_20260312_222756.h5"
OUTPUT_BASE="outputs/full_workflow_$(date +%y%m%d_%H%M%S)"
DEVICE="cuda"

# Create output directory
mkdir -p "${OUTPUT_BASE}"

echo "======================================================================"
echo "SmolVLA Full Evaluation Workflow"
echo "======================================================================"
echo "Checkpoint:  ${CHECKPOINT}"
echo "Episode:     ${EPISODE}"
echo "Output dir:  ${OUTPUT_BASE}"
echo "======================================================================"
echo ""

# Step 1: Evaluate model on episode
echo "======================================================================"
echo "Step 1/5: Evaluating model on episode..."
echo "======================================================================"
python3 /data/public/NAS/Insertion_VLA_Sim2/Eval/evaluate_episode_lerobot.py \
    --pretrained_model "${CHECKPOINT}" \
    --episode "${EPISODE}" \
    --output_dir "${OUTPUT_BASE}/evaluation" \
    --device "${DEVICE}"

CSV_FILE="${OUTPUT_BASE}/evaluation/frame_by_frame_comparison.csv"

if [ ! -f "${CSV_FILE}" ]; then
    echo "❌ Error: Evaluation failed, CSV not found"
    exit 1
fi

echo "✅ Step 1 complete: Evaluation results saved"
echo ""

# Step 2: Reproduce Original trajectory
echo "======================================================================"
echo "Step 2/5: Reproducing Original trajectory from source qpos..."
echo "======================================================================"
python3 /data/public/NAS/Insertion_VLA_Sim2/Sim/reproduce_episode.py \
    "${EPISODE}" \
    --output_file "${OUTPUT_BASE}/reproduced_original.h5"

if [ ! -f "${OUTPUT_BASE}/reproduced_original.h5" ]; then
    echo "❌ Error: Original reproduction failed"
    exit 1
fi

echo "✅ Step 2 complete: Original trajectory reproduced"
echo ""

# Step 3: Reproduce GT trajectory
echo "======================================================================"
echo "Step 3/5: Reproducing Ground Truth trajectory..."
echo "======================================================================"
python3 /data/public/NAS/Insertion_VLA_Sim2/Eval/reproduce_from_csv.py \
    --csv "${CSV_FILE}" \
    --episode "${EPISODE}" \
    --mode gt \
    --output "${OUTPUT_BASE}/reproduced_gt.h5"

if [ ! -f "${OUTPUT_BASE}/reproduced_gt.h5" ]; then
    echo "❌ Error: GT reproduction failed"
    exit 1
fi

echo "✅ Step 3 complete: GT trajectory reproduced"
echo ""

# Step 4: Reproduce Predicted trajectory
echo "======================================================================"
echo "Step 4/5: Reproducing Predicted trajectory..."
echo "======================================================================"
python3 /data/public/NAS/Insertion_VLA_Sim2/Eval/reproduce_from_csv.py \
    --csv "${CSV_FILE}" \
    --episode "${EPISODE}" \
    --mode pred \
    --output "${OUTPUT_BASE}/reproduced_pred.h5"

if [ ! -f "${OUTPUT_BASE}/reproduced_pred.h5" ]; then
    echo "❌ Error: Predicted reproduction failed"
    exit 1
fi

echo "✅ Step 4 complete: Predicted trajectory reproduced"
echo ""

# Step 5: Compare trajectories
echo "======================================================================"
echo "Step 5/5: Comparing GT vs Predicted trajectories..."
echo "======================================================================"
python3 /data/public/NAS/Insertion_VLA_Sim2/Eval/compare_reproductions.py \
    --gt "${OUTPUT_BASE}/reproduced_gt.h5" \
    --pred "${OUTPUT_BASE}/reproduced_pred.h5" \
    --output "${OUTPUT_BASE}/trajectory_comparison.png"

if [ ! -f "${OUTPUT_BASE}/trajectory_comparison.png" ]; then
    echo "❌ Error: Comparison failed"
    exit 1
fi

echo "✅ Step 5 complete: Comparison plot created"
echo ""

# Summary
echo "======================================================================"
echo "✅ Full Evaluation Workflow Complete!"
echo "======================================================================"
echo "Output directory: ${OUTPUT_BASE}"
echo ""
echo "Generated files:"
echo "  📊 Evaluation Results:"
echo "     - evaluation/frame_by_frame_comparison.csv"
echo "     - evaluation/summary.yaml"
echo "     - evaluation/error_over_time.png"
echo "     - evaluation/action_comparison.png"
echo "     - evaluation/trajectory_3d.png"
echo ""
echo "  🎬 Reproduced Trajectories:"
echo "     - reproduced_original.h5 (Original qpos replay)"
echo "     - reproduced_gt.h5    (Ground Truth)"
echo "     - reproduced_pred.h5  (Predicted)"
echo ""
echo "  📈 Comparison:"
echo "     - trajectory_comparison.png"
echo ""
echo "======================================================================"
echo "Next steps:"
echo "  1. Check evaluation/summary.yaml for overall statistics"
echo "  2. View trajectory_comparison.png to see GT vs Pred differences"
echo "  3. Examine trajectory_3d.png for predicted trajectory visualization"
echo "======================================================================"
