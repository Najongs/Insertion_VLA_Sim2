#!/bin/bash
# Complete evaluation workflow:
# 1. Evaluate model on episode -> CSV
# 2. Reproduce GT trajectory -> HDF5
# 3. Reproduce Predicted trajectory -> HDF5
# 4. Compare GT vs Pred -> PNG

set -e  # Exit on error

# Configuration
CHECKPOINT="/home/najo/NAS/VLA/Insertion_VLA_Sim2/TRAIN/SmolVLA/outputs/train/smolvla/checkpoints/checkpoint_step_7000.pt"
CONFIG="/home/najo/NAS/VLA/Insertion_VLA_Sim2/TRAIN/SmolVLA/train_config_smolvla_sim.yaml"
STATS="/home/najo/NAS/VLA/Insertion_VLA_Sim2/TRAIN/SmolVLA/dataset_stats.yaml"
EPISODE="/home/najo/NAS/VLA/Insertion_VLA_Sim2/Dataset/Insert_never_random/collected_data_merged/worker0_episode_20260204_100003.h5"
OUTPUT_BASE="/home/najo/NAS/VLA/Insertion_VLA_Sim2/Eval/outputs/full_workflow_$(date +%y%m%d_%H%M%S)"
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
echo "Step 1/4: Evaluating model on episode..."
echo "======================================================================"
python3 /home/najo/NAS/VLA/Insertion_VLA_Sim2/Eval/evaluate_episode_normalized.py \
    --checkpoint "${CHECKPOINT}" \
    --config "${CONFIG}" \
    --stats "${STATS}" \
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

# Step 2: Reproduce GT trajectory
echo "======================================================================"
echo "Step 2/4: Reproducing Ground Truth trajectory..."
echo "======================================================================"
python3 /home/najo/NAS/VLA/Insertion_VLA_Sim2/Eval/reproduce_from_csv.py \
    --csv "${CSV_FILE}" \
    --mode gt \
    --output "${OUTPUT_BASE}/reproduced_gt.h5"

if [ ! -f "${OUTPUT_BASE}/reproduced_gt.h5" ]; then
    echo "❌ Error: GT reproduction failed"
    exit 1
fi

echo "✅ Step 2 complete: GT trajectory reproduced"
echo ""

# Step 3: Reproduce Predicted trajectory
echo "======================================================================"
echo "Step 3/4: Reproducing Predicted trajectory..."
echo "======================================================================"
python3 /home/najo/NAS/VLA/Insertion_VLA_Sim2/Eval/reproduce_from_csv.py \
    --csv "${CSV_FILE}" \
    --mode pred \
    --output "${OUTPUT_BASE}/reproduced_pred.h5"

if [ ! -f "${OUTPUT_BASE}/reproduced_pred.h5" ]; then
    echo "❌ Error: Predicted reproduction failed"
    exit 1
fi

echo "✅ Step 3 complete: Predicted trajectory reproduced"
echo ""

# Step 4: Compare trajectories
echo "======================================================================"
echo "Step 4/4: Comparing GT vs Predicted trajectories..."
echo "======================================================================"
python3 /home/najo/NAS/VLA/Insertion_VLA_Sim2/Eval/compare_reproductions.py \
    --gt "${OUTPUT_BASE}/reproduced_gt.h5" \
    --pred "${OUTPUT_BASE}/reproduced_pred.h5" \
    --output "${OUTPUT_BASE}/trajectory_comparison.png"

if [ ! -f "${OUTPUT_BASE}/trajectory_comparison.png" ]; then
    echo "❌ Error: Comparison failed"
    exit 1
fi

echo "✅ Step 4 complete: Comparison plot created"
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
