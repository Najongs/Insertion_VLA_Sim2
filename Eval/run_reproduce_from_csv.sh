#!/bin/bash
# Reproduce trajectory from CSV evaluation results in MuJoCo simulation

# Configuration
CSV_FILE="/home/najo/NAS/VLA/Insertion_VLA_Sim2/Eval/outputs/evaluation_3000_step/frame_by_frame_comparison.csv"
MODE="pred"  # Options: "pred" (predicted) or "gt" (ground truth)
OUTPUT_DIR="/home/najo/NAS/VLA/Insertion_VLA_Sim2/Eval/outputs/reproduction_$(date +%y%m%d_%H%M%S)"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Set output file path
OUTPUT_FILE="${OUTPUT_DIR}/reproduced_${MODE}.h5"

echo "======================================================================"
echo "CSV Trajectory Reproduction"
echo "======================================================================"
echo "CSV file:     ${CSV_FILE}"
echo "Mode:         ${MODE}"
echo "Output file:  ${OUTPUT_FILE}"
echo "======================================================================"
echo ""

# Check if CSV exists
if [ ! -f "${CSV_FILE}" ]; then
    echo "❌ Error: CSV file not found: ${CSV_FILE}"
    echo "Please update the CSV_FILE path in this script."
    exit 1
fi

# Run reproduction
python3 /home/najo/NAS/VLA/Insertion_VLA_Sim2/Eval/reproduce_from_csv.py \
    --csv "${CSV_FILE}" \
    --mode "${MODE}" \
    --output "${OUTPUT_FILE}"

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "✅ Reproduction complete!"
    echo "======================================================================"
    echo "Output files:"
    echo "  - ${OUTPUT_FILE}"
    echo ""
    echo "You can now:"
    echo "  1. Visualize with h5py or custom viewer"
    echo "  2. Compare with original episode"
    echo "  3. Change MODE to 'gt' to reproduce ground truth trajectory"
    echo "======================================================================"
else
    echo ""
    echo "❌ Reproduction failed!"
fi
