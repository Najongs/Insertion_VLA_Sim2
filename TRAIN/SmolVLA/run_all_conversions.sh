#!/bin/bash

# 모든 New_dataset 들을 순차적으로 LeRobot 포맷으로 변환하는 스크립트입니다.
# - New_dataset: HDF5 기반 real dataset
# - New_dataset2~: 폴더 기반 multi-view dataset

cd /data/public/NAS/Insertion_VLA_Sim2/TRAIN/SmolVLA

DATASETS=("New_dataset" "New_dataset2" "New_dataset3" "New_dataset4" "New_dataset5")

for DS in "${DATASETS[@]}"; do
    echo "================================================="
    echo "🚀 Starting conversion for $DS"
    echo "================================================="
    
    # repo-id를 소문자로 변환 (예: new_dataset2_real)
    REPO_ID=$(echo "$DS" | tr '[:upper:]' '[:lower:]')_real

    if [ "$DS" = "New_dataset" ]; then
        python3 convert_real_hdf5_to_lerobot.py \
            --hdf5-dir "/data/public/NAS/Insertion_VLA_Sim2/Dataset/dataset/$DS/collected_data" \
            --output-dir "/data/public/NAS/Insertion_VLA_Sim2/TRAIN/SmolVLA/lerobot_multi_datasets" \
            --repo-id "$REPO_ID" \
            --num-workers 8
    else
        python3 convert_folder_to_lerobot.py \
            --dataset-dir "/data/public/NAS/Insertion_VLA_Sim2/Dataset/dataset/$DS" \
            --repo-id "$REPO_ID" \
            --num-workers 8
    fi
    
    if [ $? -eq 0 ]; then
        echo "✅ Successfully completed conversion for $DS"
    else
        echo "❌ Failed conversion for $DS"
    fi
    echo ""
done

echo "🎉 All datasets processed!"
