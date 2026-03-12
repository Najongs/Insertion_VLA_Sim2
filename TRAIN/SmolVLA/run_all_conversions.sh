#!/bin/bash

# 모든 New_dataset 들을 순차적으로 LeRobot 포맷으로 변환하는 스크립트입니다.
# 데이터셋마다 View 개수가 다를 수 있으므로 공통으로 사용하는 View1~View3만 변환합니다.

cd /data/public/NAS/Insertion_VLA_Sim2/TRAIN/SmolVLA

DATASETS=("New_dataset" "New_dataset2" "New_dataset3" "New_dataset4" "New_dataset5" "New_dataset6")

for DS in "${DATASETS[@]}"; do
    echo "================================================="
    echo "🚀 Starting conversion for $DS"
    echo "================================================="
    
    # repo-id를 소문자로 변환 (예: new_dataset2_real)
    REPO_ID=$(echo "$DS" | tr '[:upper:]' '[:lower:]')_real
    
    python3 convert_folder_to_lerobot.py \
        --dataset-dir "/data/public/NAS/Insertion_VLA_Sim2/Dataset/dataset/$DS" \
        --repo-id "$REPO_ID" \
        --camera-views View1 View2 View3 \
        --num-workers 8
    
    if [ $? -eq 0 ]; then
        echo "✅ Successfully completed conversion for $DS"
    else
        echo "❌ Failed conversion for $DS"
    fi
    echo ""
done

echo "🎉 All datasets processed!"
