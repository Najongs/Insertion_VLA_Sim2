#!/usr/bin/env python3
import os
import sys
import json
import shutil
from pathlib import Path

# Try importing LeRobot and pyarrow
try:
    sys.path.insert(0, '/data/public/NAS/Insertion_VLA_Sim2/TRAIN/SmolVLA/lerobot/src')
    from lerobot.datasets.aggregate import aggregate_datasets
except ImportError:
    print("Error: Could not import LeRobot aggregate_datasets. Ensure paths are correct.")
    sys.exit(1)

try:
    import pyarrow.parquet as pq
    import pyarrow as pa
except ImportError:
    print("Error: pyarrow is required to patch dataset columns. (pip install pyarrow)")
    sys.exit(1)

# Configuration
DATASETS_TO_MERGE = [
    {
        "repo_id": "eye_trocar_real",
        "root": "/data/public/NAS/Insertion_VLA_Sim2/TRAIN/SmolVLA/lerobot_real_dataset/eye_trocar_real"
    },
    {
        "repo_id": "new_dataset2_real",
        "root": "/data/public/NAS/Insertion_VLA_Sim2/TRAIN/SmolVLA/lerobot_multi_datasets/new_dataset2_real",
        "needs_patching": True
    },
    {
        "repo_id": "new_dataset3_real",
        "root": "/data/public/NAS/Insertion_VLA_Sim2/TRAIN/SmolVLA/lerobot_multi_datasets/new_dataset3_real",
        "needs_patching": True
    },
    {
        "repo_id": "new_dataset4_real",
        "root": "/data/public/NAS/Insertion_VLA_Sim2/TRAIN/SmolVLA/lerobot_multi_datasets/new_dataset4_real",
        "needs_patching": True
    },
    {
        "repo_id": "new_dataset5_real",
        "root": "/data/public/NAS/Insertion_VLA_Sim2/TRAIN/SmolVLA/lerobot_multi_datasets/new_dataset5_real",
        "needs_patching": True
    }
]

EXPECTED_ROBOT_TYPE = 'surgical_robot_6dof'
VIEW_TO_CAMERA_MAP = {
    'observation.images.View1': 'observation.images.camera1',
    'observation.images.View2': 'observation.images.camera2',
    'observation.images.View3': 'observation.images.camera3',
}


def patch_dataset_metadata_and_structure(dataset_path: Path):
    """
    Patches a LeRobot dataset so its schema matches the expected 'eye_trocar_real' baseline:
    1. Sets robot_type = 'surgical_robot_6dof'
    2. Maps features View1, View2, View3 -> camera1, camera2, camera3
    3. Renames physical video directories
    4. Updates Parquet column headers
    """
    if not dataset_path.exists():
        return

    print(f"\n[Patching] {dataset_path.name}")
    
    # ---------------------------------------------------------
    # 1. Patch info.json (robot_type and camera features)
    # ---------------------------------------------------------
    info_path = dataset_path / 'meta' / 'info.json'
    if info_path.exists():
        try:
            with open(info_path, 'r') as f:
                data = json.load(f)
            
            changed = False
            
            # Patch Robot Type
            if data.get('robot_type') != EXPECTED_ROBOT_TYPE:
                data['robot_type'] = EXPECTED_ROBOT_TYPE
                changed = True
                
            # Patch Features
            features = data.get('features', {})
            new_features = {}
            for key, val in features.items():
                if key in VIEW_TO_CAMERA_MAP:
                    new_features[VIEW_TO_CAMERA_MAP[key]] = val
                    changed = True
                elif key.startswith('observation.images.View'):
                    # Drop extra views (View4, View5) to match the 3-camera baseline
                    changed = True
                else:
                    new_features[key] = val
                    
            if changed:
                data['features'] = new_features
                with open(info_path, 'w') as f:
                    json.dump(data, f, indent=4)
                print("  - Updated meta/info.json (robot_type & features)")
        except Exception as e:
            print(f"  - Error patching info.json: {e}")

    # ---------------------------------------------------------
    # 2. Rename Physical Video Directories
    # ---------------------------------------------------------
    videos_dir = dataset_path / "videos"
    if videos_dir.exists():
        # Rename 1 to 3
        for i in range(1, 4):
            old_dir = videos_dir / f"observation.images.View{i}"
            new_dir = videos_dir / f"observation.images.camera{i}"
            if old_dir.exists():
                old_dir.rename(new_dir)
                print(f"  - Renamed video dir: View{i} -> camera{i}")
        
        # Remove 4 and 5
        for i in [4, 5]:
            old_dir = videos_dir / f"observation.images.View{i}"
            if old_dir.exists():
                shutil.rmtree(old_dir)
                print(f"  - Removed unused video dir: View{i}")

    # ---------------------------------------------------------
    # 3. Patch Parquet Table Columns
    # ---------------------------------------------------------
    meta_episodes_dir = dataset_path / "meta" / "episodes"
    if meta_episodes_dir.exists():
        for parquet_path in meta_episodes_dir.rglob("*.parquet"):
            try:
                table = pq.read_table(parquet_path)
                col_names = table.column_names
                
                columns_to_drop = []
                new_names = []
                
                needs_rewrite = False
                for name in col_names:
                    if 'View4' in name or 'View5' in name:
                        columns_to_drop.append(name)
                        needs_rewrite = True
                    elif 'View1' in name:
                        new_names.append(name.replace('View1', 'camera1'))
                        needs_rewrite = True
                    elif 'View2' in name:
                        new_names.append(name.replace('View2', 'camera2'))
                        needs_rewrite = True
                    elif 'View3' in name:
                        new_names.append(name.replace('View3', 'camera3'))
                        needs_rewrite = True
                    else:
                        new_names.append(name)
                
                if needs_rewrite:
                    if columns_to_drop:
                        table = table.drop(columns_to_drop)
                    table = table.rename_columns(new_names)
                    pq.write_table(table, parquet_path)
                    print(f"  - Patched Parquet columns in {parquet_path.name}")
            except Exception as e:
                print(f"  - Error patching Parquet {parquet_path.name}: {e}")


def main():
    print("==================================================")
    print("Dataset Patching and Aggregation Script")
    print("==================================================")
    
    repo_ids = []
    roots = []

    # 1. Patch and collect valid datasets
    for ds in DATASETS_TO_MERGE:
        ds_path = Path(ds["root"])
        if ds_path.exists():
            if ds.get("needs_patching", False):
                patch_dataset_metadata_and_structure(ds_path)
            
            repo_ids.append(ds["repo_id"])
            roots.append(str(ds_path))
            print(f"[Valid] Found dataset: {ds['repo_id']}")
        else:
            print(f"[Skip] Dataset not found: {ds['repo_id']}")

    if len(repo_ids) == 0:
        print("\nError: No valid datasets found to merge.")
        sys.exit(1)

    # 2. Aggregate datasets
    aggr_repo_id = "combined_real_dataset"
    aggr_root = Path(f"/data/public/NAS/Insertion_VLA_Sim2/TRAIN/SmolVLA/lerobot_multi_datasets/{aggr_repo_id}")

    if aggr_root.exists():
        print(f"\n[Cleanup] Removing previous merged dataset at {aggr_root}")
        shutil.rmtree(aggr_root)

    print(f"\n[Merging] Combining {len(repo_ids)} datasets into: {aggr_repo_id}...")
    print("This requires validating metadata and packing all chunks. Please wait.")
    
    aggregate_datasets(
        repo_ids=repo_ids,
        aggr_repo_id=aggr_repo_id,
        roots=roots,
        aggr_root=aggr_root,
    )

    print("\n==================================================")
    print("✅ Successfully merged datasets!")
    print("==================================================")
    print("Update your train_smolvla_sim.sh with:")
    print(f'DATASET_REPO_ID="{aggr_repo_id}"')
    print(f'DATASET_ROOT="{aggr_root}"')


if __name__ == "__main__":
    main()
