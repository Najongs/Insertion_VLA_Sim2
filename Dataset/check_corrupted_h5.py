"""Scan all HDF5 files and move corrupted ones to a separate folder."""
import h5py
import os
import shutil
import glob
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

H5_DIR = "/data/public/NAS/Insertion_VLA_Sim2/Dataset/all_h5"
CORRUPTED_DIR = "/data/public/NAS/Insertion_VLA_Sim2/Dataset/corrupted_h5"


def visit_all(obj):
    """Recursively visit all groups/datasets to detect deep corruption."""
    if isinstance(obj, h5py.Dataset):
        # Read a small slice to trigger actual data access
        if len(obj.shape) > 0 and obj.shape[0] > 0:
            _ = obj[0]
    elif isinstance(obj, h5py.Group):
        for key in obj.keys():
            visit_all(obj[key])


def check_file(path):
    """Try to open an HDF5 file and deeply read all objects. Returns (path, error) or (path, None)."""
    try:
        with h5py.File(path, "r") as f:
            visit_all(f)
        return path, None
    except Exception as e:
        return path, str(e)


def main():
    files = sorted(glob.glob(os.path.join(H5_DIR, "*.h5")))
    print(f"Total files to check: {len(files)}")

    corrupted = []
    checked = 0

    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(check_file, f): f for f in files}
        for future in as_completed(futures):
            path, error = future.result()
            checked += 1
            if error:
                corrupted.append((path, error))
                print(f"  CORRUPTED: {os.path.basename(path)} - {error}")
            if checked % 5000 == 0:
                print(f"  Checked {checked}/{len(files)}...")

    print(f"\nResults: {len(corrupted)} corrupted out of {len(files)} total")

    if corrupted:
        os.makedirs(CORRUPTED_DIR, exist_ok=True)
        print(f"\nMoving corrupted files to {CORRUPTED_DIR}:")
        for path, error in corrupted:
            dest = os.path.join(CORRUPTED_DIR, os.path.basename(path))
            shutil.move(path, dest)
            print(f"  Moved: {os.path.basename(path)}")
        print(f"\nDone. Moved {len(corrupted)} files.")
    else:
        print("No corrupted files found!")


if __name__ == "__main__":
    main()
