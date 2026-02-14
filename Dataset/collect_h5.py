"""
Dataset 폴더 내 모든 .h5 파일을 하나의 폴더로 모으는 스크립트.
- 원본 파일은 그대로 두고 복사(copy) 합니다.
- 파일명 충돌 시 원본 폴더 이름을 prefix로 붙여 구분합니다.
"""

import os
import shutil
from pathlib import Path

SRC_DIR = Path("/data/public/NAS/Insertion_VLA_Sim2/Dataset")
DST_DIR = SRC_DIR / "all_h5"


def main():
    DST_DIR.mkdir(parents=True, exist_ok=True)

    h5_files = sorted(SRC_DIR.rglob("*.h5"))
    # all_h5 폴더 자체에서 찾은 파일은 제외
    h5_files = [f for f in h5_files if not f.is_relative_to(DST_DIR)]

    print(f"Found {len(h5_files)} .h5 files")

    copied = 0
    skipped = 0
    seen_names = {}  # filename -> source path (충돌 감지용)

    for src in h5_files:
        fname = src.name

        # 파일명 충돌 처리: 상위 폴더 이름(New0, New1 등)을 prefix로 추가
        if fname in seen_names and seen_names[fname] != src:
            # 상위 2단계 폴더명 사용 (e.g. New0)
            parent_name = src.relative_to(SRC_DIR).parts[0]
            fname = f"{parent_name}_{fname}"

        dst = DST_DIR / fname

        if dst.exists():
            skipped += 1
            continue

        seen_names[src.name] = src
        shutil.copy2(src, dst)
        copied += 1

        if copied % 1000 == 0:
            print(f"  Copied {copied} files...")

    print(f"Done! Copied: {copied}, Skipped (already exists): {skipped}")
    print(f"Destination: {DST_DIR}")


if __name__ == "__main__":
    main()
