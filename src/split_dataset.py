"""
split_dataset.py

อ่าน .npy จาก data/processed/<action>
แล้วสุ่มแบ่งเป็น train / val / test ตามสัดส่วนที่กำหนด
จากนั้น copy ไปไว้ที่

  data/processed_train/<action>/
  data/processed_val/<action>/
  data/processed_test/<action>/

รันจาก root โปรเจกต์:
    python src/split_dataset.py
"""

import os
import random
import shutil
from pathlib import Path

# ---------------- CONFIG ----------------
RANDOM_SEED = 42

# สัดส่วน train / val / test
TRAIN_RATIO = 0.6
VAL_RATIO = 0.20  # ที่เหลือเป็น test

# ชื่อคลาสให้ตรงกับที่คุณใช้จริง
ACTIONS = [
    'distention',
    'fever',
    'feverish',
    'no_action',
    'wounded'
]

# หา path base จากตำแหน่งไฟล์นี้ (อยู่ใน src/)
BASE_DIR = Path(__file__).resolve().parents[1]  # โฟลเดอร์โปรเจกต์หลัก
DATA_DIR = BASE_DIR / "data"

SRC_DIR = DATA_DIR / "processed"
TRAIN_OUTDIR = DATA_DIR / "processed_train"
VAL_OUTDIR = DATA_DIR / "processed_val"
TEST_OUTDIR = DATA_DIR / "processed_test"
# ----------------------------------------


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def split_and_copy_for_action(action: str):
    src_action_dir = SRC_DIR / action
    if not src_action_dir.is_dir():
        print(f"[WARN] ไม่พบโฟลเดอร์ {src_action_dir}, ข้าม…")
        return

    # list เฉพาะ .npy
    files = sorted([f for f in src_action_dir.iterdir() if f.suffix == ".npy"])

    if not files:
        print(f"[WARN] ไม่มีไฟล์ .npy ใน {src_action_dir}, ข้าม…")
        return

    # สุ่มลำดับ (fix seed เพื่อ reproducible)
    random.shuffle(files)

    n_total = len(files)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)
    n_test = n_total - n_train - n_val

    train_files = files[:n_train]
    val_files = files[n_train : n_train + n_val]
    test_files = files[n_train + n_val :]

    print(f"Action: {action}")
    print(f"  total = {n_total}")
    print(
        f"  train = {len(train_files)}, val = {len(val_files)}, test = {len(test_files)}"
    )

    # เตรียมโฟลเดอร์ปลายทางของ action นั้น ๆ
    train_dst_dir = TRAIN_OUTDIR / action
    val_dst_dir = VAL_OUTDIR / action
    test_dst_dir = TEST_OUTDIR / action

    ensure_dir(train_dst_dir)
    ensure_dir(val_dst_dir)
    ensure_dir(test_dst_dir)

    # copy แต่ละชุด
    for src_path in train_files:
        shutil.copy2(src_path, train_dst_dir / src_path.name)
    for src_path in val_files:
        shutil.copy2(src_path, val_dst_dir / src_path.name)
    for src_path in test_files:
        shutil.copy2(src_path, test_dst_dir / src_path.name)


def main():
    random.seed(RANDOM_SEED)

    print("Base dir:", BASE_DIR)
    print("อ่านข้อมูลจาก:", SRC_DIR)

    for action in ACTIONS:
        split_and_copy_for_action(action)

    print("\n✅ เสร็จแล้ว ลองไปเช็คโฟลเดอร์:")
    print(f"  {TRAIN_OUTDIR}")
    print(f"  {VAL_OUTDIR}")
    print(f"  {TEST_OUTDIR}")


if __name__ == "__main__":
    main()
