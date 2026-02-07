import os
import random
import shutil
from pathlib import Path

# ---------------- CONFIG ----------------
RANDOM_SEED = 42

# สัดส่วน train / val / test
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15  # ที่เหลือเป็น test (0.10)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
SRC_DIR = DATA_DIR / "processed"

TRAIN_OUTDIR = DATA_DIR / "processed_train"
VAL_OUTDIR = DATA_DIR / "processed_val"
TEST_OUTDIR = DATA_DIR / "processed_test"
# ----------------------------------------

def ensure_clean_dir(path: Path):
    """สร้างโฟลเดอร์ใหม่ ถ้ามีของเก่าให้ลบทิ้งก่อน (กันไฟล์ขยะค้าง)"""
    if path.exists():
        shutil.rmtree(path) # ลบของเก่าทิ้ง
    path.mkdir(parents=True, exist_ok=True)

def split_and_copy_for_action(action: str):
    src_action_dir = SRC_DIR / action
    if not src_action_dir.is_dir():
        return

    # list เฉพาะ .npy
    files = sorted([f for f in src_action_dir.iterdir() if f.suffix == ".npy"])
    
    if not files:
        print(f"[SKIP] {action}: ไม่พบไฟล์ .npy")
        return

    # สุ่มลำดับ
    random.shuffle(files)

    n_total = len(files)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)
    # n_test คือส่วนที่เหลือ

    train_files = files[:n_train]
    val_files = files[n_train : n_train + n_val]
    test_files = files[n_train + n_val :]

    print(f"Action: {action:<15} | Total: {n_total:<4} -> Train: {len(train_files):<4} Val: {len(val_files):<4} Test: {len(test_files):<4}")

    # สร้างโฟลเดอร์ปลายทาง
    for category, file_list in [("train", train_files), ("val", val_files), ("test", test_files)]:
        if category == "train": dst_base = TRAIN_OUTDIR
        elif category == "val": dst_base = VAL_OUTDIR
        else: dst_base = TEST_OUTDIR
        
        target_dir = dst_base / action
        target_dir.mkdir(parents=True, exist_ok=True)
        
        for src_path in file_list:
            shutil.copy2(src_path, target_dir / src_path.name)

def main():
    random.seed(RANDOM_SEED)
    print(f"Reading from: {SRC_DIR}\n")

    # 1. อ่านชื่อ Class อัตโนมัติจากโฟลเดอร์ (ไม่ต้องแก้ code บ่อยๆ)
    actions = [d.name for d in SRC_DIR.iterdir() if d.is_dir()]
    actions.sort()
    
    if not actions:
        print("❌ ไม่พบโฟลเดอร์ Class ใน data/processed เลย!")
        return

    # 2. ล้างโฟลเดอร์ปลายทางก่อนเริ่ม
    print("Clearing old datasets...")
    ensure_clean_dir(TRAIN_OUTDIR)
    ensure_clean_dir(VAL_OUTDIR)
    ensure_clean_dir(TEST_OUTDIR)

    # 3. เริ่มแบ่งไฟล์
    print("-" * 60)
    for action in actions:
        split_and_copy_for_action(action)
    print("-" * 60)

    print("\n✅ เสร็จสิ้น! ข้อมูลพร้อมเทรนแล้วที่:")
    print(f"  Train: {TRAIN_OUTDIR}")
    print(f"  Val:   {VAL_OUTDIR}")
    print(f"  Test:  {TEST_OUTDIR}")

if __name__ == "__main__":
    main()