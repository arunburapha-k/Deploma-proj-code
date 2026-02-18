"""
augment_offline.py

สคริปต์สำหรับทำ Offline Data Augmentation
เฉพาะ train set (เช่น data/processed_train)

⚠️ เวอร์ชันนี้ถูกปรับให้ตรงกับโครงสร้างฟีเจอร์ของโปรเจกต์นี้แล้ว:
    - Pose   : 33 จุด × 4 ค่า (x, y, z, visibility) = 132
    - L-Hand : 21 จุด × 3 ค่า (x, y, z)           = 63
    - R-Hand : 21 จุด × 3 ค่า (x, y, z)           = 63
    รวมทั้งหมด                                  = 258 ค่า ต่อเฟรม

การทำงาน:
- หา .npy ทุกไฟล์ใน data/processed_train/<action>/*
- ข้ามไฟล์ที่เคย augment แล้ว (ชื่อไฟล์ลงท้ายด้วย suffix ต่าง ๆ)
- จากแต่ละไฟล์ต้นฉบับ (shape = (30, 258)) สร้าง:
    1) flip ทั้งสเกลตันตามแกน x                    -> *_flip.npy
    2) เพิ่ม Gaussian noise                          -> *_noise1.npy
    3) temporal shift เลื่อนเฟรมเล็กน้อย            -> *_tshift.npy
    4) joint dropout (สุ่มดับข้อต่อบางจุด)           -> *_drop.npy
    5) scale + translate (ขยาย/ย่อ + เลื่อนนิดหน่อย) -> *_st.npy
    6) time-warp (ท่าช้า/เร็วต่างกันเล็กน้อย)        -> *_tw.npy
    7) partial-sequence (เห็นท่าไม่ครบ)              -> *_ps.npy
    8) prefix/suffix no_action                        -> *_psna.npy

รันด้วย:
    python augment_offline.py
"""

import os
import numpy as np
import random

# ================== CONFIG ทั่วไป ==================

ROOT_DIR = "data"
TRAIN_DIR = os.path.join(ROOT_DIR, "processed_train")

RANDOM_SEED = 42

SEQ_LEN = 30
FEAT_DIM = 258

# โครงสร้างฟีเจอร์ตาม extractkeypoint.py
POSE_LM = 33
POSE_DIM = 4
HAND_LM = 21
HAND_DIM = 3

POSE_SIZE = POSE_LM * POSE_DIM  # 33*4 = 132
LH_START = POSE_SIZE  # 132
LH_SIZE = HAND_LM * HAND_DIM  # 63
RH_START = LH_START + LH_SIZE  # 195
RH_SIZE = HAND_LM * HAND_DIM  # 63
FEATURE_TOTAL = POSE_SIZE + LH_SIZE + RH_SIZE

assert FEATURE_TOTAL == FEAT_DIM, "FEAT_DIM ต้องเท่ากับ Pose(33*4)+LH(21*3)+RH(21*3)=258"

# Config ของแต่ละเทคนิค
NOISE_STD = 0.05 # ของเดิม 0.02
MAX_SHIFT_FRAMES = 5 # ของเดิม 2
JOINT_DROP_PROB = 0.10 # ของเดิม 0.05
SCALE_RANGE = (0.80, 1.20) # ของเดิม (0.9, 1.1)
TRANSLATE_STD = 0.05 # ของเดิม 0.02

TIME_WARP_RANGE = (0.70, 1.30) # ของเดิม (0.85, 1.15)
PARTIAL_KEEP_RANGE = (0.75, 0.95)
PREFIX_MAX_FRAMES = 3
SUFFIX_MAX_FRAMES = 3

NO_ACTION_CLASS_NAME = "no_action"

# สุ่มแบบ deterministic
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ----------------- index helper สำหรับ x,y,z,vis -----------------

# Pose: 0..POSE_SIZE-1 เป็น (x,y,z,vis) interleave ทีละ 4
POSE_X_IDX = np.arange(0, POSE_SIZE, 4)
POSE_Y_IDX = np.arange(1, POSE_SIZE, 4)
POSE_Z_IDX = np.arange(2, POSE_SIZE, 4)
POSE_VIS_IDX = np.arange(3, POSE_SIZE, 4)

# Left hand
LH_X_IDX = LH_START + np.arange(0, LH_SIZE, 3)
LH_Y_IDX = LH_START + np.arange(1, LH_SIZE, 3)
LH_Z_IDX = LH_START + np.arange(2, LH_SIZE, 3)

# Right hand
RH_X_IDX = RH_START + np.arange(0, RH_SIZE, 3)
RH_Y_IDX = RH_START + np.arange(1, RH_SIZE, 3)
RH_Z_IDX = RH_START + np.arange(2, RH_SIZE, 3)

# ใช้ pair ของ landmark index สำหรับสลับซ้าย-ขวา (ให้ match กับ trainbi_gru.py)
POSE_FLIP_PAIRS = np.array(
    [(11, 12), (13, 14), (15, 16), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]
)
POSE_FLIP_INDICES = []
for l_idx, r_idx in POSE_FLIP_PAIRS:
    for d in range(POSE_DIM):  # x,y,z,vis
        POSE_FLIP_INDICES.append((l_idx * POSE_DIM + d, r_idx * POSE_DIM + d))


# ================== UTILITIES ==================


def is_augmented_filename(fname: str) -> bool:
    """
    กันไม่ให้วนไป augment ซ้ำ ๆ บนไฟล์ที่สร้างมาแล้วเอง
    """
    name, ext = os.path.splitext(fname)
    if ext != ".npy":
        return True
    suffixes = ["_flip", "_noise1", "_tshift", "_drop", "_st", "_tw", "_ps", "_psna"]
    return any(name.endswith(suf) for suf in suffixes)


def load_no_action_pool(train_dir: str, class_name: str = "no_action"):
    """
    โหลด sequence ของ no_action มาบางส่วนไว้ใช้สำหรับ prefix/suffix
    """
    pool = []
    na_dir = os.path.join(train_dir, class_name)
    if not os.path.isdir(na_dir):
        print(f"[INFO] ไม่พบโฟลเดอร์ no_action สำหรับ prefix/suffix: {na_dir}")
        return pool

    files = sorted(
        [
            f
            for f in os.listdir(na_dir)
            if f.endswith(".npy") and not is_augmented_filename(f)
        ]
    )

    for fname in files:
        path = os.path.join(na_dir, fname)
        try:
            seq = np.load(path)
        except Exception as e:
            print(f"[WARN] โหลด no_action ไม่สำเร็จ: {path} | {e}")
            continue

        if seq.ndim == 2 and seq.shape[0] == SEQ_LEN and seq.shape[1] == FEAT_DIM:
            pool.append(seq)

    print(f"[INFO] โหลด no_action pool มาใช้ได้ {len(pool)} sequences")
    return pool


# ================== ฟังก์ชัน augment ระดับเฟรมเดียว (1D = 258) ==================
def flip_keypoints_frame(keypoints: np.ndarray) -> np.ndarray:
    """
    flip skeleton ทั้งตัวในเฟรมเดียว (ฉบับแก้ไขสำหรับ Relative Coordinates)
    - flip แกน x: เปลี่ยนเครื่องหมาย (x -> -x)
    - สลับ landmark ซ้าย/ขวา
    """
    flipped = np.copy(keypoints)

    # 1) flip x pose (แก้ตรงนี้: ใช้ -x แทน 1.0-x)
    flipped[POSE_X_IDX] = -flipped[POSE_X_IDX]

    # 2) flip x มือ (แก้ตรงนี้เหมือนกัน)
    flipped[LH_X_IDX] = -flipped[LH_X_IDX]
    flipped[RH_X_IDX] = -flipped[RH_X_IDX]

    # 3) swap pose left/right landmarks ทั้ง 4 ค่า (x,y,z,vis)
    for l_flat, r_flat in POSE_FLIP_INDICES:
        flipped[l_flat], flipped[r_flat] = flipped[r_flat], flipped[l_flat]

    # 4) swap LH/RH block ทั้งก้อน
    lh_block = np.copy(flipped[LH_START : LH_START + LH_SIZE])
    rh_block = np.copy(flipped[RH_START : RH_START + RH_SIZE])
    flipped[LH_START : LH_START + LH_SIZE] = rh_block
    flipped[RH_START : RH_START + RH_SIZE] = lh_block

    return flipped

def add_gaussian_noise_frame(keypoints: np.ndarray, std: float = 0.02) -> np.ndarray:
    """
    เพิ่ม Gaussian noise ให้ vector 1 เฟรม
    """
    noise = np.random.normal(loc=0.0, scale=std, size=keypoints.shape)
    return keypoints + noise


# ================== ฟังก์ชัน augment ระดับ sequence (T, 258) ==================


def horizontal_flip_sequence(seq: np.ndarray) -> np.ndarray:
    """
    flip ทั้ง sequence โดยใช้ flip_keypoints_frame frame-by-frame
    """
    flipped_seq = np.empty_like(seq)
    for t in range(seq.shape[0]):
        flipped_seq[t] = flip_keypoints_frame(seq[t])
    return flipped_seq


def add_gaussian_noise(seq: np.ndarray, std: float = NOISE_STD) -> np.ndarray:
    """
    เพิ่ม Gaussian noise ให้ทั้ง sequence
    """
    noise = np.random.normal(loc=0.0, scale=std, size=seq.shape)
    return seq + noise


def temporal_shift(seq: np.ndarray, max_shift: int = MAX_SHIFT_FRAMES) -> np.ndarray:
    """
    เลื่อน sequence ไปข้างหน้า/ข้างหลังเล็กน้อย
    - shift > 0: ขยับไปข้างหน้า เติมหัวด้วยเฟรมแรก
    - shift < 0: ขยับไปข้างหลัง เติมท้ายด้วยเฟรมสุดท้าย
    """
    T = seq.shape[0]
    if T <= 1 or max_shift <= 0:
        return seq.copy()

    shift = np.random.randint(-max_shift, max_shift + 1)
    if shift == 0:
        return seq.copy()

    shifted = np.empty_like(seq)
    if shift > 0:
        shifted[shift:] = seq[:-shift]
        shifted[:shift] = seq[0]
    else:
        k = -shift
        shifted[:-k] = seq[k:]
        shifted[-k:] = seq[-1]

    return shifted


def joint_dropout(seq: np.ndarray, drop_prob: float = JOINT_DROP_PROB) -> np.ndarray:
    """
    สุ่มดับข้อต่อบางจุดตลอดทั้ง sequence
    - Pose: ดับชุด (x,y,z,vis)
    - Hand: ดับชุด (x,y,z)
    """
    dropped = seq.copy()

    # Pose joints
    for j in range(POSE_LM):
        if random.random() < drop_prob:
            base = j * POSE_DIM
            dropped[:, base : base + POSE_DIM] = 0.0

    # Left hand joints
    for j in range(HAND_LM):
        if random.random() < drop_prob:
            base = LH_START + j * HAND_DIM
            dropped[:, base : base + HAND_DIM] = 0.0

    # Right hand joints
    for j in range(HAND_LM):
        if random.random() < drop_prob:
            base = RH_START + j * HAND_DIM
            dropped[:, base : base + HAND_DIM] = 0.0

    return dropped


def scale_translate(
    seq: np.ndarray, scale_range=SCALE_RANGE, translate_std=TRANSLATE_STD
) -> np.ndarray:
    """
    (Version: Relative Coordinates)
    ขยาย/ย่อ + เลื่อนตำแหน่ง
    """
    st = seq.copy()
    T = st.shape[0]

    # รวม index x,y ของทุกจุด
    x_idx = np.concatenate([POSE_X_IDX, LH_X_IDX, RH_X_IDX])
    y_idx = np.concatenate([POSE_Y_IDX, LH_Y_IDX, RH_Y_IDX])

    scale = np.random.uniform(scale_range[0], scale_range[1])
    tx = np.random.normal(loc=0.0, scale=translate_std)
    ty = np.random.normal(loc=0.0, scale=translate_std)

    # จุดหมุนคือ (0,0) เพราะเป็นพิกัดเทียบไหล่
    cx, cy = 0.0, 0.0

    for t in range(T):
        st[t, x_idx] = (st[t, x_idx] - cx) * scale + cx + tx
        st[t, y_idx] = (st[t, y_idx] - cy) * scale + cy + ty

    # ไม่ต้อง Clip ค่า (เพราะ Relative Coordinate ติดลบได้)

    return st


def time_warp(seq: np.ndarray, scale_range=TIME_WARP_RANGE) -> np.ndarray:
    """
    ปรับความเร็วของ sequence โดย warp เวลาแบบง่าย ๆ (nearest-neighbor)
    ยังคงได้ความยาวเท่าเดิม (SEQ_LEN)
    """
    T, F = seq.shape
    scale = np.random.uniform(scale_range[0], scale_range[1])

    new_seq = np.empty_like(seq)
    for t in range(T):
        src_pos = t * scale
        if src_pos >= T:
            src_pos = T - 1
        idx = int(round(src_pos))
        idx = max(0, min(T - 1, idx))
        new_seq[t] = seq[idx]

    return new_seq


def partial_sequence(seq: np.ndarray, keep_range=PARTIAL_KEEP_RANGE) -> np.ndarray:
    """
    ใช้เฉพาะ sub-sequence ยาว 75–95% ของเดิม แล้ว resample กลับมาเป็น SEQ_LEN เฟรม
    จำลองกรณีกล้องจับท่าไม่ครบทั้งช่วง
    """
    T, F = seq.shape
    keep_ratio = np.random.uniform(keep_range[0], keep_range[1])
    keep_len = max(2, int(round(T * keep_ratio)))
    if keep_len >= T:
        return seq.copy()

    start = np.random.randint(0, T - keep_len + 1)
    sub = seq[start : start + keep_len]

    indices = np.linspace(0, keep_len - 1, num=T).astype(int)
    return sub[indices]


def prefix_suffix_no_action(
    seq: np.ndarray,
    no_action_pool,
    max_prefix_frames: int = PREFIX_MAX_FRAMES,
    max_suffix_frames: int = SUFFIX_MAX_FRAMES,
) -> np.ndarray:
    """
    เอาเฟรมจาก no_action มาต่อหน้า/หลังท่า แล้ว resample กลับเป็น SEQ_LEN
    ใช้เฉพาะตอน action ไม่ใช่ no_action เอง
    """
    if not no_action_pool:
        return seq.copy()

    T, F = seq.shape

    na_seq = random.choice(no_action_pool)
    if na_seq.shape != seq.shape:
        return seq.copy()

    prefix_len = np.random.randint(0, max_prefix_frames + 1)
    suffix_len = np.random.randint(0, max_suffix_frames + 1)

    if prefix_len == 0 and suffix_len == 0:
        return seq.copy()

    prefix = na_seq[:prefix_len] if prefix_len > 0 else np.empty((0, F))
    suffix = na_seq[-suffix_len:] if suffix_len > 0 else np.empty((0, F))

    combined = np.concatenate([prefix, seq, suffix], axis=0)
    total_len = combined.shape[0]

    indices = np.linspace(0, total_len - 1, num=T).astype(int)
    return combined[indices]


# ================== MAIN AUGMENT FOR ONE FILE ==================


def augment_file(action_dir: str, fname: str, no_action_pool, is_no_action_class: bool):
    """
    ทำ augmentation ให้ไฟล์เดียว (เฉพาะไฟล์ต้นฉบับที่ยังไม่ augment)
    """
    path = os.path.join(action_dir, fname)
    try:
        seq = np.load(path)
    except Exception as e:
        print(f"[SKIP] โหลดไฟล์ไม่สำเร็จ: {path} | error: {e}")
        return

    if seq.ndim != 2 or seq.shape[1] != FEAT_DIM:
        print(f"[WARN] shape แปลก (คาดว่า (T,{FEAT_DIM})): {path}, shape={seq.shape}")
        return

    T, D = seq.shape
    if T != SEQ_LEN:
        print(f"[WARN] seq_len ไม่เท่ากับ {SEQ_LEN}: {path}, T={T}")
    print(f"  - {fname}: seq_len={T}, feat={D}")

    base_name, _ = os.path.splitext(fname)

    # 1) Horizontal Flip
    flip_seq = horizontal_flip_sequence(seq)
    flip_name = f"{base_name}_flip.npy"
    np.save(os.path.join(action_dir, flip_name), flip_seq)
    print(f"    -> saved {flip_name}")

    # 2) Noise
    noise_seq = add_gaussian_noise(seq, NOISE_STD)
    noise_name = f"{base_name}_noise1.npy"
    np.save(os.path.join(action_dir, noise_name), noise_seq)
    print(f"    -> saved {noise_name}")

    # 3) Temporal Shift
    tshift_seq = temporal_shift(seq, MAX_SHIFT_FRAMES)
    tshift_name = f"{base_name}_tshift.npy"
    np.save(os.path.join(action_dir, tshift_name), tshift_seq)
    print(f"    -> saved {tshift_name}")

    # 4) Joint Dropout
    drop_seq = joint_dropout(seq, JOINT_DROP_PROB)
    drop_name = f"{base_name}_drop.npy"
    np.save(os.path.join(action_dir, drop_name), drop_seq)
    print(f"    -> saved {drop_name}")

    # 5) Scale + Translate
    st_seq = scale_translate(seq, SCALE_RANGE, TRANSLATE_STD)
    st_name = f"{base_name}_st.npy"
    np.save(os.path.join(action_dir, st_name), st_seq)
    print(f"    -> saved {st_name}")

    # 6) Time-warp
    tw_seq = time_warp(seq, TIME_WARP_RANGE)
    tw_name = f"{base_name}_tw.npy"
    np.save(os.path.join(action_dir, tw_name), tw_seq)
    print(f"    -> saved {tw_name}")

    # 7) Partial-sequence
    ps_seq = partial_sequence(seq, PARTIAL_KEEP_RANGE)
    ps_name = f"{base_name}_ps.npy"
    np.save(os.path.join(action_dir, ps_name), ps_seq)
    print(f"    -> saved {ps_name}")

    # 8) Prefix/Suffix no_action (ไม่ทำถ้าเป็นคลาส no_action เอง)
    if (not is_no_action_class) and no_action_pool:
        psna_seq = prefix_suffix_no_action(
            seq, no_action_pool, PREFIX_MAX_FRAMES, SUFFIX_MAX_FRAMES
        )
        psna_name = f"{base_name}_psna.npy"
        np.save(os.path.join(action_dir, psna_name), psna_seq)
        print(f"    -> saved {psna_name}")


# ================== MAIN ==================


def main():
    if not os.path.isdir(TRAIN_DIR):
        print(f"[ERROR] ไม่พบโฟลเดอร์ TRAIN_DIR: {TRAIN_DIR}")
        print("กรุณาเช็ค path หรือสร้าง data/processed_train ก่อน")
        return

    # โหลด no_action pool ก่อน (ถ้ามี)
    no_action_pool = load_no_action_pool(TRAIN_DIR, NO_ACTION_CLASS_NAME)

    # หา action folders ทั้งหมดใน processed_train
    actions = [
        d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))
    ]

    print("พบ action ทั้งหมด:", actions)

    for action in actions:
        action_dir = os.path.join(TRAIN_DIR, action)
        print(f"\n=== Action: {action} ===")

        files = sorted(
            [
                f
                for f in os.listdir(action_dir)
                if f.endswith(".npy") and not is_augmented_filename(f)
            ]
        )

        if not files:
            print("  (ไม่มีไฟล์ใหม่ให้ augment หรือไฟล์ทั้งหมดถูก augment แล้ว)")
            continue

        print(f"  จำนวนไฟล์ต้นฉบับ: {len(files)}")

        is_no_action_class = action == NO_ACTION_CLASS_NAME

        for fname in files:
            augment_file(action_dir, fname, no_action_pool, is_no_action_class)


if __name__ == "__main__":
    main()
