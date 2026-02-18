"""
augment_offline.py (Version: 183 Features + Low FPS)

⚠️ เวอร์ชันนี้ปรับโครงสร้างฟีเจอร์เป็น 183 (ตัด Z ออก):
    - Pose   : 33 จุด × 3 ค่า (x, y, vis) = 99
    - L-Hand : 21 จุด × 2 ค่า (x, y)      = 42
    - R-Hand : 21 จุด × 2 ค่า (x, y)      = 42
    รวมทั้งหมด                                = 183 ค่า ต่อเฟรม
"""

import os
import numpy as np
import random

# ================== CONFIG ทั่วไป ==================

ROOT_DIR = "data"
TRAIN_DIR = os.path.join(ROOT_DIR, "processed_train")

RANDOM_SEED = 42

SEQ_LEN = 30
# 🔥 แก้ Dimension เป็น 183
FEAT_DIM = 183 

# 🔥 แก้โครงสร้างย่อย
POSE_LM = 33
POSE_DIM = 3    # x, y, vis (ไม่มี z)
HAND_LM = 21
HAND_DIM = 2    # x, y (ไม่มี z)

POSE_SIZE = POSE_LM * POSE_DIM  # 99
LH_START = POSE_SIZE  # 99
LH_SIZE = HAND_LM * HAND_DIM  # 42
RH_START = LH_START + LH_SIZE  # 141
RH_SIZE = HAND_LM * HAND_DIM  # 42
FEATURE_TOTAL = POSE_SIZE + LH_SIZE + RH_SIZE

assert FEATURE_TOTAL == FEAT_DIM, f"FEAT_DIM ต้องเท่ากับ {FEATURE_TOTAL} (Pose99+LH42+RH42)"

# Config (Aggressive for Mobile)
NOISE_STD = 0.05
MAX_SHIFT_FRAMES = 5
JOINT_DROP_PROB = 0.10
SCALE_RANGE = (0.80, 1.20)
TRANSLATE_STD = 0.05

TIME_WARP_RANGE = (0.70, 1.30)
PARTIAL_KEEP_RANGE = (0.75, 0.95)
PREFIX_MAX_FRAMES = 3
SUFFIX_MAX_FRAMES = 3
LOW_FPS_DROP_RATE = 0.5 

NO_ACTION_CLASS_NAME = "no_action"

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ----------------- index helper (Updated for 183) -----------------
# Pose: 0..98 เป็น (x,y,vis) interleave ทีละ 3
POSE_X_IDX = np.arange(0, POSE_SIZE, 3)
POSE_Y_IDX = np.arange(1, POSE_SIZE, 3)
# ไม่มี Z แล้ว

# Left hand: x,y interleave ทีละ 2
LH_X_IDX = LH_START + np.arange(0, LH_SIZE, 2)
LH_Y_IDX = LH_START + np.arange(1, LH_SIZE, 2)

# Right hand
RH_X_IDX = RH_START + np.arange(0, RH_SIZE, 2)
RH_Y_IDX = RH_START + np.arange(1, RH_SIZE, 2)

# Pair สำหรับ Flip (ยังต้องใช้อยู่)
POSE_FLIP_PAIRS = np.array(
    [(11, 12), (13, 14), (15, 16), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]
)
POSE_FLIP_INDICES = []
for l_idx, r_idx in POSE_FLIP_PAIRS:
    for d in range(POSE_DIM): # วนแค่ 0,1,2 (x,y,vis)
        POSE_FLIP_INDICES.append((l_idx * POSE_DIM + d, r_idx * POSE_DIM + d))


# ================== UTILITIES ==================

def is_augmented_filename(fname: str) -> bool:
    name, ext = os.path.splitext(fname)
    if ext != ".npy": return True
    suffixes = ["_flip", "_noise1", "_tshift", "_drop", "_st", "_tw", "_ps", "_psna", "_lowfps"]
    return any(name.endswith(suf) for suf in suffixes)

def load_no_action_pool(train_dir: str, class_name: str = "no_action"):
    pool = []
    na_dir = os.path.join(train_dir, class_name)
    if not os.path.isdir(na_dir):
        print(f"[INFO] ไม่พบโฟลเดอร์ no_action สำหรับ prefix/suffix: {na_dir}")
        return pool
    files = sorted([f for f in os.listdir(na_dir) if f.endswith(".npy") and not is_augmented_filename(f)])
    for fname in files:
        path = os.path.join(na_dir, fname)
        try:
            seq = np.load(path)
        except: continue
        if seq.ndim == 2 and seq.shape[0] == SEQ_LEN and seq.shape[1] == FEAT_DIM:
            pool.append(seq)
    print(f"[INFO] โหลด no_action pool มาใช้ได้ {len(pool)} sequences")
    return pool

# ================== Augmentation Functions ==================

def flip_keypoints_frame(keypoints: np.ndarray) -> np.ndarray:
    flipped = np.copy(keypoints)
    
    # 1. Flip X (กลับเครื่องหมาย)
    flipped[POSE_X_IDX] = -flipped[POSE_X_IDX]
    flipped[LH_X_IDX] = -flipped[LH_X_IDX]
    flipped[RH_X_IDX] = -flipped[RH_X_IDX]

    # 2. Swap Left/Right Body Landmarks
    for l_flat, r_flat in POSE_FLIP_INDICES:
        flipped[l_flat], flipped[r_flat] = flipped[r_flat], flipped[l_flat]

    # 3. Swap Hands (ทั้งก้อน)
    lh_block = np.copy(flipped[LH_START : LH_START + LH_SIZE])
    rh_block = np.copy(flipped[RH_START : RH_START + RH_SIZE])
    flipped[LH_START : LH_START + LH_SIZE] = rh_block
    flipped[RH_START : RH_START + RH_SIZE] = lh_block

    return flipped

def horizontal_flip_sequence(seq: np.ndarray) -> np.ndarray:
    flipped_seq = np.empty_like(seq)
    for t in range(seq.shape[0]):
        flipped_seq[t] = flip_keypoints_frame(seq[t])
    return flipped_seq

def add_gaussian_noise(seq: np.ndarray, std: float = NOISE_STD) -> np.ndarray:
    noise = np.random.normal(loc=0.0, scale=std, size=seq.shape)
    return seq + noise

def temporal_shift(seq: np.ndarray, max_shift: int = MAX_SHIFT_FRAMES) -> np.ndarray:
    T = seq.shape[0]
    if T <= 1 or max_shift <= 0: return seq.copy()
    shift = np.random.randint(-max_shift, max_shift + 1)
    if shift == 0: return seq.copy()
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
    dropped = seq.copy()
    for j in range(POSE_LM):
        if random.random() < drop_prob:
            base = j * POSE_DIM
            dropped[:, base : base + POSE_DIM] = 0.0
    for j in range(HAND_LM):
        if random.random() < drop_prob:
            base = LH_START + j * HAND_DIM
            dropped[:, base : base + HAND_DIM] = 0.0
    for j in range(HAND_LM):
        if random.random() < drop_prob:
            base = RH_START + j * HAND_DIM
            dropped[:, base : base + HAND_DIM] = 0.0
    return dropped

def scale_translate(seq: np.ndarray, scale_range=SCALE_RANGE, translate_std=TRANSLATE_STD) -> np.ndarray:
    st = seq.copy()
    T = st.shape[0]
    x_idx = np.concatenate([POSE_X_IDX, LH_X_IDX, RH_X_IDX])
    y_idx = np.concatenate([POSE_Y_IDX, LH_Y_IDX, RH_Y_IDX])
    scale = np.random.uniform(scale_range[0], scale_range[1])
    tx = np.random.normal(loc=0.0, scale=translate_std)
    ty = np.random.normal(loc=0.0, scale=translate_std)
    cx, cy = 0.0, 0.0
    for t in range(T):
        st[t, x_idx] = (st[t, x_idx] - cx) * scale + cx + tx
        st[t, y_idx] = (st[t, y_idx] - cy) * scale + cy + ty
    return st

def time_warp(seq: np.ndarray, scale_range=TIME_WARP_RANGE) -> np.ndarray:
    T, F = seq.shape
    scale = np.random.uniform(scale_range[0], scale_range[1])
    new_seq = np.empty_like(seq)
    for t in range(T):
        src_pos = t * scale
        if src_pos >= T: src_pos = T - 1
        idx = int(round(src_pos))
        idx = max(0, min(T - 1, idx))
        new_seq[t] = seq[idx]
    return new_seq

def partial_sequence(seq: np.ndarray, keep_range=PARTIAL_KEEP_RANGE) -> np.ndarray:
    T, F = seq.shape
    keep_ratio = np.random.uniform(keep_range[0], keep_range[1])
    keep_len = max(2, int(round(T * keep_ratio)))
    if keep_len >= T: return seq.copy()
    start = np.random.randint(0, T - keep_len + 1)
    sub = seq[start : start + keep_len]
    indices = np.linspace(0, keep_len - 1, num=T).astype(int)
    return sub[indices]

def prefix_suffix_no_action(seq: np.ndarray, no_action_pool, max_prefix=PREFIX_MAX_FRAMES, max_suffix=SUFFIX_MAX_FRAMES) -> np.ndarray:
    if not no_action_pool: return seq.copy()
    T, F = seq.shape
    na_seq = random.choice(no_action_pool)
    if na_seq.shape != seq.shape: return seq.copy()
    prefix_len = np.random.randint(0, max_prefix + 1)
    suffix_len = np.random.randint(0, max_suffix + 1)
    if prefix_len == 0 and suffix_len == 0: return seq.copy()
    prefix = na_seq[:prefix_len] if prefix_len > 0 else np.empty((0, F))
    suffix = na_seq[-suffix_len:] if suffix_len > 0 else np.empty((0, F))
    combined = np.concatenate([prefix, seq, suffix], axis=0)
    indices = np.linspace(0, combined.shape[0] - 1, num=T).astype(int)
    return combined[indices]

def simulate_low_fps(seq: np.ndarray, drop_rate: float = LOW_FPS_DROP_RATE) -> np.ndarray:
    T = seq.shape[0]
    new_seq = seq.copy()
    for t in range(1, T):
        if np.random.rand() < drop_rate:
            new_seq[t] = new_seq[t-1]
    return new_seq

# ================== MAIN AUGMENT FOR ONE FILE ==================

def augment_file(action_dir: str, fname: str, no_action_pool, is_no_action_class: bool):
    path = os.path.join(action_dir, fname)
    try: seq = np.load(path)
    except Exception as e:
        print(f"[SKIP] {path}: {e}")
        return

    if seq.ndim != 2 or seq.shape[1] != FEAT_DIM:
        print(f"[WARN] shape mismatch: {path}, shape={seq.shape}")
        return

    base_name, _ = os.path.splitext(fname)
    
    # Run all augmentations
    np.save(os.path.join(action_dir, f"{base_name}_flip.npy"), horizontal_flip_sequence(seq))
    np.save(os.path.join(action_dir, f"{base_name}_noise1.npy"), add_gaussian_noise(seq, NOISE_STD))
    np.save(os.path.join(action_dir, f"{base_name}_tshift.npy"), temporal_shift(seq, MAX_SHIFT_FRAMES))
    np.save(os.path.join(action_dir, f"{base_name}_drop.npy"), joint_dropout(seq, JOINT_DROP_PROB))
    np.save(os.path.join(action_dir, f"{base_name}_st.npy"), scale_translate(seq, SCALE_RANGE, TRANSLATE_STD))
    np.save(os.path.join(action_dir, f"{base_name}_tw.npy"), time_warp(seq, TIME_WARP_RANGE))
    np.save(os.path.join(action_dir, f"{base_name}_ps.npy"), partial_sequence(seq, PARTIAL_KEEP_RANGE))
    
    if (not is_no_action_class) and no_action_pool:
        np.save(os.path.join(action_dir, f"{base_name}_psna.npy"), prefix_suffix_no_action(seq, no_action_pool))
        
    np.save(os.path.join(action_dir, f"{base_name}_lowfps.npy"), simulate_low_fps(seq, LOW_FPS_DROP_RATE))
    
    print(f"  -> Augmented {base_name}")

# ================== MAIN ==================

def main():
    if not os.path.isdir(TRAIN_DIR):
        print(f"[ERROR] ไม่พบโฟลเดอร์ TRAIN_DIR: {TRAIN_DIR}")
        return

    no_action_pool = load_no_action_pool(TRAIN_DIR, NO_ACTION_CLASS_NAME)
    actions = [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]

    for action in actions:
        action_dir = os.path.join(TRAIN_DIR, action)
        print(f"\n=== Action: {action} ===")
        files = sorted([f for f in os.listdir(action_dir) if f.endswith(".npy") and not is_augmented_filename(f)])
        
        if not files:
            print("  (No files to augment)")
            continue
            
        is_no_action_class = action == NO_ACTION_CLASS_NAME
        for fname in files:
            augment_file(action_dir, fname, no_action_pool, is_no_action_class)

if __name__ == "__main__":
    main()