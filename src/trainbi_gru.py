import os
import time
import numpy as np
import random
import tensorflow as tf
import json
import math

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Conv1D, Bidirectional,
    GRU, BatchNormalization, MaxPooling1D
)
from tensorflow.keras.optimizers import Adam

# ---------------- 0) EXPERIMENT CONFIG ----------------
# ตรงนี้คือส่วนที่เปลี่ยนบ่อย เวลาอยากลองหลายแบบ

EXPERIMENT_NAME = "exp_gru_v2"
RNN_TYPE = "lstm"                  # "gru" หรือ "lstm"
CONV_FILTERS = 64
RNN_UNITS    = 64
DENSE_UNITS1 = 64
DENSE_UNITS2 = 32
DROPOUT_RATE = 0.4

LEARNING_RATE = 1e-3
NUM_EPOCHS    = 30
BATCH_SIZE    = 32

EARLY_STOPPING_PATIENCE = 10      # ถ้า val_loss ไม่ดีขึ้นต่อเนื่อง 10 epoch จะหยุด

# class balancing
USE_CLASS_WEIGHT      = False      # ส่ง class_weight เข้า model.fit
USE_BALANCED_SAMPLING = False     # ใช้ balanced_data_generator แทน data_generator


# ---------------- 1) CONFIG พื้นฐาน ----------------

DATA_DIR   = "data"
TRAIN_DIR  = os.path.join(DATA_DIR, "processed_train")
VAL_DIR    = os.path.join(DATA_DIR, "processed_val")
TEST_DIR   = os.path.join(DATA_DIR, "processed_test")

# !!! อย่าลืมแก้ตรงนี้ให้ครบทุกท่าที่มีจริงในโฟลเดอร์นะครับ !!!
actions = np.array([
    'distention',
    'fever',
    'feverish',
    'no_action',
    'wounded'
])

sequence_length = 30
num_features    = 258

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


# ---------------- 2) ฟังก์ชันโหลดชุด train / val / test ----------------

def load_split(split_dir):
    """
    อ่าน .npy จาก split_dir/<action>/*
    แล้วคืน X, y_one_hot
    """
    sequences, labels = [], []
    action_map = {action: idx for idx, action in enumerate(actions)}

    print(f"\nLoading split from: {split_dir}")
    for action in actions:
        action_path = os.path.join(split_dir, action)
        if not os.path.isdir(action_path):
            print(f"  [WARN] Missing folder for action '{action}': {action_path}")
            continue

        npy_files = [f for f in os.listdir(action_path) if f.endswith(".npy")]
        npy_files.sort()

        print(f"  Action '{action}': {len(npy_files)} files")
        for npy_file in npy_files:
            npy_path = os.path.join(action_path, npy_file)
            res = np.load(npy_path)
            if res.shape == (sequence_length, num_features):
                sequences.append(res)
                labels.append(action_map[action])
            else:
                print(f"    [Warning] Skipping {npy_path} - incorrect shape: {res.shape}")

    X = np.array(sequences)
    y = np.array(labels)
    y_one_hot = to_categorical(y, num_classes=len(actions))
    print(f"  -> Loaded {X.shape[0]} sequences from {split_dir}")
    return X, y_one_hot


print("Loading datasets (train / val / test)...")

X_train, y_train = load_split(TRAIN_DIR)
X_val,   y_val   = load_split(VAL_DIR)
X_test,  y_test  = load_split(TEST_DIR)

print("\nSummary of dataset shapes:")
print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"  X_val:   {X_val.shape},   y_val:   {y_val.shape}")
print(f"  X_test:  {X_test.shape},  y_test:  {y_test.shape}")


# ---------------- 2.1) คำนวณ class weights จาก y_train ----------------

print("\nComputing class weights from training set...")
y_train_int  = np.argmax(y_train, axis=1)
class_counts = np.bincount(y_train_int, minlength=len(actions))

print("  Class counts:")
for i, cnt in enumerate(class_counts):
    print(f"    {i}: {actions[i]} -> {int(cnt)} samples")

class_weights = {}
total       = float(np.sum(class_counts))
num_classes = len(actions)
for i, cnt in enumerate(class_counts):
    if cnt == 0:
        class_weights[i] = 0.0
    else:
        class_weights[i] = total / (num_classes * float(cnt))

print("  Class weights:")
for i, w in class_weights.items():
    print(f"    {i}: {actions[i]} -> {w:.3f}")


# ---------------- 3) Data Augmentation (ออนไลน์) - เก็บไว้แต่ไม่เรียกใช้ ----------------

# layout ตาม mediapipe pose + hands
POSE_FLIP_PAIRS = np.array([
    (11, 12), (13, 14), (15, 16),
    (23, 24), (25, 26), (27, 28),
    (29, 30), (31, 32),
])

POSE_FLIP_INDICES = []
for l_idx, r_idx in POSE_FLIP_PAIRS:
    for i in range(4):  # x,y,z,visibility
        POSE_FLIP_INDICES.append((l_idx * 4 + i, r_idx * 4 + i))

POSE_SIZE = 33 * 4
LH_START, LH_END = 132, 132 + 63
RH_START, RH_END = 132 + 63, 132 + 63 + 63


def add_noise(keypoints, noise_level=0.01):
    noise = np.random.normal(0, noise_level, keypoints.shape)
    return keypoints + noise


def flip_keypoints(keypoints):
    """
    keypoints: เวกเตอร์ยาว 258 ของเฟรมเดียว
    layout: Pose(33*4) + LH(21*3) + RH(21*3)
    """
    flipped_kps = np.copy(keypoints)

    # flip x pose (ทุก 4 ช่อง: x,y,z,vis)
    flipped_kps[0:POSE_SIZE:4] = 1.0 - flipped_kps[0:POSE_SIZE:4]

    # flip x มือ (ทุก 3 ช่อง: x,y,z)
    flipped_kps[LH_START:RH_END:3] = 1.0 - flipped_kps[LH_START:RH_END:3]

    # swap pose L/R ตาม POSE_FLIP_INDICES
    for l_idx, r_idx in POSE_FLIP_INDICES:
        flipped_kps[l_idx], flipped_kps[r_idx] = (
            flipped_kps[r_idx], flipped_kps[l_idx]
        )

    # swap LH/RH block
    lh_block = np.copy(flipped_kps[LH_START:LH_END])
    rh_block = np.copy(flipped_kps[RH_START:RH_END])
    flipped_kps[LH_START:LH_END] = rh_block
    flipped_kps[RH_START:RH_END] = lh_block

    return flipped_kps


def data_generator(X_data, y_data, batch_size=32, augment=True): # <--- เปลี่ยน default เป็น False
    """
    generator ปกติ: shuffle index ทั้งก้อน แล้วแบ่ง batch
    """
    num_samples = X_data.shape[0]
    indices     = np.arange(num_samples)

    while True:
        np.random.shuffle(indices)
        for start in range(0, num_samples, batch_size):
            end           = min(start + batch_size, num_samples)
            batch_indices = indices[start:end]

            X_batch = X_data[batch_indices]
            y_batch = y_data[batch_indices]

            if augment:
                X_batch_augmented = []
                for sequence in X_batch:
                    augmented_sequence = np.copy(sequence)
                    # flip ทั้ง sequence
                    if random.random() < 0.3:
                        for i in range(sequence.shape[0]):
                            augmented_sequence[i] = flip_keypoints(augmented_sequence[i])
                    # เพิ่ม noise
                    if random.random() < 0.5:
                        for i in range(sequence.shape[0]):
                            augmented_sequence[i] = add_noise(
                                augmented_sequence[i], noise_level=0.005
                            )
                    X_batch_augmented.append(augmented_sequence)
                X_batch = np.array(X_batch_augmented)

            yield X_batch, y_batch


def balanced_data_generator(X_data, y_data, batch_size=32, augment=True): # <--- เปลี่ยน default เป็น False
    """
    generator แบบ class-balanced:
    - random เลือก class แบบ uniform
    - แล้ว random sample ในคลาสนั้น
    """
    num_classes = y_data.shape[1]
    y_int       = np.argmax(y_data, axis=1)

    class_indices = []
    for c in range(num_classes):
        idx_c = np.where(y_int == c)[0]
        if len(idx_c) == 0:
            print(f"[WARN] Class {c} ({actions[c]}) ไม่มี sample ใน X_data")
        class_indices.append(idx_c)

    classes = np.arange(num_classes)

    while True:
        X_batch_list, y_batch_list = [], []

        for _ in range(batch_size):
            c = int(np.random.choice(classes))
            if len(class_indices[c]) == 0:
                continue
            idx = int(np.random.choice(class_indices[c]))
            X_batch_list.append(X_data[idx])
            y_batch_list.append(y_data[idx])

        if len(X_batch_list) == 0:
            continue

        X_batch = np.array(X_batch_list)
        y_batch = np.array(y_batch_list)

        if augment:
            X_batch_augmented = []
            for sequence in X_batch:
                augmented_sequence = np.copy(sequence)
                if random.random() < 0.3:
                    for i in range(sequence.shape[0]):
                        augmented_sequence[i] = flip_keypoints(augmented_sequence[i])
                if random.random() < 0.5:
                    for i in range(sequence.shape[0]):
                        augmented_sequence[i] = add_noise(
                            augmented_sequence[i], noise_level=0.005
                        )
                X_batch_augmented.append(augmented_sequence)
            X_batch = np.array(X_batch_augmented)

        yield X_batch, y_batch


# ---------------- 4) สร้างสถาปัตยกรรมโมเดล ----------------

print("\n=== EXPERIMENT CONFIG ===")
print(f"  EXPERIMENT_NAME        = {EXPERIMENT_NAME}")
print(f"  RNN_TYPE               = {RNN_TYPE}")
print(f"  LEARNING_RATE          = {LEARNING_RATE}")
print(f"  NUM_EPOCHS             = {NUM_EPOCHS}")
print(f"  BATCH_SIZE             = {BATCH_SIZE}")
print(f"  EARLY_STOP_PATIENCE    = {EARLY_STOPPING_PATIENCE}")
print(f"  USE_CLASS_WEIGHT       = {USE_CLASS_WEIGHT}")
print(f"  USE_BALANCED_SAMPLING  = {USE_BALANCED_SAMPLING}")

print("\nBuilding the Hybrid (Conv1D + Bi-RNN) model...")

model = Sequential()

# Conv1D: จับ pattern ระยะสั้น
model.add(Conv1D(
    filters=CONV_FILTERS,
    kernel_size=3,
    activation="relu",
    padding="same",
    input_shape=(sequence_length, num_features)
))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))

# เลือก RNN layer จาก config
rnn_type_lower = RNN_TYPE.lower()
if rnn_type_lower == "gru":
    rnn_layer_cls = GRU
elif rnn_type_lower == "lstm":
    rnn_layer_cls = LSTM
else:
    print(f"[WARN] RNN_TYPE '{RNN_TYPE}' ไม่รู้จัก ใช้ GRU แทน")
    rnn_layer_cls = GRU

model.add(Bidirectional(
    rnn_layer_cls(RNN_UNITS, return_sequences=False)
))

# Classification head
model.add(Dense(DENSE_UNITS1, activation="relu"))
model.add(Dropout(DROPOUT_RATE))
model.add(Dense(DENSE_UNITS2, activation="relu"))
model.add(Dense(actions.shape[0], activation="softmax"))


# ---------------- 5) คอมไพล์โมเดล ----------------

print("Compiling the model...")
optimizer = Adam(learning_rate=LEARNING_RATE)
model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()


# ---------------- 6) Callbacks ----------------

log_dir = os.path.join("logs", EXPERIMENT_NAME)
os.makedirs(log_dir, exist_ok=True)
tb_callback = TensorBoard(log_dir=log_dir)

# << เปลี่ยนตรงนี้ ให้เซฟโมเดลไว้ใน models/ โดยตรง >>
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

checkpoint_path = os.path.join(MODEL_DIR, "best_model.keras")

checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor="val_accuracy",
    verbose=1,
    save_best_only=True,
    mode="max"
)

early_stopping_callback = EarlyStopping(
    monitor="val_loss",                # จะเปลี่ยนเป็น 'val_accuracy' ก็ได้
    patience=EARLY_STOPPING_PATIENCE,
    restore_best_weights=True,
    verbose=1
)

callbacks_list = [tb_callback, checkpoint_callback, early_stopping_callback]


# ---------------- 7) เริ่มเทรนโมเดล ----------------

print("\nStarting training...")

# >>> แก้ไขตรงนี้ครับ: เปลี่ยน augment=True เป็น augment=False <<<
if USE_BALANCED_SAMPLING:
    print("[INFO] Using BALANCED data generator for training (NO ONLINE AUGMENTATION).")
    train_generator = balanced_data_generator(
        X_train, y_train, BATCH_SIZE, augment=True 
    )
else:
    print("[INFO] Using NORMAL data generator for training (NO ONLINE AUGMENTATION).")
    train_generator = data_generator(
        X_train, y_train, BATCH_SIZE, augment=True
    )

# Val set ต้องไม่ augment อยู่แล้ว (ถูกแล้ว)
val_generator = data_generator(
    X_val, y_val, BATCH_SIZE, augment=True
)

steps_per_epoch  = max(1, math.ceil(len(X_train) / BATCH_SIZE))
validation_steps = max(1, math.ceil(len(X_val)   / BATCH_SIZE))

# class_weight argument
class_weight_arg = None
if USE_CLASS_WEIGHT and not USE_BALANCED_SAMPLING:
    class_weight_arg = class_weights
    print("[INFO] Using CLASS WEIGHTS in model.fit()")
elif USE_CLASS_WEIGHT and USE_BALANCED_SAMPLING:
    print("[INFO] Balanced sampling เปิดอยู่แล้ว -> class_weight=None")
else:
    print("[INFO] Not using class_weight (uniform loss).")

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=NUM_EPOCHS,
    validation_data=val_generator,
    validation_steps=validation_steps,
    callbacks=callbacks_list,
    class_weight=class_weight_arg
)

print("--- Training Complete ---")


# ---------------- 8) บันทึกโมเดล ----------------

last_model_path = os.path.join(MODEL_DIR, "last_model.keras")
model.save(last_model_path)
print(f"Final model saved to {last_model_path}")
print(f"Best model saved to {checkpoint_path}")

# ---------------- 9) ประเมินบน TEST set ----------------

print("\nEvaluating on TEST set...")
test_loss, test_acc = model.evaluate(
    X_test, y_test, batch_size=BATCH_SIZE, verbose=1
)
print(f"Test loss = {test_loss:.4f}, Test accuracy = {test_acc:.4f}")


# ---------------- 10) บันทึก label_map ----------------

print("\nSaving label map...")
label_map_file = {i: action for i, action in enumerate(actions)}
label_map_path = os.path.join(MODEL_DIR, "label_map.json")
with open(label_map_path, "w", encoding="utf-8") as f:
    json.dump(label_map_file, f, ensure_ascii=False, indent=4)
print(f"Label map saved to {label_map_path}")


# ---------------- 11) Calibrate thresholds ด้วย VALIDATION set ----------------

print("\nCalibrating per-class thresholds on VALIDATION set...")

probs   = model.predict(X_val, batch_size=BATCH_SIZE, verbose=0)
y_true  = np.argmax(y_val, axis=1)

def best_threshold_for_class(c):
    """
    วนหา Threshold (0.30 - 0.90) ที่ให้ F1-score สูงสุดสำหรับคลาส c
    """
    y_true_bin = (y_true == c).astype(np.int32)
    best_f1, best_th = -1.0, 0.5

    for i in range(30, 91):  # 0.30 ... 0.90
        th = i / 100.0
        y_pred_bin = (probs[:, c] >= th).astype(np.int32)

        tp = int(np.sum((y_pred_bin == 1) & (y_true_bin == 1)))
        fp = int(np.sum((y_pred_bin == 1) & (y_true_bin == 0)))
        fn = int(np.sum((y_pred_bin == 0) & (y_true_bin == 1)))

        if tp + fp == 0 or tp + fn == 0:
            f1 = 0.0
        else:
            precision = tp / (tp + fp)
            recall    = tp / (tp + fn)
            f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

        if f1 > best_f1:
            best_f1, best_th = f1, th

    return best_th, best_f1


thresholds = {}
print("Finding optimal threshold for each class...")
for c, name in enumerate(actions):
    th, f1 = best_threshold_for_class(c)
    thresholds[name] = {"threshold": float(th), "f1_at_threshold": float(f1)}
    print(f"  Class '{name}': Best Threshold = {th:.2f} (F1 = {f1:.4f})")

thresholds_path = os.path.join(MODEL_DIR, "thresholds.json")
with open(thresholds_path, "w", encoding="utf-8") as f:
    json.dump(thresholds, f, ensure_ascii=False, indent=4)
print(f"Thresholds saved to {thresholds_path}")
