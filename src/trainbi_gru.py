import os
import time
import numpy as np
import random
import tensorflow as tf
import json
import math

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    TensorBoard,
    EarlyStopping,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    Conv1D,
    Bidirectional,
    GRU,
    BatchNormalization,
    MaxPooling1D,
    Input,
    Layer,
)
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

# ---------------- 0) EXPERIMENT CONFIG ----------------
EXPERIMENT_NAME = "exp_gru_attention_v4"  # เปลี่ยนชื่อนิดนึงจะได้รู้ว่าเป็นเวอร์ชันใหม่
RNN_TYPE = "gru"
CONV_FILTERS = 64
RNN_UNITS = 64
DENSE_UNITS1 = 64
DENSE_UNITS2 = 32
DROPOUT_RATE = 0.6

LEARNING_RATE = 1e-3
NUM_EPOCHS = 100  # เพิ่มรอบหน่อย เพราะมี LR Scheduler ช่วย
BATCH_SIZE = 32

EARLY_STOPPING_PATIENCE = 15  # รอได้นานขึ้นอีกนิด

# class balancing
USE_CLASS_WEIGHT = True
USE_BALANCED_SAMPLING = True


# ---------------- 1) CONFIG พื้นฐาน ----------------
DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, "processed_train")
VAL_DIR = os.path.join(DATA_DIR, "processed_val")
TEST_DIR = os.path.join(DATA_DIR, "processed_test")

actions = np.array(
    [
        # 'distention',
        "fever",
        "feverish",
        "no_action",
        # 'wounded'
    ]
)

sequence_length = 30
num_features = 258

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


# ---------------- 2) Helper Functions ----------------
# (ส่วนโหลดข้อมูลคงเดิม)
def load_split(split_dir):
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
    X = np.array(sequences)
    y = np.array(labels)
    y_one_hot = to_categorical(y, num_classes=len(actions))
    print(f"  -> Loaded {X.shape[0]} sequences from {split_dir}")
    return X, y_one_hot


print("Loading datasets (train / val / test)...")
X_train, y_train = load_split(TRAIN_DIR)
X_val, y_val = load_split(VAL_DIR)
X_test, y_test = load_split(TEST_DIR)

# คำนวณ Class Weights
y_train_int = np.argmax(y_train, axis=1)
class_counts = np.bincount(y_train_int, minlength=len(actions))
class_weights = {}
total = float(np.sum(class_counts))
for i, cnt in enumerate(class_counts):
    if cnt == 0:
        class_weights[i] = 0.0
    else:
        class_weights[i] = total / (len(actions) * float(cnt))
# ---------------- 2.1) Data Generators (ตัวตักข้อมูล) ----------------


def data_generator(X_data, y_data, batch_size=32, augment=False):
    """
    Generator แบบปกติ: สุ่มลำดับข้อมูลแล้วส่งไปทีละ Batch
    (ตัดส่วน Augment ออกแล้ว เพื่อความคลีน)
    """
    num_samples = X_data.shape[0]
    indices = np.arange(num_samples)

    while True:
        np.random.shuffle(indices)
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_indices = indices[start:end]

            X_batch = X_data[batch_indices]
            y_batch = y_data[batch_indices]

            # ส่งข้อมูลดิบๆ เลย (เพราะเราทำ Offline Augment มาแล้ว)
            yield X_batch, y_batch


def balanced_data_generator(X_data, y_data, batch_size=32, augment=False):
    """
    Generator แบบ Balanced: สุ่มหยิบทีละคลาสเท่าๆ กัน
    เหมาะมากสำหรับข้อมูลที่ไม่สมดุล (Imbalanced Data)
    """
    num_classes = y_data.shape[1]
    y_int = np.argmax(y_data, axis=1)

    # แยก Index ของแต่ละคลาสเตรียมไว้
    class_indices = [np.where(y_int == c)[0] for c in range(num_classes)]
    classes = np.arange(num_classes)

    while True:
        X_batch_list, y_batch_list = [], []

        # วนลูปหยิบของใส่ตะกร้าจนเต็ม Batch
        for _ in range(batch_size):
            # 1. สุ่มเลือกคลาสมา 1 คลาส
            c = int(np.random.choice(classes))
            if len(class_indices[c]) == 0:
                continue

            # 2. สุ่มหยิบตัวอย่างในคลาสนั้นมา 1 อัน
            idx = int(np.random.choice(class_indices[c]))

            X_batch_list.append(X_data[idx])
            y_batch_list.append(y_data[idx])

        if not X_batch_list:
            continue

        yield np.array(X_batch_list), np.array(y_batch_list)

# ---------------- 3) สร้าง Custom Attention Layer ----------------
# ★★★ จุดที่ 1: เพิ่ม Attention Layer ตรงนี้ ★★★
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], 1),
            initializer="normal",
            trainable=True,
        )
        self.b = self.add_weight(
            name="attention_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True,
        )
        super(Attention, self).build(input_shape)

    def call(self, x):
        # x shape: (batch, seq_len, features)
        # score = tanh(xW + b)
        e = K.tanh(K.dot(x, self.W) + self.b)
        # weights = softmax(score)
        a = K.softmax(e, axis=1)
        # context = sum(x * weights)
        output = x * a
        return K.sum(output, axis=1)


# ---------------- 4) สร้างโมเดล (Sequential with Attention) ----------------
print(f"\nBuilding Model: {RNN_TYPE} + Attention + Conv1D")

model = Sequential()

# 1. Conv1D
model.add(
    Conv1D(
        filters=CONV_FILTERS,
        kernel_size=3,
        activation="relu",
        padding="same",
        input_shape=(sequence_length, num_features),
    )
)
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))

# 2. RNN (Bi-GRU/LSTM)
# ★★★ จุดที่ 2: ต้องเปิด return_sequences=True เพื่อส่งต่อให้ Attention ★★★
rnn_layer_cls = GRU if RNN_TYPE.lower() == "gru" else LSTM
model.add(Bidirectional(rnn_layer_cls(RNN_UNITS, return_sequences=True)))

# 3. Attention
# ★★★ จุดที่ 3: แทรก Attention เข้าไป ★★★
model.add(Attention())

# 4. Classification Head
model.add(Dense(DENSE_UNITS1, activation="relu"))
model.add(Dropout(DROPOUT_RATE))
model.add(Dense(DENSE_UNITS2, activation="relu"))
model.add(Dense(actions.shape[0], activation="softmax"))


# ---------------- 5) คอมไพล์โมเดล ----------------
print("Compiling the model with Label Smoothing...")

# ★★★ จุดที่ 4: ใช้ Label Smoothing แทน Focal Loss ★★★
loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE), loss=loss_fn, metrics=["accuracy"]
)
model.summary()


# ---------------- 6) Callbacks & Training ----------------
log_dir = os.path.join("logs", EXPERIMENT_NAME)
MODEL_DIR = "models"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

checkpoint_path = os.path.join(MODEL_DIR, "best_model.keras")

# Callbacks
tb_callback = TensorBoard(log_dir=log_dir)
checkpoint_callback = ModelCheckpoint(
    checkpoint_path, monitor="val_accuracy", verbose=1, save_best_only=True, mode="max"
)
early_stopping_callback = EarlyStopping(
    monitor="val_accuracy",
    patience=EARLY_STOPPING_PATIENCE,
    restore_best_weights=True,
    verbose=1,
)

# ★★★ จุดที่ 5: เพิ่ม ReduceLROnPlateau ★★★
reduce_lr_callback = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,  # ลด LR เหลือ 50%
    patience=4,  # ถ้ารอ 4 epoch แล้ว loss ไม่ลด
    min_lr=1e-6,
    verbose=1,
)

callbacks_list = [
    tb_callback,
    checkpoint_callback,
    early_stopping_callback,
    reduce_lr_callback,
]

print("\nStarting training...")

if USE_BALANCED_SAMPLING:
    print("[INFO] Using BALANCED data generator for training.")
    train_generator = balanced_data_generator(
        X_train, y_train, BATCH_SIZE, augment=False
    )
else:
    print("[INFO] Using NORMAL data generator for training.")
    train_generator = data_generator(X_train, y_train, BATCH_SIZE, augment=False)

# ★★★ จุดที่ 6: Val Generator ต้องปิด Augmentation (augment=False) ★★★
val_generator = data_generator(X_val, y_val, BATCH_SIZE, augment=False)

class_weight_arg = (
    class_weights if (USE_CLASS_WEIGHT and not USE_BALANCED_SAMPLING) else None
)

history = model.fit(
    train_generator,
    steps_per_epoch=max(1, math.ceil(len(X_train) / BATCH_SIZE)),
    epochs=NUM_EPOCHS,
    validation_data=val_generator,
    validation_steps=max(1, math.ceil(len(X_val) / BATCH_SIZE)),
    callbacks=callbacks_list,
    class_weight=class_weight_arg,
)

# ---------------- 7) บันทึกและประเมินผล ----------------
model.save(os.path.join(MODEL_DIR, "last_model.keras"))
print(f"Models saved to {MODEL_DIR}")

print("\nEvaluating on TEST set...")
test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=1)
print(f"Test loss = {test_loss:.4f}, Test accuracy = {test_acc:.4f}")

# Save Label Map
label_map = {i: action for i, action in enumerate(actions)}
with open(os.path.join(MODEL_DIR, "label_map.json"), "w", encoding="utf-8") as f:
    json.dump(label_map, f, ensure_ascii=False, indent=4)

# Calibrate Thresholds
print("\nCalibrating thresholds...")
probs = model.predict(X_val, batch_size=BATCH_SIZE, verbose=0)
y_true = np.argmax(y_val, axis=1)
thresholds = {}
for c, name in enumerate(actions):
    y_true_c = (y_true == c).astype(int)
    best_f1, best_th = -1, 0.5
    for th in np.linspace(0.3, 0.9, 61):
        y_pred_c = (probs[:, c] >= th).astype(int)
        tp = np.sum((y_pred_c == 1) & (y_true_c == 1))
        fp = np.sum((y_pred_c == 1) & (y_true_c == 0))
        fn = np.sum((y_pred_c == 0) & (y_true_c == 1))
        f1 = 0.0 if (tp + fp == 0 or tp + fn == 0) else (2 * tp) / (2 * tp + fp + fn)
        if f1 > best_f1:
            best_f1, best_th = f1, th
    thresholds[name] = {"threshold": float(best_th), "f1": float(best_f1)}
    print(f"  {name}: Th={best_th:.2f}, F1={best_f1:.4f}")

with open(os.path.join(MODEL_DIR, "thresholds.json"), "w", encoding="utf-8") as f:
    json.dump(thresholds, f, ensure_ascii=False, indent=4)
print("Done.")
