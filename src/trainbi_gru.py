import os
import time
import numpy as np
import random
import tensorflow as tf
import json
import math

# os.environ["TF_USE_LEGACY_KERAS"] = "1" # Uncomment ถ้าใช้ TF 2.16+
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
    SpatialDropout1D,
)
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

# ---------------- 0) EXPERIMENT CONFIG ----------------
EXPERIMENT_NAME = "bi-gru-v1.2.3"
RNN_TYPE = "gru"

# --- FIXED HYPERPARAMETERS (กำหนดค่าเองตามต้องการ) ---
CONV_FILTERS = 256
CONV_KERNEL = 3
RNN_UNITS = 64
DENSE_UNITS1 = 128
SPATIAL_DROPOUT_RATE = 0.5
DROPOUT_RATE = 0.5

LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
BATCH_SIZE = 64

# class balancing
USE_CLASS_WEIGHT = False
USE_BALANCED_SAMPLING = True

sequence_length = 30
num_features = 258

# ---------------- 1) CONFIG พื้นฐาน ----------------
DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, "processed_train")
VAL_DIR = os.path.join(DATA_DIR, "processed_val")
TEST_DIR = os.path.join(DATA_DIR, "processed_test")

actions = np.array(
    [
        "anxiety",
        "breathing_difficulty",
        "fever",
        "feverish",
        "insomnia",
        "itching",
        "no_action",
        "polyuria",
        "suffocated",
        "wounded",
    ]
)


SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


# ---------------- 2) Helper Functions (โหลดข้อมูล) ----------------
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


# ---------------- 2.1) Data Generators ----------------
def data_generator(X_data, y_data, batch_size=32):
    num_samples = X_data.shape[0]
    indices = np.arange(num_samples)
    while True:
        np.random.shuffle(indices)
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_indices = indices[start:end]
            yield X_data[batch_indices], y_data[batch_indices]


def balanced_data_generator(X_data, y_data, batch_size=32):
    num_classes = y_data.shape[1]
    y_int = np.argmax(y_data, axis=1)
    class_indices = [np.where(y_int == c)[0] for c in range(num_classes)]
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
        if not X_batch_list:
            continue
        yield np.array(X_batch_list), np.array(y_batch_list)


# ---------------- 3) Attention Layer ----------------
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight", shape=(input_shape[-1], 1), initializer="normal"
        )
        self.b = self.add_weight(
            name="att_bias", shape=(input_shape[1], 1), initializer="zeros"
        )
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)


# ---------------- 4) สร้าง Model (Fixed Config) ----------------
def build_model():
    model = Sequential()

    # 1. Conv1D Block
    model.add(
        Conv1D(
            filters=CONV_FILTERS,
            kernel_size=CONV_KERNEL,
            activation="relu",
            padding="same",
            input_shape=(sequence_length, num_features),
        )
    )
    model.add(BatchNormalization())

    # Optional: ใช้ SpatialDropout1D จะดีกว่า Dropout ธรรมดาสำหรับ Sequence
    model.add(SpatialDropout1D(SPATIAL_DROPOUT_RATE))
    model.add(MaxPooling1D(pool_size=2))

    # 2. RNN Block
    rnn_layer_cls = GRU if RNN_TYPE.lower() == "gru" else LSTM
    model.add(Bidirectional(rnn_layer_cls(RNN_UNITS, return_sequences=True)))

    # 3. Attention
    model.add(Attention())

    # 4. Dense Block
    model.add(Dense(DENSE_UNITS1, activation="relu"))
    model.add(Dropout(DROPOUT_RATE))

    # Dense ชั้นที่ 2
    # model.add(Dense(DENSE_UNITS2, activation="relu"))

    # Output Layer
    model.add(Dense(actions.shape[0], activation="softmax"))

    # Compile
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.0)

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=loss_fn,
        metrics=["accuracy"],
    )
    return model


# ---------------- 5) Training Process ----------------
print(f"\n--- Starting Training (Experiment: {EXPERIMENT_NAME}) ---")
print(
    f"Config: Filters={CONV_FILTERS}, RNN={RNN_UNITS}, Dense={DENSE_UNITS1}, Drop={DROPOUT_RATE}"
)

# สร้างโมเดล
model = build_model()
model.summary()

# Setup Callbacks
log_dir = os.path.join("logs", EXPERIMENT_NAME)
MODEL_DIR = "models"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

checkpoint_path = os.path.join(MODEL_DIR, "best_model.keras")

tb_callback = TensorBoard(log_dir=log_dir)
checkpoint_callback = ModelCheckpoint(
    checkpoint_path, monitor="val_accuracy", verbose=1, save_best_only=True, mode="max"
)
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1
)
early_stop = EarlyStopping(
    monitor="val_accuracy", patience=15, restore_best_weights=True
)

callbacks_list = [tb_callback, checkpoint_callback, reduce_lr, early_stop]

# เตรียม Generators
if USE_BALANCED_SAMPLING:
    print("[INFO] Using BALANCED data generator.")
    train_gen = balanced_data_generator(X_train, y_train, BATCH_SIZE)
else:
    print("[INFO] Using NORMAL data generator.")
    train_gen = data_generator(X_train, y_train, BATCH_SIZE)
val_gen = data_generator(X_val, y_val, BATCH_SIZE)

# เริ่มเทรน
history = model.fit(
    train_gen,
    steps_per_epoch=max(1, math.ceil(len(X_train) / BATCH_SIZE)),
    epochs=NUM_EPOCHS,
    validation_data=val_gen,
    validation_steps=max(1, math.ceil(len(X_val) / BATCH_SIZE)),
    callbacks=callbacks_list,
)


# ---------------- 6) Evaluation & Save ----------------
print("\nEvaluating Model on TEST set...")
test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=1)
print(f"Test loss = {test_loss:.4f}, Test accuracy = {test_acc:.4f}")

# Save Final Model
model.save(os.path.join(MODEL_DIR, "final_model.keras"))
print(f"Final model saved to {os.path.join(MODEL_DIR, 'final_model.keras')}")

# Save Label Map
label_map = {i: action for i, action in enumerate(actions)}
with open(os.path.join(MODEL_DIR, "label_map.json"), "w", encoding="utf-8") as f:
    json.dump(label_map, f, ensure_ascii=False, indent=4)

# Threshold Calibration
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
