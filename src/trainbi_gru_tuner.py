import os
import time
import numpy as np
import random
import tensorflow as tf
import json
import math
import keras_tuner as kt  # <--- 🔥 เพิ่มบรรทัดนี้

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
    SpatialDropout1D
)
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

# ---------------- 0) EXPERIMENT CONFIG ----------------
EXPERIMENT_NAME = "bi-gru-tuned-v1" # เปลี่ยนชื่อนิดหน่อยให้รู้ว่าผ่านการจูนมาแล้ว
RNN_TYPE = "gru"

# --- FIXED CONFIG (ค่าคงที่สำหรับการเทรน) ---
NUM_EPOCHS_SEARCH = 30  # จำนวน Epoch สูงสุดตอน "ค้นหา" (Hyperband)
NUM_EPOCHS_FINAL  = 50  # จำนวน Epoch ตอน "เทรนจริง" (หลังจากได้ค่าที่ดีที่สุดแล้ว)
BATCH_SIZE        = 32

# class balancing
USE_CLASS_WEIGHT = True
USE_BALANCED_SAMPLING = True

# ---------------- 1) CONFIG พื้นฐาน ----------------
DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, "processed_train")
VAL_DIR = os.path.join(DATA_DIR, "processed_val")
TEST_DIR = os.path.join(DATA_DIR, "processed_test")

# ⚠️ อย่าลืมแก้ชื่อท่าตรงนี้ให้ครบนะครับ
actions = np.array(["fever", "feverish", "no_action"])

sequence_length = 30
num_features = 258

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
    
    # เพิ่ม get_config เพื่อให้ Keras Tuner/Keras 3 บันทึกโมเดลได้สมบูรณ์
    def get_config(self):
        config = super(Attention, self).get_config()
        return config


# ---------------- 4) 🔥 ปรับแก้: build_model รองรับ HP ----------------
def build_model(hp):
    model = Sequential()

    # --- Tuning 1: Conv1D ---
    # สุ่ม filters 32-128, kernel 3-7
    hp_filters = hp.Int('conv_filters', min_value=32, max_value=128, step=32)
    hp_kernel = hp.Int('conv_kernel', min_value=3, max_value=7, step=2)

    model.add(Conv1D(
        filters=hp_filters,
        kernel_size=hp_kernel,
        activation="relu",
        padding="same",
        input_shape=(sequence_length, num_features),
    ))
    model.add(BatchNormalization())
    
    # --- Tuning 2: Spatial Dropout ---
    hp_spatial_drop = hp.Float('spatial_dropout', min_value=0.1, max_value=0.5, step=0.1)
    model.add(SpatialDropout1D(hp_spatial_drop)) 
    model.add(MaxPooling1D(pool_size=2))

    # --- Tuning 3: RNN Units ---
    hp_rnn_units = hp.Int('rnn_units', min_value=32, max_value=128, step=32)
    rnn_layer_cls = GRU if RNN_TYPE.lower() == "gru" else LSTM
    model.add(Bidirectional(rnn_layer_cls(hp_rnn_units, return_sequences=True)))
    
    # Attention Layer
    model.add(Attention())

    # --- Tuning 4: Dense Layers ---
    hp_dense1 = hp.Int('dense_units_1', min_value=32, max_value=128, step=32)
    model.add(Dense(hp_dense1, activation="relu"))
    
    hp_dropout = hp.Float('dropout_rate', min_value=0.2, max_value=0.6, step=0.1)
    model.add(Dropout(hp_dropout))

    hp_dense2 = hp.Int('dense_units_2', min_value=16, max_value=64, step=16)
    model.add(Dense(hp_dense2, activation="relu"))

    # Output Layer
    model.add(Dense(actions.shape[0], activation="softmax"))

    # --- Tuning 5: Learning Rate ---
    hp_lr = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.0)
    
    model.compile(
        optimizer=Adam(learning_rate=hp_lr),
        loss=loss_fn,
        metrics=["accuracy"],
    )
    return model


# ---------------- 5) 🔥 เริ่มกระบวนการ Tuning ----------------
print(f"\n--- Starting Hyperparameter Search (Experiment: {EXPERIMENT_NAME}) ---")

# เตรียม Generators
if USE_BALANCED_SAMPLING:
    print("[INFO] Using BALANCED data generator.")
    train_gen = balanced_data_generator(X_train, y_train, BATCH_SIZE)
else:
    print("[INFO] Using NORMAL data generator.")
    train_gen = data_generator(X_train, y_train, BATCH_SIZE)
val_gen = data_generator(X_val, y_val, BATCH_SIZE)

# คำนวณ Steps
train_steps = max(1, math.ceil(len(X_train) / BATCH_SIZE))
val_steps = max(1, math.ceil(len(X_val) / BATCH_SIZE))

# สร้าง Tuner (Hyperband)
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=NUM_EPOCHS_SEARCH, # รอบสูงสุดตอนค้นหา
    factor=3,
    directory='tuner_results',    # โฟลเดอร์เก็บผลจูน
    project_name=EXPERIMENT_NAME
)

# Callback สำหรับหยุดตอนจูนถ้าค่าแย่ลง
stop_early = EarlyStopping(monitor='val_loss', patience=5)

# สั่งเริ่มค้นหา
tuner.search(
    train_gen,
    steps_per_epoch=train_steps,
    epochs=NUM_EPOCHS_SEARCH,
    validation_data=val_gen,
    validation_steps=val_steps,
    callbacks=[stop_early]
)

# ดึงค่า Hyperparameters ที่ดีที่สุดออกมา
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print("\n" + "="*50)
print("✅ TUNING COMPLETE! Best Hyperparameters:")
print(f" - Conv Filters:    {best_hps.get('conv_filters')}")
print(f" - Conv Kernel:     {best_hps.get('conv_kernel')}")
print(f" - Spatial Dropout: {best_hps.get('spatial_dropout')}")
print(f" - RNN Units:       {best_hps.get('rnn_units')}")
print(f" - Dense 1 Units:   {best_hps.get('dense_units_1')}")
print(f" - Dense 2 Units:   {best_hps.get('dense_units_2')}")
print(f" - Dropout Rate:    {best_hps.get('dropout_rate')}")
print(f" - Learning Rate:   {best_hps.get('learning_rate')}")
print("="*50 + "\n")


# ---------------- 6) 🔥 Retrain โมเดลด้วยค่าที่ดีที่สุด ----------------
print(f"--- Retraining Final Model with Best Hyperparameters ({NUM_EPOCHS_FINAL} Epochs) ---")

# สร้างโมเดลจากค่า HP ที่ดีที่สุด
model = tuner.hypermodel.build(best_hps)
model.summary()

# Setup Callbacks สำหรับเทรนจริง
log_dir = os.path.join("logs", EXPERIMENT_NAME + "_final")
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
early_stop_final = EarlyStopping(
    monitor="val_accuracy", patience=15, restore_best_weights=True
)

callbacks_list = [tb_callback, checkpoint_callback, reduce_lr, early_stop_final]

# เริ่มเทรนจริง
history = model.fit(
    train_gen,
    steps_per_epoch=train_steps,
    epochs=NUM_EPOCHS_FINAL,
    validation_data=val_gen,
    validation_steps=val_steps,
    callbacks=callbacks_list,
)


# ---------------- 7) Evaluation & Save (เหมือนเดิม) ----------------
print("\nEvaluating Final Model on TEST set...")
test_loss, test_acc = model.evaluate(
    X_test, y_test, batch_size=BATCH_SIZE, verbose=1
)
print(f"Test loss = {test_loss:.4f}, Test accuracy = {test_acc:.4f}")

# Save Final Model (ตัวล่าสุด)
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