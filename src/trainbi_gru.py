import os
import time
import numpy as np
import random
import tensorflow as tf
import json
import math
import keras_tuner as kt  # <--- ต้องลง library นี้เพิ่ม: pip install keras-tuner

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
EXPERIMENT_NAME = "exp_gru_tuner"  # เปลี่ยนชื่อเป็น Tuner
RNN_TYPE = "gru"

# Config พื้นฐานสำหรับการ Tuning (เราจะให้ Tuner เลือกค่าในช่วงเหล่านี้)
# MIN_VALUE, MAX_VALUE, STEP
# ไม่ต้องกำหนดค่าตายตัวตรงนี้แล้ว

NUM_EPOCHS_SEARCH = 20   # จำนวน Epoch สูงสุดตอน Search (เอาพอประมาณให้รู้แนวโน้ม)
NUM_EPOCHS_FINAL = 50    # จำนวน Epoch ตอนเทรนจริงด้วยค่าที่ดีที่สุด
BATCH_SIZE = 32

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
    'fever',
    'feverish',
    'no_action',
    ]
)

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
            if len(class_indices[c]) == 0: continue
            idx = int(np.random.choice(class_indices[c]))
            X_batch_list.append(X_data[idx])
            y_batch_list.append(y_data[idx])
        if not X_batch_list: continue
        yield np.array(X_batch_list), np.array(y_batch_list)


# ---------------- 3) Attention Layer ----------------
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)


# ---------------- 4) สร้าง Model ด้วย Keras Tuner (Step 2) ----------------
def build_model(hp):
    """
    ฟังก์ชันสำหรับสร้างโมเดล โดยรับ hp (HyperParameters) เข้ามา
    เพื่อให้ Tuner เลือกค่าต่างๆ ให้
    """
    model = Sequential()

    # 1. Tuning Conv1D
    # ให้เลือก Filters ระหว่าง 32 ถึง 128 (เพิ่มทีละ 32)
    hp_filters = hp.Int('conv_filters', min_value=32, max_value=128, step=32)
    # ให้เลือก Kernel Size ว่าจะเป็น 3, 5 หรือ 7
    hp_kernel = hp.Choice('conv_kernel', values=[3, 5, 7])

    model.add(
        Conv1D(
            filters=hp_filters,
            kernel_size=hp_kernel,
            activation="relu",
            padding="same",
            input_shape=(sequence_length, num_features),
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))

    # 2. Tuning RNN
    # ให้เลือก RNN Units
    hp_rnn_units = hp.Int('rnn_units', min_value=32, max_value=128, step=32)
    rnn_layer_cls = GRU if RNN_TYPE.lower() == "gru" else LSTM
    
    model.add(Bidirectional(rnn_layer_cls(hp_rnn_units, return_sequences=True)))

    # 3. Attention (Fixed)
    model.add(Attention())

    # 4. Tuning Dense & Dropout
    hp_dense1 = hp.Int('dense_units1', min_value=32, max_value=128, step=32)
    hp_dropout = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)

    model.add(Dense(hp_dense1, activation="relu"))
    model.add(Dropout(hp_dropout))
    
    # Dense ชั้นที่ 2 (Optional: จะ fix หรือ tune ก็ได้)
    model.add(Dense(hp_dense1 // 2, activation="relu")) # ให้เป็นครึ่งหนึ่งของชั้นแรก
    
    model.add(Dense(actions.shape[0], activation="softmax"))

    # 5. Tuning Learning Rate
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    # Compile
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.0) # ตามคำแนะนำ Step 1
    model.compile(
        optimizer=Adam(learning_rate=hp_learning_rate),
        loss=loss_fn,
        metrics=["accuracy"]
    )
    return model


# ---------------- 5) Setup Tuner ----------------
print("\n--- Starting Hyperparameter Tuning (Keras Tuner) ---")

tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=NUM_EPOCHS_SEARCH,
    factor=3,
    directory='tuner_dir',
    project_name=EXPERIMENT_NAME
)

# Callbacks สำหรับตอน Search
stop_early = EarlyStopping(monitor='val_loss', patience=5)

# เตรียม Generators
if USE_BALANCED_SAMPLING:
    train_gen = balanced_data_generator(X_train, y_train, BATCH_SIZE)
else:
    train_gen = data_generator(X_train, y_train, BATCH_SIZE)
val_gen = data_generator(X_val, y_val, BATCH_SIZE)

# เริ่ม Search
# หมายเหตุ: เราใช้ steps_per_epoch เพื่อบอกว่า 1 epoch คือกี่ batch
tuner.search(
    train_gen,
    steps_per_epoch=max(1, math.ceil(len(X_train) / BATCH_SIZE)),
    validation_data=val_gen,
    validation_steps=max(1, math.ceil(len(X_val) / BATCH_SIZE)),
    epochs=NUM_EPOCHS_SEARCH,
    callbacks=[stop_early],
    # class_weight=class_weights # Tuner บางเวอร์ชันอาจมีปัญหากับ class_weight ใน argument นี้ แต่ลองใส่ได้
)


# ---------------- 6) Get Best Model & Hyperparameters ----------------
print("\n--- Tuning Complete ---")
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. 
The optimal number of units in the first densely-connected layer is {best_hps.get('dense_units1')} 
and the optimal learning rate for the optimizer is {best_hps.get('learning_rate')}.
Conv Filters: {best_hps.get('conv_filters')}
Conv Kernel: {best_hps.get('conv_kernel')}
RNN Units: {best_hps.get('rnn_units')}
Dropout: {best_hps.get('dropout_rate')}
""")

# สร้างโมเดลจากค่าที่ดีที่สุด
best_model = tuner.hypermodel.build(best_hps)

# ---------------- 7) Retrain Best Model (Final Training) ----------------
print("\n--- Retraining the best model ---")

# Setup Callbacks เหมือนเดิม
log_dir = os.path.join("logs", EXPERIMENT_NAME + "_best")
MODEL_DIR = "models"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

checkpoint_path = os.path.join(MODEL_DIR, "best_tuned_model.keras")
tb_callback = TensorBoard(log_dir=log_dir)
checkpoint_callback = ModelCheckpoint(checkpoint_path, monitor="val_accuracy", verbose=1, save_best_only=True, mode="max")
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1)
early_stop = EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True)

callbacks_list = [tb_callback, checkpoint_callback, reduce_lr, early_stop]

history = best_model.fit(
    train_gen,
    steps_per_epoch=max(1, math.ceil(len(X_train) / BATCH_SIZE)),
    epochs=NUM_EPOCHS_FINAL,
    validation_data=val_gen,
    validation_steps=max(1, math.ceil(len(X_val) / BATCH_SIZE)),
    callbacks=callbacks_list,
    # class_weight=class_weights # ถ้าใช้ Balanced Generator ไม่ต้องใส่ class_weight ก็ได้
)


# ---------------- 8) Evaluation & Save ----------------
print("\nEvaluating Best Model on TEST set...")
test_loss, test_acc = best_model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=1)
print(f"Test loss = {test_loss:.4f}, Test accuracy = {test_acc:.4f}")

# Save Model
best_model.save(os.path.join(MODEL_DIR, "final_tuned_model.keras"))
print(f"Final model saved to {os.path.join(MODEL_DIR, 'final_tuned_model.keras')}")

# Save Label Map
label_map = {i: action for i, action in enumerate(actions)}
with open(os.path.join(MODEL_DIR, "label_map.json"), "w", encoding="utf-8") as f:
    json.dump(label_map, f, ensure_ascii=False, indent=4)

# Threshold Calibration (เหมือนเดิม)
print("\nCalibrating thresholds...")
probs = best_model.predict(X_val, batch_size=BATCH_SIZE, verbose=0)
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