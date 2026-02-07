# eval_confusion.py
import os, json, argparse, math, random
# os.environ["TF_USE_LEGACY_KERAS"] = "1"
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K

# ----------------- 1. ใส่ Class Attention ให้เหมือนไฟล์เทรนเป๊ะๆ -----------------
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', 
                                 shape=(input_shape[-1], 1), 
                                 initializer='normal', trainable=True)
        self.b = self.add_weight(name='attention_bias', 
                                 shape=(input_shape[1], 1), 
                                 initializer='zeros', trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)

    def get_config(self):
        config = super(Attention, self).get_config()
        return config

# ----------------- CLI -----------------
parser = argparse.ArgumentParser(description="Evaluate best_model.keras and export confusion matrix & metrics.")
parser.add_argument("--model-dir", default="models", help="Directory containing best_model.keras / label_map.json")

# แก้ไข Default ให้ชี้ไปที่ processed_test โดยตรง
parser.add_argument("--data-dir",  default=os.path.join("data","processed_test"), help="Processed .npy root dir")

# แก้ไข Default ให้ใช้ข้อมูลทั้งหมด (all) เพราะใน folder test คือ test ล้วนๆ แล้ว
parser.add_argument("--subset",    choices=["test","all"], default="all", help="Evaluate on test split or all data")

parser.add_argument("--test-size", type=float, default=0.2, help="Test size for split when subset=test")
parser.add_argument("--seed",      type=int, default=42, help="Random seed")
parser.add_argument("--batch",     type=int, default=32, help="Prediction batch size")
parser.add_argument("--out",       default="reports", help="Output directory for reports")
args = parser.parse_args()

np.random.seed(args.seed)
tf.random.set_seed(args.seed)
random.seed(args.seed)

MODEL_PATH = os.path.join(args.model_dir, "best_model.keras")
LABEL_MAP_PATH = os.path.join(args.model_dir, "label_map.json")

SEQ_LEN, NUM_FEAT = 30, 258
os.makedirs(args.out, exist_ok=True)

# ----------------- Load labels -----------------
def load_labels():
    if os.path.exists(LABEL_MAP_PATH):
        with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
            lm = json.load(f)  # {"0": "classA", "1": "classB", ...}
        # sort by numeric key to preserve index order
        idxs = sorted(int(k) for k in lm.keys())
        labels = [lm[str(i)] for i in idxs]
        return labels
    # fallback: read subfolders alphabetically
    classes = [d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))]
    classes.sort()
    return classes

labels = load_labels()
num_classes = len(labels)
print(f"[INFO] Classes ({num_classes}):", labels)

# ----------------- Load dataset -----------------
X, y = [], []
print(f"[INFO] Loading data from: {args.data_dir}")

for ci, cname in enumerate(labels):
    cdir = os.path.join(args.data_dir, cname)
    if not os.path.isdir(cdir):
        print(f"[WARN] Missing class folder: {cdir} (skip this class)")
        continue
    files = [f for f in os.listdir(cdir) if f.endswith(".npy")]
    files.sort()
    count = 0
    for f in files:
        arr = np.load(os.path.join(cdir, f))
        if arr.shape != (SEQ_LEN, NUM_FEAT):
            print(f"[SKIP] {cname}/{f} shape={arr.shape}")
            continue
        X.append(arr.astype(np.float32))
        y.append(ci)
        count += 1
    print(f"  - {cname}: {count} samples")

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y, dtype=np.int32)
print(f"[INFO] Total dataset: X={X.shape}, y={y.shape}")

if len(X) == 0:
    print("[ERROR] No data found! Please check data path.")
    exit()

# ----------------- Split -----------------
if args.subset == "test":
    X_train, X_eval, y_train, y_eval = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    print(f"[INFO] Using SPLIT (subset=test): {X_eval.shape[0]} samples for eval")
else:
    X_eval, y_eval = X, y
    print(f"[INFO] Using ALL loaded data (subset=all): {X_eval.shape[0]} samples")

# ----------------- Load model -----------------
print("[INFO] Loading model:", MODEL_PATH)

# 2. ส่ง custom_objects เข้าไปเพื่อให้รู้จัก Attention
model = tf.keras.models.load_model(
    MODEL_PATH, 
    custom_objects={'Attention': Attention}
)

# ----------------- Predict -----------------
print("Predicting...")
probs = model.predict(X_eval, batch_size=args.batch, verbose=1)  # [N, C]
y_pred = np.argmax(probs, axis=1)

# ----------------- Metrics -----------------
cm = confusion_matrix(y_eval, y_pred, labels=list(range(num_classes)))  # counts
cm_norm = cm.astype(np.float32) / np.maximum(cm.sum(axis=1, keepdims=True), 1)  # row-normalized

report = classification_report(y_eval, y_pred, target_names=labels, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report).transpose()

# Per-class quick table
p, r, f1, support = precision_recall_fscore_support(
    y_eval, y_pred, labels=list(range(num_classes)), zero_division=0
)
per_class_df = pd.DataFrame({
    "class": labels,
    "precision": p,
    "recall": r,
    "f1": f1,
    "support": support
}).sort_values("class")

# ----------------- Save CSVs -----------------
pd.DataFrame(cm, index=labels, columns=labels).to_csv(os.path.join(args.out, "confusion_counts.csv"), encoding="utf-8-sig")
pd.DataFrame(cm_norm, index=labels, columns=labels).round(4).to_csv(os.path.join(args.out, "confusion_norm.csv"), encoding="utf-8-sig")
report_df.to_csv(os.path.join(args.out, "classification_report.csv"), encoding="utf-8-sig", float_format="%.4f")
per_class_df.to_csv(os.path.join(args.out, "per_class_table.csv"), index=False, encoding="utf-8-sig", float_format="%.4f")

# ----------------- Plot Confusion Matrices -----------------
def plot_cm(M, title, fn, normalize=False):
    plt.figure(figsize=(max(6, 0.6*num_classes), max(5, 0.6*num_classes)))
    plt.imshow(M, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, labels, rotation=90)
    plt.yticks(tick_marks, labels)
    fmt = ".2f" if normalize else "d"
    thresh = (M.max()/2.0) if M.size > 0 else 0.5
    for i in range(num_classes):
        for j in range(num_classes):
            val = M[i, j]
            s = f"{val:{fmt}}"
            plt.text(j, i, s,
                     horizontalalignment="center",
                     color="white" if val > thresh else "black",
                     fontsize=8)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, fn), dpi=180)
    plt.close()

plot_cm(cm, "Confusion Matrix (Counts)", "confusion_counts.png", normalize=False)
plot_cm(cm_norm, "Confusion Matrix (Row-normalized)", "confusion_norm.png", normalize=True)

# ----------------- Summary -----------------
overall_acc = (y_pred == y_eval).mean()
summary = [
    f"Samples eval: {X_eval.shape[0]}",
    f"Overall accuracy: {overall_acc:.4f}",
    "",
    "Per-class (worst 3 by F1):"
]
worst3 = per_class_df.sort_values("f1").head(3)
best3  = per_class_df.sort_values("f1", ascending=False).head(3)
for _, row in worst3.iterrows():
    summary.append(f"  - {row['class']}: F1={row['f1']:.3f}, P={row['precision']:.3f}, R={row['recall']:.3f}, n={int(row['support'])}")
summary.append("")
summary.append("Per-class (best 3 by F1):")
for _, row in best3.iterrows():
    summary.append(f"  + {row['class']}: F1={row['f1']:.3f}, P={row['precision']:.3f}, R={row['recall']:.3f}, n={int(row['support'])}")

with open(os.path.join(args.out, "summary.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(summary))

print("\n".join(summary))
print(f"\n[OK] Saved reports to: {os.path.abspath(args.out)}")