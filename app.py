# run_live_lstm_tflite.py
# ทดลองรันโมเดล LSTM (.tflite – FP16) แบบเปิดกล้อง + UI แสดงชื่อท่าและเปอร์เซ็นต์ (Top-3 bars)
# ต้องมีไฟล์:
#   models/model_fp16.tflite
#   models/label_map.json  (รูปแบบ {"0":"fever","1":"feverish",...})
#   models/thresholds.json (ออปชัน)

import os, json, collections
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf  # ใช้ tf.lite.Interpreter

# ========== CONFIG ==========
MODEL_DIR        = "models"
TFLITE_MODEL     = os.path.join(MODEL_DIR, "model_fp16.tflite")
LABEL_MAP_PATH   = os.path.join(MODEL_DIR, "label_map.json")
THRESH_PATH      = os.path.join(MODEL_DIR, "thresholds.json")  # optional

SEQ_LEN   = 30
FEAT_DIM  = 258

# UI / เสถียรภาพ
PROCESS_EVERY_N = 1      # ประมวลผลทุก N เฟรม (1=ทุกเฟรม)
ALPHA_EMA       = 0.20   # EMA smoothing (ต่ำ=นิ่ง)
DEFAULT_THRESH  = 0.70   # เกณฑ์ความมั่นใจขั้นต่ำต่อคลาส
TOP2_MARGIN     = 0.20   # อันดับ 1 ต้องทิ้งห่างอันดับ 2
MIN_COVERAGE    = 0.50   # อย่างน้อยกี่เฟรมจาก 30 ที่มีแลนด์มาร์ก (>0 คือมีข้อมูลจริง)
STABLE_FRAMES   = 10      # ผ่านเงื่อนไขซ้ำ ๆ กี่ครั้งจึงจะยืนยันผล (กันกระพริบ)

CAM_INDEX        = 0
FRAME_W, FRAME_H = 640, 480
MODEL_COMPLEXITY = 0     # Mediapipe Holistic: 0 เร็วสุด / 1 สมดุล / 2 แม่นกว่าแต่ช้ากว่า

# ========== Utils ==========
def nonzero_frames_ratio(seq30x258: np.ndarray) -> float:
    if seq30x258.shape != (SEQ_LEN, FEAT_DIM):
        return 0.0
    # เฟรมที่ไม่ใช่ศูนย์ (มีแลนด์มาร์กจริง)
    return float(np.any(seq30x258 != 0.0, axis=1).sum()) / float(SEQ_LEN)

# ดึง 258 ฟีเจอร์จากผลลัพธ์ Mediapipe
def extract_258(res) -> np.ndarray:
    """
    สกัดฟีเจอร์ 258 ค่า:
    Pose: 33 * (x,y,z,visibility) = 132
    LH:   21 * (x,y,z) = 63
    RH:   21 * (x,y,z) = 63
    layout เหมือนตอนเทรนใน extractkeypoint.py
    """
    # Pose 33*(x,y,z,visibility) = 132
    if res.pose_landmarks:
        pose = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility]
             for lm in res.pose_landmarks.landmark],
            dtype=np.float32
        ).ravel()
    else:
        pose = np.zeros(33 * 4, dtype=np.float32)

    # Left hand 21*(x,y,z) = 63
    if res.left_hand_landmarks:
        lh = np.array(
            [[lm.x, lm.y, lm.z]
             for lm in res.left_hand_landmarks.landmark],
            dtype=np.float32
        ).ravel()
    else:
        lh = np.zeros(21 * 3, dtype=np.float32)

    # Right hand 21*(x,y,z) = 63
    if res.right_hand_landmarks:
        rh = np.array(
            [[lm.x, lm.y, lm.z]
             for lm in res.right_hand_landmarks.landmark],
            dtype=np.float32
        ).ravel()
    else:
        rh = np.zeros(21 * 3, dtype=np.float32)

    return np.concatenate([pose, lh, rh]).astype(np.float32)  # (258,)

def draw_header(image, label_text, conf):
    H, W = image.shape[:2]
    cv2.rectangle(image, (0, 0), (W, 64), (0, 0, 0), -1)
    cv2.putText(image, f"{label_text} ({conf*100:.1f}%)",
                (12, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255, 255, 255), 2, cv2.LINE_AA)

def draw_topk_bars(image, labels, probs, k=3, origin=(12, 80)):
    H, W = image.shape[:2]
    x0, y0 = origin
    bar_w = int(W * 0.45)
    bar_h = 24
    gap   = 10
    idxs = np.argsort(probs)[-k:][::-1]
    for i, idx in enumerate(idxs):
        p = float(probs[idx])
        w = int(bar_w * p)
        y = y0 + i * (bar_h + gap)
        cv2.rectangle(image, (x0, y), (x0 + bar_w, y + bar_h), (60, 60, 60), 1)
        cv2.rectangle(image, (x0, y), (x0 + w, y + bar_h), (40, 180, 255), -1)
        cv2.putText(image, f"{labels[idx]}  {p*100:5.1f}%",
                    (x0 + bar_w + 12, y + bar_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)

# ========== Load labels / thresholds ==========
print("TF:", tf.__version__)

if not os.path.exists(TFLITE_MODEL):
    raise FileNotFoundError(f"Not found TFLite model file: {TFLITE_MODEL}")
print("[OK] Found TFLite model:", TFLITE_MODEL)

if not os.path.exists(LABEL_MAP_PATH):
    raise FileNotFoundError(f"Not found label map: {LABEL_MAP_PATH}")
with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
    label_map = json.load(f)   # {"0":"classA","1":"classB",...}
labels = [label_map[str(i)] for i in range(len(label_map))]
num_classes = len(labels)
print(f"[OK] Labels ({num_classes}):", labels)

per_th = {c: DEFAULT_THRESH for c in labels}
if os.path.exists(THRESH_PATH):
    try:
        with open(THRESH_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # รองรับทั้ง {"0":{"threshold":0.7}, ...} หรือ {"fever":{"threshold":0.7}, ...}
        for k, v in raw.items():
            cname = labels[int(k)] if k.isdigit() and int(k) < len(labels) else k
            if cname in per_th:
                if isinstance(v, dict) and "threshold" in v:
                    per_th[cname] = float(v["threshold"])
                else:
                    per_th[cname] = float(v)
        print("[OK] thresholds.json loaded.")
    except Exception as e:
        print("[Warn] thresholds.json ignored:", e)

# ========== Load TFLite model ==========
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL)

# เตรียม input shape ให้ตรง [1, SEQ_LEN, FEAT_DIM]
input_details = interpreter.get_input_details()
input_index   = input_details[0]['index']
input_shape   = list(input_details[0]['shape'])

if input_shape != [1, SEQ_LEN, FEAT_DIM]:
    try:
        interpreter.resize_tensor_input(input_index, [1, SEQ_LEN, FEAT_DIM])
        print(f"[Info] Resize TFLite input from {input_shape} to [1,{SEQ_LEN},{FEAT_DIM}]")
    except Exception as e:
        print("[Warn] cannot resize input tensor:", e)

interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
output_index   = output_details[0]['index']
input_dtype    = input_details[0]['dtype']

print("[OK] TFLite interpreter ready.")
print("  Input:", input_details[0]['shape'], input_dtype)
print("  Output:", output_details[0]['shape'], output_details[0]['dtype'])

def run_tflite(x_1x30x258: np.ndarray) -> np.ndarray:
    """
    x_1x30x258: np.ndarray รูป [1,SEQ_LEN,FEAT_DIM]
    คืนค่า probs: np.ndarray [num_classes] (float32)
    """
    # แปลง dtype ให้ตรงกับโมเดล (FP16/FP32)
    x_in = x_1x30x258.astype(input_dtype)
    interpreter.set_tensor(input_details[0]['index'], x_in)
    interpreter.invoke()
    out = interpreter.get_tensor(output_index)  # [1, num_classes]
    # แปลงกลับเป็น float32 สำหรับคำนวณต่อ
    return out[0].astype(np.float32)

# ========== Mediapipe ==========
mp_holistic = mp.solutions.holistic
holistic_kwargs = dict(
    static_image_mode=False,
    model_complexity=MODEL_COMPLEXITY,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ========== Camera loop ==========
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

seq_buf = collections.deque(maxlen=SEQ_LEN)
prev_probs = None
frame_id = 0

shown_label, shown_conf = "...", 0.0
candidate_label, candidate_streak = None, 0

with mp_holistic.Holistic(**holistic_kwargs) as holistic:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # สะท้อนภาพซ้าย-ขวาให้เหมือนหน้ากระจก ใช้กับท่าทางมือถนัดกว่า
        frame = cv2.flip(frame, 1)

        # รัน mediapipe (แปลงเป็น RGB ข้างใน)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = holistic.process(rgb)
        rgb.flags.writeable = True

        # เก็บฟีเจอร์ 258 ลงบัฟเฟอร์ 30 เฟรม
        seq_buf.append(extract_258(res))

        # คำนวณผลเป็นจังหวะ
        if (frame_id % PROCESS_EVERY_N == 0) and len(seq_buf) == SEQ_LEN:
            x = np.array(seq_buf, dtype=np.float32)[None, ...]  # [1,30,258]

            # ===== ใช้ TFLite แทน Keras =====
            probs = run_tflite(x)  # [C], float32

            # EMA smoothing เพื่อความนิ่ง
            smoothed = probs if prev_probs is None else (
                ALPHA_EMA * probs + (1 - ALPHA_EMA) * prev_probs
            )
            prev_probs = smoothed

            # จัดอันดับ
            top3 = np.argsort(smoothed)[-3:][::-1]
            top, second = int(top3[0]), int(top3[1])
            name_top = labels[top]
            conf_top = float(smoothed[top])
            conf_second = float(smoothed[second])
            margin = conf_top - conf_second

            need  = per_th.get(name_top, DEFAULT_THRESH)
            cover = nonzero_frames_ratio(x[0])   # ตรวจว่ามีแลนด์มาร์กจริงพอหรือยัง

            passed = (conf_top >= need) and (margin >= TOP2_MARGIN) and (cover >= MIN_COVERAGE)

            # hysteresis กันกระพริบ
            if passed:
                if candidate_label == name_top:
                    candidate_streak += 1
                else:
                    candidate_label, candidate_streak = name_top, 1
                if candidate_streak >= STABLE_FRAMES:
                    shown_label, shown_conf = name_top, conf_top
            else:
                candidate_label, candidate_streak = None, 0
                shown_label, shown_conf = "...", 0.0

            # วาด Top-3 bars
            draw_topk_bars(frame, labels, smoothed, k=3, origin=(12, 84))

        # แถบหัวแสดงผล
        draw_header(frame, shown_label, shown_conf)

        cv2.imshow("TSL Medical Gesture — LSTM (TFLite FP16, live)", frame)
        frame_id += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()