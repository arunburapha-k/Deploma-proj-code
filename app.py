# run_live_lstm_tflite.py
# ฉบับอัปเดต:
# 1. แสดงผลแบบ 9:16 Center Crop (เหมือน collect_data.py)
# 2. ใช้ Logic สกัดจุดแบบเดียวกับ extractkeypoint.py เป๊ะๆ (Pose + Hands)
# 3. เพิ่มการวาดเส้นมือ (Hands) ลงบนหน้าจอเพื่อความชัดเจน

import os, json, collections
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# ========== CONFIG ==========
MODEL_DIR = "models"
TFLITE_MODEL = os.path.join(MODEL_DIR, "model_fp16.tflite")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map.json")
THRESH_PATH = os.path.join(MODEL_DIR, "thresholds.json")

SEQ_LEN = 30
FEAT_DIM = 258

# UI / เสถียรภาพ
PROCESS_EVERY_N = 1
ALPHA_EMA = 0.20
DEFAULT_THRESH = 0.70
TOP2_MARGIN = 0.20
MIN_COVERAGE = 0.50
STABLE_FRAMES = 30

CAM_INDEX = 0
# ตั้งค่ากล้องให้ละเอียดที่สุด (เดี๋ยวเราจะมาตัดออก)
FRAME_W, FRAME_H = 1920, 1080

MODEL_COMPLEXITY = 1


# ========== Utils ==========
def nonzero_frames_ratio(seq30x258: np.ndarray) -> float:
    if seq30x258.shape != (SEQ_LEN, FEAT_DIM):
        return 0.0
    return float(np.any(seq30x258 != 0.0, axis=1).sum()) / float(SEQ_LEN)


# 🔥🔥🔥 แก้ไขฟังก์ชันนี้ให้เหมือน extractkeypoint.py เป๊ะๆ (รวมมือด้วย) 🔥🔥🔥
def extract_258(results) -> np.ndarray:
    """
    สกัดฟีเจอร์ 258 ค่า: Pose(132) + L_Hand(63) + R_Hand(63)
    """
    ref_x, ref_y, ref_z = 0.5, 0.5, 0.0  # 🔥 เพิ่ม Z
    body_size = 1.0

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        ref_x = (landmarks[11].x + landmarks[12].x) / 2
        ref_y = (landmarks[11].y + landmarks[12].y) / 2
        ref_z = (landmarks[11].z + landmarks[12].z) / 2  # 🔥 เพิ่ม Z

        dist_x = landmarks[11].x - landmarks[12].x
        dist_y = landmarks[11].y - landmarks[12].y
        body_size = np.sqrt(dist_x**2 + dist_y**2)
        if body_size < 0.001:
            body_size = 1.0

    def get_relative_coords(landmarks_obj, include_vis=False):
        if not landmarks_obj:
            # 🔥 อัปเดตขนาด Padding
            return np.zeros(33 * 4) if include_vis else np.zeros(21 * 3)

        data = []
        for res in landmarks_obj.landmark:
            rel_x = (res.x - ref_x) / body_size
            rel_y = (res.y - ref_y) / body_size
            rel_z = (res.z - ref_z) / body_size  # 🔥 เพิ่ม Z

            if include_vis:
                data.append([rel_x, rel_y, rel_z, res.visibility])
            else:
                data.append([rel_x, rel_y, rel_z])

        return np.array(data, dtype=np.float32).flatten()

    pose = get_relative_coords(results.pose_landmarks, include_vis=True)
    lh = get_relative_coords(results.left_hand_landmarks, include_vis=False)
    rh = get_relative_coords(results.right_hand_landmarks, include_vis=False)

    return np.concatenate([pose, lh, rh])

def draw_header(image, label_text, conf):
    H, W = image.shape[:2]
    cv2.rectangle(image, (0, 0), (W, 80), (0, 0, 0), -1)

    cv2.putText(
        image,
        f"{label_text}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0) if conf > 0 else (200, 200, 200),
        2,
        cv2.LINE_AA,
    )

    if conf > 0:
        cv2.putText(
            image,
            f"Conf: {conf*100:.1f}%",
            (20, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def draw_topk_bars(image, labels, probs, k=3, origin=(20, 100)):
    H, W = image.shape[:2]
    x0, y0 = origin
    bar_w = int(W * 0.85)
    bar_h = 20
    gap = 15
    idxs = np.argsort(probs)[-k:][::-1]

    cv2.putText(
        image,
        "Top Predictions:",
        (x0, y0 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    for i, idx in enumerate(idxs):
        p = float(probs[idx])
        w = int(bar_w * p)
        y = y0 + i * (bar_h + gap)

        cv2.rectangle(image, (x0, y), (x0 + bar_w, y + bar_h), (50, 50, 50), 1)
        color = (0, 255, 255) if i == 0 and p > 0.5 else (100, 100, 100)
        cv2.rectangle(image, (x0, y), (x0 + w, y + bar_h), color, -1)

        text = f"{labels[idx]}: {p*100:.1f}%"
        cv2.putText(
            image,
            text,
            (x0 + 5, y + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255) if p > 0.5 else (180, 180, 180),
            1,
            cv2.LINE_AA,
        )


# ========== Load Config ==========
print("TF:", tf.__version__)

if not os.path.exists(TFLITE_MODEL):
    raise FileNotFoundError(f"Not found TFLite model: {TFLITE_MODEL}")
if not os.path.exists(LABEL_MAP_PATH):
    raise FileNotFoundError(f"Not found label map: {LABEL_MAP_PATH}")

with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
    label_map = json.load(f)
labels = [label_map[str(i)] for i in range(len(label_map))]

per_th = {c: DEFAULT_THRESH for c in labels}
if os.path.exists(THRESH_PATH):
    try:
        with open(THRESH_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for k, v in raw.items():
            cname = labels[int(k)] if k.isdigit() and int(k) < len(labels) else k
            if cname in per_th:
                if isinstance(v, dict) and "threshold" in v:
                    per_th[cname] = float(v["threshold"])
                else:
                    per_th[cname] = float(v)
    except:
        pass

# ========== Load TFLite ==========
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL)
input_details = interpreter.get_input_details()
input_index = input_details[0]["index"]
input_shape = list(input_details[0]["shape"])

if input_shape != [1, SEQ_LEN, FEAT_DIM]:
    try:
        interpreter.resize_tensor_input(input_index, [1, SEQ_LEN, FEAT_DIM])
    except:
        pass

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
output_index = output_details[0]["index"]
input_dtype = input_details[0]["dtype"]


def run_tflite(x_in):
    x_in = x_in.astype(input_dtype)
    interpreter.set_tensor(input_details[0]["index"], x_in)
    interpreter.invoke()
    return interpreter.get_tensor(output_index)[0].astype(np.float32)


# ========== Mediapipe ==========
mp_holistic = mp.solutions.holistic
holistic_kwargs = dict(
    static_image_mode=False,
    model_complexity=MODEL_COMPLEXITY,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ========== Camera Setup ==========
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

# อ่านค่าจริงเพื่อคำนวณการ Crop (Logic เดียวกับ collect_data.py)
real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera Resolution: {real_w}x{real_h}")

# --- 🔥 คำนวณจุดตัด 9:16 🔥 ---
target_w = int(real_h * (9 / 16))  # คำนวณความกว้างตามสัดส่วน 9:16
start_x = (real_w - target_w) // 2
end_x = start_x + target_w

print(f"Crop Region: x={start_x} to {end_x} (Width={target_w}, Height={real_h})")

seq_buf = collections.deque(maxlen=SEQ_LEN)
prev_probs = None
frame_id = 0

shown_label, shown_conf = "Scanning...", 0.0
candidate_label, candidate_streak = None, 0

with mp_holistic.Holistic(**holistic_kwargs) as holistic:
    while True:
        ok, raw_frame = cap.read()
        if not ok:
            break

        # 1. Flip กระจก
        raw_frame = cv2.flip(raw_frame, 1)

        # 2. Crop ตรงกลาง
        frame = raw_frame[:, start_x:end_x]

        # 3. Process
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = holistic.process(rgb)
        rgb.flags.writeable = True

        seq_buf.append(extract_258(res))

        if (frame_id % PROCESS_EVERY_N == 0) and len(seq_buf) == SEQ_LEN:
            x = np.array(seq_buf, dtype=np.float32)[None, ...]
            probs = run_tflite(x)

            smoothed = (
                probs
                if prev_probs is None
                else (ALPHA_EMA * probs + (1 - ALPHA_EMA) * prev_probs)
            )
            prev_probs = smoothed

            top3 = np.argsort(smoothed)[-3:][::-1]
            top, second = int(top3[0]), int(top3[1])
            name_top = labels[top]
            conf_top = float(smoothed[top])
            conf_second = float(smoothed[second])
            margin = conf_top - conf_second

            need = per_th.get(name_top, DEFAULT_THRESH)
            cover = nonzero_frames_ratio(x[0])

            passed = (
                (conf_top >= need)
                and (margin >= TOP2_MARGIN)
                and (cover >= MIN_COVERAGE)
            )

            if passed:
                if candidate_label == name_top:
                    candidate_streak += 1
                else:
                    candidate_label, candidate_streak = name_top, 1
                if candidate_streak >= STABLE_FRAMES:
                    shown_label, shown_conf = name_top, conf_top
            else:
                candidate_label, candidate_streak = None, 0
                shown_label, shown_conf = "Scanning...", 0.0

            draw_topk_bars(frame, labels, smoothed, k=3, origin=(20, 130))

            # 🔥🔥🔥 เพิ่มการวาดเส้นมือทั้งสองข้าง 🔥🔥🔥
            if res.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, res.pose_landmarks, mp_holistic.POSE_CONNECTIONS
                )

            if res.left_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, res.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS
                )

            if res.right_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, res.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
                )

        draw_header(frame, shown_label, shown_conf)

        # แสดงผล
        disp_h = 800
        disp_w = int(target_w * (disp_h / real_h))
        cv2.imshow(
            "App Simulator (9:16 Cropped + Hands)", cv2.resize(frame, (disp_w, disp_h))
        )

        frame_id += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
