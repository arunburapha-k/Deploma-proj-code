import cv2
import mediapipe as mp
import numpy as np
import os

mp_holistic = mp.solutions.holistic


def mediapipe_process(image, model):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = model.process(image_rgb)
    image_rgb.flags.writeable = True
    return results


def extract_keypoints(results):
    """
    ฉบับตัด Z ออก: เหลือแค่ X, Y (และ Visibility สำหรับ Pose)
    Dimension รวม: 183
    """
    # 1. หาจุดอ้างอิง (Reference Point) และ "ขนาดตัว" (Body Size)
    ref_x, ref_y = 0.5, 0.5  # ตัด ref_z ออก
    body_size = 1.0

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        # จุดอ้างอิง: กึ่งกลางไหล่
        ref_x = (landmarks[11].x + landmarks[12].x) / 2
        ref_y = (landmarks[11].y + landmarks[12].y) / 2

        # ขนาดตัว (Distance ไหล่ซ้าย-ขวา)
        dist_x = landmarks[11].x - landmarks[12].x
        dist_y = landmarks[11].y - landmarks[12].y
        body_size = np.sqrt(dist_x**2 + dist_y**2)

        if body_size < 0.001:
            body_size = 1.0

    def get_relative_coords(landmarks_obj, include_vis=False):
        # 🔥 แก้ไขการจองพื้นที่ 0 (Zero Padding)
        if not landmarks_obj:
            # Pose: 33 * 3 (x,y,vis) | Hand: 21 * 2 (x,y)
            return np.zeros(33 * 3) if include_vis else np.zeros(21 * 2)

        data = []
        for res in landmarks_obj.landmark:
            rel_x = (res.x - ref_x) / body_size
            rel_y = (res.y - ref_y) / body_size

            # 🔥 ตัด Z ทิ้ง ไม่เอามาใส่ใน list
            if include_vis:
                data.append([rel_x, rel_y, res.visibility])  # 3 ค่า
            else:
                data.append([rel_x, rel_y])  # 2 ค่า

        return np.array(data).flatten()

    # 2. เรียกใช้
    pose = get_relative_coords(results.pose_landmarks, include_vis=True)
    lh = get_relative_coords(results.left_hand_landmarks, include_vis=False)
    rh = get_relative_coords(results.right_hand_landmarks, include_vis=False)

    return np.concatenate([pose, lh, rh])


# --- Config หลัก ---
RAW_DATA_PATH = os.path.join("data", "raw")
PROCESSED_DATA_PATH = os.path.join("data", "processed")

# รายชื่อท่าทาง
actions = np.array(["fever", "feverish", "insomnia", "no_action", "wounded"])

sequence_length = 30
num_features = 18

# --- 4. สร้างโฟลเดอร์ปลายทาง ---
for action in actions:
    action_path = os.path.join(PROCESSED_DATA_PATH, action)
    os.makedirs(action_path, exist_ok=True)
print(f"Ensured '{PROCESSED_DATA_PATH}' folders exist.")

# --- 5. ลูปหลักสำหรับประมวลผลวิดีโอ ---
print("--- Starting Video Preprocessing (Relative Coordinates) ---")

with mp_holistic.Holistic(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
) as holistic:

    for action in actions:
        action_raw_path = os.path.join(RAW_DATA_PATH, action)
        action_processed_path = os.path.join(PROCESSED_DATA_PATH, action)

        if not os.path.exists(action_raw_path):
            print(
                f"[Warning] Source folder not found: {action_raw_path}. Skipping '{action}'."
            )
            continue

        video_files = [
            f
            for f in os.listdir(action_raw_path)
            if f.endswith((".mp4", ".avi", ".mov", ".MOV", ".mkv"))
        ]

        print(f"\nProcessing Action: '{action}' ({len(video_files)} videos found)")

        for sequence_idx, video_file in enumerate(video_files):
            video_path = os.path.join(action_raw_path, video_file)

            sequence_data = []
            cap = cv2.VideoCapture(video_path)

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames < sequence_length:
                print(
                    f"  [Warning] Video {video_file} is too short ({total_frames} frames). Skipping."
                )
                cap.release()
                continue

            frame_indices = np.linspace(0, total_frames - 1, sequence_length, dtype=int)

            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                success, frame = cap.read()

                if not success:
                    sequence_data.append(np.zeros(num_features))
                    continue
                # เลข 1 หมายถึง Flip แนวนอน (Horizontal)
                frame = cv2.flip(frame, 1)
                results = mediapipe_process(frame, holistic)
                keypoints = extract_keypoints(results)  # ใช้ฟังก์ชันใหม่ตรงนี้
                sequence_data.append(keypoints)

            cap.release()

            npy_path = os.path.join(action_processed_path, f"{sequence_idx}.npy")
            np.save(npy_path, np.array(sequence_data))

            print(
                f"\r  Processed {sequence_idx + 1}/{len(video_files)} videos...", end=""
            )

        print(f'\nAction "{action}" complete.')

print("\n--- Preprocessing Complete! ---")
