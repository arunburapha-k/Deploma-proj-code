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


def extract_keypoints(results, prev_lh=None, prev_rh=None):
    """
    ดึงค่าพิกัดสัมพัทธ์ Dimension: 258
    🔥 เพิ่มระบบจำค่ามือล่าสุด (Forward Fill) ป้องกันการวาร์ปเมื่อจับมือไม่ได้
    """
    ref_x, ref_y, ref_z = 0.5, 0.5, 0.0
    body_size = 1.0

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        ref_x = (landmarks[11].x + landmarks[12].x) / 2
        ref_y = (landmarks[11].y + landmarks[12].y) / 2
        ref_z = (landmarks[11].z + landmarks[12].z) / 2

        dist_x = landmarks[11].x - landmarks[12].x
        dist_y = landmarks[11].y - landmarks[12].y
        body_size = np.sqrt(dist_x**2 + dist_y**2)

        if body_size < 0.001:
            body_size = 1.0

    def get_relative_coords(landmarks_obj, is_pose=False, prev_state=None):
        if not landmarks_obj:
            # 🔥 ถ้าไม่เจอจุด (เช่น มือหาย) ให้ใช้ค่าจากเฟรมก่อนหน้า ถ้ามี
            if prev_state is not None and np.any(prev_state != 0):
                return prev_state
            # ถ้าไม่มีจริงๆ ค่อยคืนค่าศูนย์ (กรณีเริ่มคลิปมาก็ไม่เจอมือเลย)
            return np.zeros(33 * 4) if is_pose else np.zeros(21 * 3)

        data = []
        for res in landmarks_obj.landmark:
            rel_x = (res.x - ref_x) / body_size
            rel_y = (res.y - ref_y) / body_size
            rel_z = (res.z - ref_z) / body_size

            if is_pose:
                data.append([rel_x, rel_y, rel_z, res.visibility])
            else:
                data.append([rel_x, rel_y, rel_z])

        return np.array(data).flatten()

    pose = get_relative_coords(results.pose_landmarks, is_pose=True)
    lh = get_relative_coords(
        results.left_hand_landmarks, is_pose=False, prev_state=prev_lh
    )
    rh = get_relative_coords(
        results.right_hand_landmarks, is_pose=False, prev_state=prev_rh
    )

    return np.concatenate([pose, lh, rh]), lh, rh


# --- Config หลัก ---
RAW_DATA_PATH = os.path.join("data", "raw")
PROCESSED_DATA_PATH = os.path.join("data", "processed")

# รายชื่อท่าทาง
actions = np.array(
    [
        # "anxiety",
        # "fever",
        # "feverish",
        # "insomnia",
        # "itching",
        # "no_action",
        # "polyuria",
        # "suffocated",
        # "wounded",
        "breathing_difficulty_p",
        # "fever_p",
        # "polyuria_p",
    ]
)



sequence_length = 30
num_features = 258

for action in actions:
    os.makedirs(os.path.join(PROCESSED_DATA_PATH, action), exist_ok=True)
print(f"Ensured '{PROCESSED_DATA_PATH}' folders exist.")

print("--- Starting Video Preprocessing ---")

# 🔥 อัปเกรด: ใช้ model_complexity=2 และบังคับใช้ Tracker
with mp_holistic.Holistic(
    static_image_mode=False,  # บังคับ False เพื่อใช้ฟีเจอร์ Tracking ให้เส้นเนียน
    model_complexity=2,  # เพิ่มความแม่นยำระดับสุดยอด
    smooth_landmarks=True,  # เปิดระบบลดการสั่น
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as holistic:

    for action in actions:
        action_raw_path = os.path.join(RAW_DATA_PATH, action)
        action_processed_path = os.path.join(PROCESSED_DATA_PATH, action)

        if not os.path.exists(action_raw_path):
            continue

        video_files = [
            f
            for f in os.listdir(action_raw_path)
            if f.endswith((".mp4", ".avi", ".mov"))
        ]
        print(f"\nProcessing Action: '{action}' ({len(video_files)} videos found)")

        for sequence_idx, video_file in enumerate(video_files):
            video_path = os.path.join(action_raw_path, video_file)
            cap = cv2.VideoCapture(video_path)

            all_frames_data = []  # เก็บทุกเฟรมไว้ก่อน

            # ตัวแปรจำค่ามือล่าสุดในคลิปนี้
            prev_lh = np.zeros(21 * 3)
            prev_rh = np.zeros(21 * 3)

            # 🔥 อัปเกรด: อ่านวิดีโอเรียงเฟรมตามธรรมชาติ (ไม่กระโดดข้าม) Tracker จะได้ไม่พัง
            while True:
                success, frame = cap.read()
                if not success:
                    break
                # Mirror
                frame = cv2.flip(frame, 1)
                results = mediapipe_process(frame, holistic)

                # ส่งค่า prev_lh, prev_rh เข้าไปอัปเดต
                keypoints, prev_lh, prev_rh = extract_keypoints(
                    results, prev_lh, prev_rh
                )
                all_frames_data.append(keypoints)

            cap.release()

            # 🔥 เมื่อได้ครบทุกเฟรม ค่อยมา Sample ทีหลังให้เหลือ 30 เฟรม
            total_extracted = len(all_frames_data)
            if total_extracted < sequence_length:
                print(
                    f"  [Warning] Video {video_file} is too short ({total_extracted} frames). Skipping."
                )
                continue

            # ดึง index กระจายตัวให้ได้ 30 เฟรมพอดี
            frame_indices = np.linspace(
                0, total_extracted - 1, sequence_length, dtype=int
            )
            sequence_data = np.array(all_frames_data)[frame_indices]

            npy_path = os.path.join(action_processed_path, f"{sequence_idx}.npy")
            np.save(npy_path, sequence_data)

            print(
                f"\r  Processed {sequence_idx + 1}/{len(video_files)} videos...", end=""
            )

        print(f'\nAction "{action}" complete.')

print("\n--- Preprocessing Complete! ---")
