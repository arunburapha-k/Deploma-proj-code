import cv2
import mediapipe as mp
import numpy as np
import os

# --- 1. ตั้งค่า MediaPipe Holistic ---
mp_holistic = mp.solutions.holistic
# (เราไม่ต้องการ mp_drawing เพราะเราจะไม่วาดผลลัพธ์)

def mediapipe_process(image, model):
    """
    ประมวลผลภาพ BGR (จาก OpenCV) ด้วย MediaPipe
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = model.process(image_rgb)
    image_rgb.flags.writeable = True
    # (เราไม่ต้องแปลงสีกลับ เพราะเราจะไม่แสดงผล)
    return results

def extract_keypoints(results):
    """
    ‼️ (สำคัญ) สกัดจุดเฉพาะ Pose, LH, RH (ไม่เอา Face) ‼️
    รวม 258 ค่า (Pose: 33*4 + LH: 21*3 + RH: 21*3)
    """
    # 1. Pose (33 landmarks * 4 ค่า = 132)
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    
    # 2. มือซ้าย (21 landmarks * 3 ค่า = 63)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    
    # 3. มือขวา (21 landmarks * 3 ค่า = 63)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    # รวมทั้งหมด
    return np.concatenate([pose, lh, rh])

# --- 3. ตั้งค่าหลัก (ตรงตามโครงสร้างของคุณ) ---

RAW_DATA_PATH = os.path.join('data', 'raw')
PROCESSED_DATA_PATH = os.path.join('data', 'processed')

actions = np.array([
    # 'distention',
    # 'fever',
    # 'feverish',
    # 'no_action'
    # 'wounded',
])

# จำนวนเฟรมที่จะสุ่มดึงจากวิดีโอ
sequence_length = 30
# จำนวน features ที่เราสกัด (258)
num_features = 258 # (132 + 63 + 63)

# --- 4. สร้างโฟลเดอร์ปลายทาง ---
for action in actions:
    # จะสร้าง data/processed/fever, data/processed/feverish, ...
    action_path = os.path.join(PROCESSED_DATA_PATH, action)
    os.makedirs(action_path, exist_ok=True)
print(f"Ensured '{PROCESSED_DATA_PATH}' folders exist.")


# --- 5. ลูปหลักสำหรับประมวลผลวิดีโอ ---
print("--- Starting Video Preprocessing (No Face) ---")

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    for action in actions:
        action_raw_path = os.path.join(RAW_DATA_PATH, action)
        action_processed_path = os.path.join(PROCESSED_DATA_PATH, action)
        
        if not os.path.exists(action_raw_path):
            print(f"[Warning] Source folder not found: {action_raw_path}. Skipping '{action}'.")
            continue
            
        video_files = [f for f in os.listdir(action_raw_path) 
                       if f.endswith(('.mp4', '.avi', '.mov', '.MOV', '.mkv'))]
        
        print(f"\nProcessing Action: '{action}' ({len(video_files)} videos found)")
        
        for sequence_idx, video_file in enumerate(video_files):
            video_path = os.path.join(action_raw_path, video_file)
            
            sequence_data = [] # เก็บ keypoints ของ 30 เฟรม
            cap = cv2.VideoCapture(video_path)
            
            # (A) คำนวณดัชนีเฟรมที่จะดึง (Frame Sampling)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames < sequence_length:
                print(f"  [Warning] Video {video_file} is too short ({total_frames} frames). Skipping.")
                cap.release()
                continue
                
            frame_indices = np.linspace(0, total_frames - 1, sequence_length, dtype=int)
            
            # (B) วนลูปดึงเฉพาะเฟรมที่เลือก
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx) # กระโดดไปที่เฟรม
                success, frame = cap.read()
                
                if not success:
                    sequence_data.append(np.zeros(num_features)) # เติม 0 ถ้าเฟรมมีปัญหา
                    continue

                # ประมวลผลด้วย MediaPipe (ไม่วาด)
                results = mediapipe_process(frame, holistic)
                
                # สกัด Keypoints (258 features)
                keypoints = extract_keypoints(results)
                sequence_data.append(keypoints)

            cap.release()
            
            # (C) บันทึกข้อมูล
            # sequence_data จะมี shape (30, 258)
            npy_path = os.path.join(action_processed_path, f'{sequence_idx}.npy')
            np.save(npy_path, np.array(sequence_data))
            
            # (แสดง % ความคืบหน้า)
            print(f'\r  Processed {sequence_idx + 1}/{len(video_files)} videos...', end='')
        
        print(f'\nAction "{action}" complete.')

print("\n--- Preprocessing Complete! ---")