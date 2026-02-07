import cv2
import os
import time
import numpy as np

# --- 1. การตั้งค่า (Config) ---
RAW_DATA_PATH = os.path.join('data', 'raw') 

# ชื่อท่าทางที่ต้องการเก็บ (แก้ตรงนี้ตามต้องการ)
actions = np.array(['distention']) 

no_sequences = 60     # จำนวนคลิปต่อ 1 ท่า
sequence_length = 60  # จำนวนเฟรมต่อ 1 คลิป (ตรงกับที่คุณใช้ตอน Extract)
start_delay = 2       # เวลาพักระหว่างคลิป (วินาที) ให้เราเตรียมตัว

# สร้างโฟลเดอร์รอไว้เลย
for action in actions:
    try:
        os.makedirs(os.path.join(RAW_DATA_PATH, action))
    except:
        pass

# --- 2. เริ่มเปิดกล้อง ---
cap = cv2.VideoCapture(0) # เลข 0 คือกล้อง Webcam, ถ้าใช้กล้องนอกลองเปลี่ยนเป็น 1, 2

# เช็คว่าเปิดกล้องได้ไหม
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# ตั้งค่าความละเอียด (Optional: ปรับให้ตรงกับที่จะใช้จริง)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

print("--- Starting Data Collection ---")
print("Press 'q' to quit early.")

for action in actions:
    print(f"Collecting data for action: {action}")
    
    # รอให้คนเตรียมตัวก่อนเริ่มท่าใหม่
    print("Get Ready! Starting in 5 seconds...")
    time.sleep(5) 
    
    for sequence in range(no_sequences):
        
        # ตั้งชื่อไฟล์ .mp4
        save_path = os.path.join(RAW_DATA_PATH, action, f'{action}_{sequence}.mp4')
        
        # ตั้งค่า VideoWriter เพื่อบันทึกไฟล์
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # หรือใช้ 'XVID' ถ้าเป็น .avi
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 30.0 
        out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        
        # --- Loop เก็บ 30 เฟรม ---
        frames_captured = 0
        while frames_captured < sequence_length:
            ret, frame = cap.read()
            if not ret:
                break
            
            # เขียนเฟรมลงไฟล์
            out.write(frame)
            
            # --- ส่วนแสดงผลบนหน้าจอ (GUI) ---
            display_frame = frame.copy()
            
            # ข้อความบอกสถานะ
            cv2.putText(display_frame, f'Action: {action}', (15, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(display_frame, f'Video: {sequence}/{no_sequences}', (15, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(display_frame, f'Frame: {frames_captured}/{sequence_length}', (15, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
            
            # แสดงภาพสด
            cv2.imshow('OpenCV Data Collection', display_frame)
            
            frames_captured += 1
            
            # กด q เพื่อหยุดกะทันหัน
            if cv2.waitKey(1) & 0xFF == ord('q'):
                out.release()
                cap.release()
                cv2.destroyAllWindows()
                exit()
        
        # บันทึกเสร็จ ปิดไฟล์คลิปนี้
        out.release()
        
        # --- ช่วงพัก (Break) ระหว่างคลิป ---
        # แสดงหน้าจอ "WAIT" เพื่อให้คนกลับมาท่าเตรียม (Neutral Pose)
        start_time = time.time()
        while (time.time() - start_time) < start_delay:
            ret, frame = cap.read()
            cv2.putText(frame, 'WAIT... Reset Hand', (100, 200), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4, cv2.LINE_AA)
            cv2.putText(frame, f'Next: {action} ({sequence+1}/{no_sequences})', (50, 280),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow('OpenCV Data Collection', frame)
            cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
print("All data collected!")