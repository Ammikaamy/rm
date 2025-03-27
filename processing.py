import cv2
import numpy as np
from detection import detect_abnormal_behavior, pose

def process_frame(frame):
    # แปลงภาพเป็น RGB สำหรับ MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # ตรวจจับ Pose
    results = pose.process(frame_rgb)
    
    abnormal_behavior = False
    if results.pose_landmarks:
        # แปลงข้อมูลจุด landmark เป็น array
        keypoints = [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
        abnormal_behavior = detect_abnormal_behavior(keypoints)  # ใช้โมเดลตรวจจับพฤติกรรมผิดปกติ
    
    # เปลี่ยนกรอบสีตามผลลัพธ์
    color = (0, 0, 255) if abnormal_behavior else (0, 255, 0)  # สีแดงถ้าผิดปกติ, เขียวถ้าปกติ
    frame = cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), color, 10)

    return frame
