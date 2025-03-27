import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model
from detection import detect_persons, detect_pose
from play_sound import play_beep
from database import save_behavior_data

model = load_model('gait_recognition_model.h5')
print("✅ โหลดโมเดลสำเร็จ")

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ ไม่สามารถเปิดกล้องได้")
        break

    persons = detect_persons(frame)
    actions, boxes = detect_pose(frame, persons)

    abnormal_detected = False

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        color = (0, 255, 0)
        speed = 10  # ← ปรับให้คืนค่าจริงในอนาคตจาก detect_pose
        angle = 120

        if actions[i] == "Abnormal":
            abnormal_detected = True
            color = (0, 0, 255)
            play_beep(True)
            save_behavior_data("Abnormal", speed, angle, "Keypoints")
            cv2.putText(frame, "⚠ Abnormal Behavior Detected!", (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"Action: {actions[i]}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    if not abnormal_detected:
        cv2.putText(frame, "No Abnormal Behavior", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Multi-Person Pose Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
