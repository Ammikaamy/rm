import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pandas as pd
import os
from ultralytics import YOLO

# โหลด YOLOv8
yolo_model = YOLO("yolov8n.pt")

# โหลดโมเดล gait recognition ถ้ามี
if os.path.exists("gait_recognition_model.h5"):
    gait_model = tf.keras.models.load_model("gait_recognition_model.h5")
    print("✅ โหลดโมเดล gait_recognition_model.h5 สำเร็จ")
else:
    gait_model = None
    print("⚠️ ไม่พบโมเดล gait_recognition_model.h5 — จะไม่ใช้โมเดลในการพยากรณ์")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.8
)

previous_positions = {}
speed_history = {}

CSV_FILE = "keypoints_data.csv"

def ema(prev_value, new_value, alpha=0.2):
    return alpha * new_value + (1 - alpha) * prev_value if prev_value is not None else new_value

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return angle

def detect_abnormal_behavior(speed, angle):
    return speed > 5 and angle > 100

def save_keypoints_to_csv(keypoints, speed, angle, filename=CSV_FILE):
    data = [speed, angle] + list(keypoints)
    df = pd.DataFrame([data])
    if not os.path.exists(filename) or os.stat(filename).st_size == 0:
        df.to_csv(filename, mode="w", header=False, index=False)
    else:
        df.to_csv(filename, mode="a", header=False, index=False)    

def detect_persons(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = yolo_model(image_rgb)
    persons = []
    for result in results[0].boxes.data:
        x1, y1, x2, y2, score, class_id = result.tolist()
        if int(class_id) == 0 and score > 0.5:
            persons.append((int(x1), int(y1), int(x2), int(y2)))
    return persons

def detect_pose(frame, persons):
    global previous_positions, speed_history
    actions = []
    bounding_boxes = []

    for (x1, y1, x2, y2) in persons:
        person_crop = frame[y1:y2, x1:x2]
        image_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            keypoints = np.array([[lm.x, lm.y] for lm in landmarks])
            selected_indices = [11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
            selected_keypoints = keypoints[selected_indices].flatten()[:12]

            person_id = (x1 + x2) // 2, (y1 + y2) // 2
            prev = previous_positions.get(person_id, person_id)
            raw_speed = np.linalg.norm([prev[0] - person_id[0], prev[1] - person_id[1]])
            speed = ema(speed_history.get(person_id, raw_speed), raw_speed)
            speed_history[person_id] = speed
            previous_positions[person_id] = person_id

            left_shoulder = keypoints[11]
            left_elbow = keypoints[13]
            left_wrist = keypoints[15]
            angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

            if gait_model is not None:
                input_data = np.array([[speed, angle] + list(selected_keypoints)], dtype=np.float32)
                prediction = gait_model.predict(input_data)[0][0]
                action = "Abnormal" if detect_abnormal_behavior(speed, angle) and prediction > 0.7 else "Normal"
            else:
                prediction = None
                action = "DataCollection"
                save_keypoints_to_csv(selected_keypoints, speed, angle)

            mp_drawing.draw_landmarks(frame[y1:y2, x1:x2], results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            actions.append(action)
            bounding_boxes.append((x1, y1, x2, y2))

    return actions, bounding_boxes

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ ไม่สามารถเปิดกล้องได้")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        persons = detect_persons(frame)
        actions, boxes = detect_pose(frame, persons)

        for action, (x1, y1, x2, y2) in zip(actions, boxes):
            color = (0, 0, 255) if action == "Abnormal" else (0, 255, 0)
            label = f"{action}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow("Real-Time Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()