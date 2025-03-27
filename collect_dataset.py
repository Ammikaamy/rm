import mediapipe as mp
import cv2
import csv
import time

# Initialize MediaPipe Pose, Hand, and Face Mesh
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

pose = mp_pose.Pose()
hands = mp_hands.Hands()
face_mesh = mp_face_mesh.FaceMesh()

# กำหนดไฟล์ CSV สำหรับเก็บข้อมูล
CSV_FILE = 'gait_data.csv'
HEADER = ['timestamp', 'x_nose', 'y_nose', 'z_nose', 'x_shoulder', 'y_shoulder', 'z_shoulder', 'hand_detected', 'face_detected', 'label']


def detect_abnormal_behavior(landmarks):
    """
    ฟังก์ชันตรวจจับพฤติกรรมผิดปกติ เช่น การชกต่อย, ผลักกัน, การล้ม
    """
    if not landmarks:
        return 0  # ปกติ

    nose = landmarks[mp_pose.PoseLandmark.NOSE]
    shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]

    # ตัวอย่างเงื่อนไข: ถ้ามีการยกมือขึ้นเร็วอาจเป็นการชกต่อย
    if abs(nose.y - shoulder.y) > 0.3:  # หัวสูงกว่าหัวไหล่มากผิดปกติ
        return 1  # อาจเป็นการล้ม

    return 0  # ปกติ


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Cannot access the camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    with open(CSV_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(HEADER)  # เขียน Header

        timestamp = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame.")
                break

            # แปลงภาพเป็น RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ประมวลผลท่าทาง, มือ, และใบหน้า
            pose_results = pose.process(image_rgb)
            hand_results = hands.process(image_rgb)
            face_results = face_mesh.process(image_rgb)

            # ตรวจสอบว่าพบ landmark หรือไม่
            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark
                x_nose, y_nose, z_nose = landmarks[mp_pose.PoseLandmark.NOSE].x, landmarks[mp_pose.PoseLandmark.NOSE].y, landmarks[mp_pose.PoseLandmark.NOSE].z
                x_shoulder, y_shoulder, z_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].z

                # ตรวจจับการยกมือขึ้น
                hand_detected = 1 if hand_results.multi_hand_landmarks else 0
                face_detected = 1 if face_results.multi_face_landmarks else 0

                # ตรวจจับพฤติกรรมผิดปกติ
                label = detect_abnormal_behavior(landmarks)

                # บันทึกข้อมูลลง CSV
                writer.writerow([timestamp, x_nose, y_nose, z_nose, x_shoulder, y_shoulder, z_shoulder, hand_detected, face_detected, label])
                timestamp += 1

            # แสดงภาพ
            cv2.imshow('Frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
