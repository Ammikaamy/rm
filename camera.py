import cv2

def get_camera(camera_index=0, width=640, height=480):
    try:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("❌ Cannot access the camera.")
            return None

        # ตั้งค่าความละเอียดของกล้อง
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        return cap
    except Exception as e:
        print(f"⚠️ Error accessing camera: {e}")
        return None
