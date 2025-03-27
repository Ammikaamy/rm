import cv2
from camera import get_camera
from processing import process_frame

def display_video():
    cap = get_camera()
    
    if cap is None or not cap.isOpened():
        print("❌ ไม่สามารถเปิดกล้องได้! ตรวจสอบการเชื่อมต่อกล้อง")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ ไม่สามารถอ่านเฟรมจากกล้องได้!")
                break
            
            # ประมวลผลภาพ
            frame = process_frame(frame)
            
            # แสดงผลภาพ
            cv2.imshow('Gait Detection', frame)
            
            # รอการกดปุ่ม 'q' เพื่อออกจากการแสดงผล
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("🛑 ปิดหน้าต่างแสดงผล")
                break

    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดระหว่างทำงาน: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ กล้องถูกปิดและหน้าต่างถูกทำลายสำเร็จ")

if __name__ == "__main__":
    display_video()
