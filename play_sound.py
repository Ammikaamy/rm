import platform

def play_beep(is_abnormal=False):
    system = platform.system()

    if system == "Windows":
        import winsound
        frequency = 1000  # 1000 Hz
        duration = 500  # 500 milliseconds
        if is_abnormal:
            frequency = 1500  # เสียงเตือนที่สูงขึ้นเมื่อพฤติกรรมผิดปกติ
        winsound.Beep(frequency, duration)
    else:
        import os
        os.system("printf '\\a'")  # ใช้ beep sound บน macOS และ Linux
