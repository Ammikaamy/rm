import pandas as pd

# โหลดไฟล์เดิม
df = pd.read_csv("keypoints_data.csv", header=None)

# เพิ่มคอลัมน์ label = 0 ทุกแถว
df["label"] = 0

# บันทึกเป็นไฟล์ใหม่ หรือทับไฟล์เดิม
df.to_csv("keypoints_labeled.csv", index=False, header=False)

print("✅ เพิ่มคอลัมน์ label = 0 สำเร็จ → บันทึกเป็น keypoints_labeled.csv")
