import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

CSV_FILE = "keypoints_labeled.csv"

# 🔎 ตรวจสอบไฟล์
if not os.path.exists(CSV_FILE):
    raise ValueError("❌ ไม่พบไฟล์ keypoints_data.csv กรุณารัน detection.py ก่อน")

# ✅ โหลดข้อมูลและล้างข้อมูลเสีย
df = pd.read_csv(CSV_FILE, header=None, on_bad_lines='skip')

# ✅ ลบแถวที่มีค่า missing
df = df.dropna()

# ✅ เก็บเฉพาะแถวที่ label เป็น 0 หรือ 1 (คอลัมน์สุดท้าย)
df = df[df.iloc[:, -1].isin([0, 1])]

# ✅ ตรวจว่าข้อมูลเพียงพอไหม
if df.shape[1] < 15:
    raise ValueError("❌ ต้องมีอย่างน้อย 15 คอลัมน์ (speed, angle, 12 keypoints, label)")

# ✅ แยก Features และ Labels
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# ✅ ปรับขนาดข้อมูล
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ แบ่ง Train/Test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ✅ สร้างโมเดลใหม่ (input = 14)
model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ✅ ฝึกโมเดล
model.fit(X_train, y_train, epochs=20, batch_size=8, validation_data=(X_test, y_test))

# ✅ ประเมินผล
y_pred = (model.predict(X_test) > 0.5).astype("int32")

print("\n📊 Evaluation Report:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ✅ บันทึกโมเดล
model.save("gait_recognition_model.h5")
print("✅ โมเดลถูกบันทึกเรียบร้อยแล้ว ✅")
