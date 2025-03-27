import sqlite3
from datetime import datetime

DB_NAME = "behavior.db"

def create_database():
    """สร้างฐานข้อมูลและตารางหากยังไม่มี"""
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS behavior_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                behavior_type TEXT,
                timestamp TEXT,
                speed REAL,
                angle REAL,
                keypoints TEXT
            )
        ''')
        conn.commit()

def check_table_exists():
    """ตรวจสอบว่าตาราง behavior_data มีอยู่หรือไม่"""
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='behavior_data';")
        table = c.fetchone()
        if table is None:
            print("⚠️ ตาราง behavior_data ไม่มีอยู่ในฐานข้อมูล! กำลังสร้างใหม่...")
            create_database()

def save_behavior_data(behavior_type, speed, angle, keypoints):
    """บันทึกพฤติกรรมลงฐานข้อมูล"""
    check_table_exists()  # ตรวจสอบตารางก่อนบันทึก
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        c.execute('''
            INSERT INTO behavior_data (behavior_type, timestamp, speed, angle, keypoints)
            VALUES (?, ?, ?, ?, ?)
        ''', (behavior_type, timestamp, speed, angle, keypoints))
        conn.commit()
        print(f"✅ บันทึกพฤติกรรม: {behavior_type} เวลา {timestamp}")

def fetch_behavior_data(limit=10):
    """ดึงข้อมูลพฤติกรรมล่าสุดจากฐานข้อมูล"""
    check_table_exists()
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        c.execute('SELECT * FROM behavior_data ORDER BY id DESC LIMIT ?', (limit,))
        data = c.fetchall()
    return data

def delete_behavior_data(record_id):
    """ลบข้อมูลพฤติกรรมตาม ID"""
    check_table_exists()
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        c.execute('DELETE FROM behavior_data WHERE id = ?', (record_id,))
        conn.commit()
        print(f"🗑️ ลบข้อมูล ID {record_id} สำเร็จ")

def clear_all_data():
    """ล้างข้อมูลทั้งหมดในฐานข้อมูล"""
    check_table_exists()
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        c.execute('DELETE FROM behavior_data')
        conn.commit()
        print("⚠️ ล้างข้อมูลทั้งหมดเรียบร้อย!")

def update_behavior_data(record_id, behavior_type=None, speed=None, angle=None, keypoints=None):
    """อัปเดตข้อมูลพฤติกรรมที่มีอยู่"""
    check_table_exists()
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        update_fields = []
        values = []

        if behavior_type:
            update_fields.append("behavior_type = ?")
            values.append(behavior_type)
        if speed is not None:
            update_fields.append("speed = ?")
            values.append(speed)
        if angle is not None:
            update_fields.append("angle = ?")
            values.append(angle)
        if keypoints:
            update_fields.append("keypoints = ?")
            values.append(keypoints)

        values.append(record_id)
        sql_query = f"UPDATE behavior_data SET {', '.join(update_fields)} WHERE id = ?"
        c.execute(sql_query, values)
        conn.commit()
        print(f"✏️ อัปเดตข้อมูล ID {record_id} สำเร็จ")

def search_behavior_data(behavior_type=None, start_date=None, end_date=None):
    """ค้นหาข้อมูลพฤติกรรมตามประเภท หรือช่วงวันที่"""
    check_table_exists()
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        query = "SELECT * FROM behavior_data WHERE 1=1"
        params = []

        if behavior_type:
            query += " AND behavior_type = ?"
            params.append(behavior_type)
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)

        c.execute(query, params)
        results = c.fetchall()
    return results
