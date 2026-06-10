import gzip
import pandas as pd
import os

# ระบุชื่อไฟล์ TWOSIDES ที่อยู่ในโฟลเดอร์
twosides_file = "https://github.com/PatthamonCharaschimpleekul/DigitalTwin/releases/download/v1.0/TWOSIDES.csv.gz
"

'''print("==========================================")
if os.path.exists(twosides_file):
    print(f"📦 ตรวจพบไฟล์ DDI ในเครื่อง: {twosides_file}")
    print(f"📊 ขนาดไฟล์ซิป: {os.path.getsize(twosides_file) / (1024*1024):.2f} MB")
    print("==========================================")
else:
    print(f"❌ ไม่พบไฟล์ '{twosides_file}' กรุณาตรวจสอบชื่อไฟล์ในโฟลเดอร์ครับ")
    exit()'''

print("⏳ กำลังสกัดข้อมูลคู่ยาตีกัน (DDI) จาก TWOSIDES...")

try:
    # เปิดอ่านไฟล์ซิปโดยตรง และดึงมาเฉพาะ 50,000 แถวแรกเพื่อมาส่องโครงสร้าง
    with gzip.open(twosides_file, 'rt', encoding='utf-8') as f:
        # แก้บั๊ก: เอา next() ออก และใช้ nrows ดึงข้อมูลขึ้นมาเป็น DataFrame ตรงๆ
        df_chunk = pd.read_csv(f, nrows=50000)
        
        print("\n✅ อ่านไฟล์สำเร็จ!")
        print("🔍 โครงสร้างคอลัมน์ที่พบในไฟล์ TWOSIDES:")
        print("------------------------------------------")
        print(df_chunk.columns.tolist())
        print("------------------------------------------")
        
        # ดึงมาเซฟเป็นไฟล์ตารางขนาดเล็ก 3,000 แถวสำหรับใช้งาน Local ให้เท่าฝั่งอื่น
        output_filename = "ddi_edges_sample.csv"
        df_chunk.head(3000).to_csv(output_filename, index=False)
        
        print(f"🎉 สกัดข้อมูล 3,000 แถวแรกออกมาเป็น '{output_filename}'")
        print("\n--- ตัวอย่างหน้าตาข้อมูลคู่ยาตีกันของคุณ ---")
        print(df_chunk.head(3).to_string())

except Exception as e:
    print(f"\n❌ เกิดข้อผิดพลาดในการสกัดข้อมูล: {str(e)}")
