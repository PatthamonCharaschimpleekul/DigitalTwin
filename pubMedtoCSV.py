import os
import gzip
import xml.etree.ElementTree as ET
import pandas as pd

# --- เริ่มต้นโค้ดใหม่ดึงข้อมูลจาก GitHub Release ---
import urllib.request

# 1. เปลี่ยนตรงนี้เป็นลิงก์ URL ของไฟล์ .xml.gz ที่คุณอัปโหลดไว้ใน GitHub Release
xml_online_url = "https://github.com/PatthamonCharaschimpleekul/DigitalTwin/releases/download/v1.0/pubmed26n0001.xml.gz
"

print("=========================================")
print("🌐 กำลังเชื่อมต่อดึงบิ๊กดาต้า PubMed จาก Cloud Release...")
print("=========================================")

try:
    # สร้างท่อเชื่อมต่อดาวน์โหลดไฟล์ตรงผ่านอินเทอร์เน็ต
    response = urllib.request.urlopen(xml_online_url)
    # ส่งท่อข้อมูลเข้าสู่ระบบปลดบล็อก gzip ต่อได้เลย
    xml_file_path = gzip.GzipFile(fileobj=response)
    print("✅ การเชื่อมต่อสำเร็จ! กำลังเริ่มแกะข้อมูลวิทยาศาสตร์การแพทย์...")
except Exception as e:
    print(f"❌ เกิดข้อผิดพลาดในการดึงข้อมูลจาก Cloud: {e}")
    exit()
# --- สิ้นสุดโค้ดใหม่ ---

# 2. ตั้งค่าเตรียมสกัดข้อมูล
data_list = []
max_records = 3000  # ดึงมา 3,000 เรื่องเพื่อให้สเกลเท่ากับข้อมูลฝั่ง PubChem
print(f"🧬 กำลังเริ่มแกะข้อมูลบทคัดย่อวิจัยทางการแพทย์จำนวน {max_records} เรื่อง...")

try:
    with gzip.open(xml_file_path, 'rb') as f:
        # ใช้ iterparse เพื่ออ่านไฟล์ทีละบรรทัดแบบประหยัดหน่วยความจำ
        context = ET.iterparse(f, events=('end',))
        count = 0
        
        for event, elem in context:
            if elem.tag == 'PubmedArticle':
                # สกัดเลขทะเบียนงานวิจัย (PMID)
                pmid_elem = elem.find('.//PMID')
                pmid = pmid_elem.text if pmid_elem is not None else ""
                
                # สกัดชื่อหัวข้องานวิจัย (Article Title)
                title_elem = elem.find('.//ArticleTitle')
                title = title_elem.text if title_elem is not None else ""
                
                # สกัดเนื้อหาบทคัดย่อ (Abstract Text)
                abstract_texts = elem.findall('.//AbstractText')
                abstract = " ".join([t.text for t in abstract_texts if t.text])
                
                # บันทึกเมื่อมีข้อมูลครบถ้วน
                if title and abstract:
                    data_list.append({
                        "PubMed_ID": f"PMID_{pmid}",
                        "Article_Title": title,
                        "Abstract_Text": abstract
                    })
                    count += 1
                
                # เคลียร์ข้อมูลในแรมที่อ่านเสร็จแล้วทิ้งทันที
                elem.clear()
                
                if count >= max_records:
                    break
                    
    # 3. แปลงเป็น DataFrame และเซฟไฟล์ตาราง
    df = pd.DataFrame(data_list)
    output_filename = "pubmed_sample.csv"
    df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    
    print(f"\n🎉 สำเร็จแล้ว! สกัดข้อมูลสำเร็จ {count} บทความ")
    print(f"💾 บันทึกไฟล์ตารางตัวอย่างไว้ที่: {output_filename}")


except Exception as e:
    print(f"\n❌ เกิดข้อผิดพลาดในการอ่านไฟล์ XML: {str(e)}")
