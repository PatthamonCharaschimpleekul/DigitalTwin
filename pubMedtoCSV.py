import os
import gzip
import xml.etree.ElementTree as ET
import pandas as pd

# 1. ค้นหาไฟล์ตระกูล pubmed....xml.gz ในโฟลเดอร์อัตโนมัติ
xml_file_path = None
for file in os.listdir('.'):
    if file.startswith('pubmed') and file.endswith('.xml.gz'):
        xml_file_path = file
        break

if xml_file_path:
    print("==========================================")
    print(f"📦 ตรวจพบไฟล์ PubMed ในเครื่อง: {xml_file_path}")
    print(f"📊 ขนาดไฟล์ซิป: {os.path.getsize(xml_file_path) / (1024*1024):.2f} MB")
    print("==========================================")
else:
    print("❌ ไม่พบไฟล์ที่ชื่อขึ้นต้นด้วย 'pubmed' และลงท้ายด้วย '.xml.gz' ในโฟลเดอร์นี้")
    print("กรุณาตรวจสอบว่าย้ายไฟล์ที่ดาวน์โหลดมาไว้ในโฟลเดอร์ทำงานหรือยังครับ")
    exit()

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