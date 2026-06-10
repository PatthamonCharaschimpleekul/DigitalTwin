import gzip
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Draw  # โมดูลสำหรับวาดรูปพันธะเคมี
import urllib.request

gzip_file_path = "https://github.com/PatthamonCharaschimpleekul/DigitalTwin/releases/download/v1.0/Compound_000000001_000500000.1.sdf.gz
"
image_folder = "drug_images"  # โฟลเดอร์สำหรับเก็บรูปโครงสร้างเคมี
os.makedirs(image_folder, exist_ok=True)

data_list = []
max_records = 3000  

print("🧬 กำลังงัดข้อมูลเชิงลึก + วาดรูปพันธะเคมีจาก PubChem...")

try:
    with gzip.open(gzip_file_path, 'rb') as f:
        supplier = Chem.ForwardSDMolSupplier(f)
        count = 0
        for mol in supplier:
            if mol is None:
                continue
            
            pubchem_id = mol.GetProp("_Name") if mol.HasProp("_Name") else str(count+1)
            cid_key = f"CID_{pubchem_id}"
            
            # --- 🔍 ค้นหาชื่อยาจริงด้วย Property ทุกทางเลือกที่มีในไฟล์ ---
            drug_name = None
            possible_props = ["IUPAC Name", "Compound Name", "Preferred Name", "Synonyms", "OpenEye OEToolkits Title"]
            
            for prop in possible_props:
                if mol.HasProp(prop):
                    drug_name = mol.GetProp(prop).split('\n')[0]  # เอาชื่อแรกสุด
                    break
            
            # ถ้าคุ้ยทุกช่องแล้วยังไม่เจอ ให้ลองดึงประวัติทั่วไป หรือใช้ชื่อที่มีสัญลักษณ์เคมีกำกับ
            if not drug_name:
                drug_name = f"Bioactive Compound {pubchem_id}"
            
            # ตัดให้สั้นกระชับไม่ล้น Dropdown
            display_name = (drug_name[:30] + '...') if len(drug_name) > 30 else drug_name
            
            # --- 🧪 คำนวณสูตรและแปลงเป็นรูปภาพพันธะเคมี 2D ---
            smiles = Chem.MolToSmiles(mol)
            formula = rdMolDescriptors.CalcMolFormula(mol)
            
            # สั่งให้ RDKit วาดรูปพันธะเคมีแล้วเซฟลงโฟลเดอร์ด่วน
            image_path = os.path.join(image_folder, f"{cid_key}.png")
            try:
                Draw.MolToFile(mol, image_path, size=(300, 300))
            except:
                # กรณีโมเลกุลมีโครงสร้างแปลกปลอมจนวาดไม่ได้ ให้ข้ามการวาดรูปไปก่อน
                pass
            
            data_list.append({
                "PubChem_CID": cid_key,
                "Drug_Name": display_name,
                "Chemical_Formula": formula,
                "SMILES_Structure": smiles,
                "Image_Path": image_path
            })
            
            count += 1
            if count >= max_records:
                break

    df = pd.DataFrame(data_list)
    df.to_csv("pubchem_sample.csv", index=False)
    print(f"✅ สำเร็จ! อัปเดตตารางและสร้างรูปโครงสร้างพันธะเคมีจำนวน {count} รูปในโฟลเดอร์ '{image_folder}' เรียบร้อยแล้วครับ")

except Exception as e:
    print(f"❌ เกิดข้อผิดพลาดในสคริปต์สกัด: {str(e)}")
