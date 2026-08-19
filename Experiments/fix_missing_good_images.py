"""Fill missing DS-MVTec '.../image/good/*.png' files by copying the matching
image from MVTec-AD '{category}/test/good/*.png'. DefectSpectrum only published
a subset of 'good' images since they need no defect mask, but the MMAD
annotation file expects the full MVTec-AD test/good set."""


'''
เติมไฟล์รูปภาพที่ขาดหายไปใน DS-MVTec ที่อยู่ในโฟลเดอร์ .../image/good/*.png โดยคัดลอกไฟล์รูปภาพที่มีชื่อเดียวกันจากชุดข้อมูล MVTec-AD ที่อยู่ใน 
{category}/test/good/*.png เนื่องจาก DefectSpectrum เผยแพร่รูปภาพประเภท good (ไม่มีตำหนิ) มาเพียงบางส่วนเท่านั้น เพราะรูปภาพเหล่านี้ 
ไม่จำเป็นต้องมี Defect Mask แต่ไฟล์ MMAD annotation 
กลับคาดหวังให้มีรูปภาพ good ครบทุกภาพเหมือนกับที่อยู่ในชุดข้อมูล MVTec-AD ภายใต้โฟลเดอร์ test/good
'''

import json
import os
import shutil

ANNOTATION_FILE = r"../Annotation/DS-MVTec.json"
DATASET_ROOT = r"../Dataset/MMAD"

with open(ANNOTATION_FILE, "r") as f:
    data = json.load(f)

copied, already_ok, unresolved = 0, 0, []

for key in data:
    if "/image/good/" not in key:
        continue

    parts = key.split("/")
    category = parts[1]
    filename = parts[-1]

    dst = os.path.join(DATASET_ROOT, "DS-MVTec", category, "image", "good", filename)
    if os.path.exists(dst):
        already_ok += 1
        continue

    src = os.path.join(DATASET_ROOT, "MVTec-AD", category, "test", "good", filename)
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    else:
        unresolved.append(key)

print(f"Already present: {already_ok}")
print(f"Copied from MVTec-AD: {copied}")
print(f"Still missing (no source found): {len(unresolved)}")
for k in unresolved[:20]:
    print("  -", k)


