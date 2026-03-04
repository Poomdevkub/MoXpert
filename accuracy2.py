import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. โหลดไฟล์ผลลัพธ์
df = pd.read_csv("Results_Qwen2VL.csv")

# 2. ทำความสะอาดข้อมูล
df['Predicted Answer'] = df['Predicted Answer'].astype(str).str.strip().str.upper()
df['Correct Answer'] = df['Correct Answer'].astype(str).str.strip().str.upper()

# เช็กความถูกต้อง
df['Correct'] = df['Predicted Answer'] == df['Correct Answer']

# 3. สร้างฟังก์ชันจัดกลุ่มคำถามทั้ง 5 หมวด (ตามมาตรฐาน MVTec-AD / MMAD)
def categorize_question(q):
    q_lower = q.lower()
    # Analysis (ผลกระทบ สาเหตุ)
    if any(kw in q_lower for kw in ['effect', 'affect', 'cause', 'impact', 'indicate', 'functionality', 'structural integrity']):
        return 'Analysis'
    # Classification (ประเภทของรอยตำหนิ)
    elif 'type' in q_lower:
        return 'Classification'
    # Localization (ตำแหน่ง)
    elif any(kw in q_lower for kw in ['where', 'which part', 'which half', 'positioned', 'located']):
        return 'Localization'
    # Discrimination (การแยกแยะ/มีหรือไม่มีตำหนิ)
    elif any(kw in q_lower for kw in ['is there', 'are there', 'do the', 'does the']):
        if 'how many' in q_lower: # กันข้อที่ถามจำนวนหลุดมา
            return 'Description'
        return 'Discrimination'
    # Description (ลักษณะที่ปรากฏ สี ขนาด จำนวน)
    else:
        return 'Description'

# นำฟังก์ชันไปจัดกลุ่มให้ข้อสอบแต่ละข้อ
df['Dimension'] = df['Question'].apply(categorize_question)

# 4. คำนวณค่าเฉลี่ยในแต่ละหมวดหมู่
acc_by_dim = df.groupby('Dimension')['Correct'].mean() * 100
dim_acc = {
    'Discrimination': acc_by_dim.get('Discrimination', 0),
    'Classification': acc_by_dim.get('Classification', 0),
    'Localization': acc_by_dim.get('Localization', 0),
    'Description': acc_by_dim.get('Description', 0),
    'Analysis': acc_by_dim.get('Analysis', 0),
}
# หาค่าเฉลี่ยรวม 5 หมวด (MVTec-AD Average)
avg_acc = sum(dim_acc.values()) / 5

# 5. ข้อมูล Baseline โมเดลอื่นๆ สำหรับเปรียบเทียบ
baselines = [
    ["Random Chance", "-", "50.00", "25.00", "25.00", "25.00", "25.00", "30.00"],
    ["GPT-4o", "-", "77.52", "84.94", "88.55", "92.17", "95.18", "87.67"],
    ["GPT-4o-mini", "-", "74.22", "71.78", "62.62", "79.72", "90.62", "75.79"],
    ["Gemini-2-flash", "-", "83.80", "72.37", "73.18", "77.82", "90.37", "79.49"],
    ["Gemini-2-flash-lite", "-", "82.79", "70.87", "69.57", "77.25", "89.46", "78.02"],
    ["AnomalyGPT", "7B", "82.84", "27.80", "28.33", "34.62", "34.36", "54.78"],
    ["InternVL2", "4B", "70.96", "44.81", "66.97", "59.52", "87.39", "65.93"],
    ["InternVL2", "8B", "76.88", "51.04", "59.18", "64.55", "85.73", "67.48"],
    ["MiniCPM-V2.6", "8B", "72.50", "64.07", "68.65", "79.55", "90.04", "74.96"],
    ["LLaVA-NeXT", "7B", "78.42", "45.23", "64.96", "68.18", "87.14", "68.79"],
    ["LLaVA-OneVision", "7B", "94.09", "79.59", "78.12", "83.18", "91.29", "85.25"],
    ["Qwen2-VL (Official)", "2B", "73.21", "60.08", "65.46", "74.86", "88.63", "72.45"],
]

# เพิ่มผลลัพธ์จากโมเดลของเราเข้าไปเป็นบรรทัดสุดท้าย
our_row = [
    "Ours (Qwen2-VL)", "2B", 
    f"{dim_acc['Discrimination']:.2f}", 
    f"{dim_acc['Classification']:.2f}", 
    f"{dim_acc['Localization']:.2f}", 
    f"{dim_acc['Description']:.2f}", 
    f"{dim_acc['Analysis']:.2f}",
    f"{avg_acc:.2f}"
]
baselines.append(our_row)

# 6. สร้างรายงานแบบ Text (เพื่อเซฟลงไฟล์)
report_text = "=" * 105 + "\n"
report_text += "🏆 MODEL COMPARISON REPORT (VQA on MVTec-AD)\n"
report_text += "=" * 105 + "\n"
# จัดตารางให้สวยงาม
header = f"{'Model':<20} | {'Scale':<5} | {'Discrimination':<14} | {'Classification':<14} | {'Localization':<12} | {'Description':<11} | {'Analysis':<8} | {'MVTec-AD Avg':<12}"
report_text += header + "\n"
report_text += "-" * 105 + "\n"

for row in baselines:
    report_text += f"{row[0]:<20} | {row[1]:<5} | {row[2]:<14} | {row[3]:<14} | {row[4]:<12} | {row[5]:<11} | {row[6]:<8} | {row[7]:<12}\n"

report_text += "=" * 105 + "\n"

# 7. เซฟและพิมพ์ผลลัพธ์
print(report_text)
with open("accuracy_report.txt", "w", encoding="utf-8") as f:
    f.write(report_text)

print("\n✅ รายงานเปรียบเทียบโมเดลเสร็จสมบูรณ์! (เซฟลงไฟล์ accuracy_report2.txt เรียบร้อยแล้ว)") 
print("👉 คุณสามารถเปิดไฟล์ accuracy_report2.txt")