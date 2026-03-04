import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. โหลดไฟล์ผลลัพธ์
df = pd.read_csv("Results_Qwen2VL.csv")

# 2. ทำความสะอาดข้อมูล (ลบช่องว่าง และแปลงเป็นพิมพ์ใหญ่ ป้องกัน AI ตอบ a, b, c)
y_true = df['Correct Answer'].astype(str).str.strip().str.upper().tolist()
y_pred = df['Predicted Answer'].astype(str).str.strip().str.upper().tolist()

# กรองเอาเฉพาะ Label ที่ควรจะเป็น (A, B, C, D) ป้องกันข้อความขยะ
valid_labels = ['A', 'B', 'C', 'D']
labels = sorted([l for l in set(y_true + y_pred) if l in valid_labels])

# 3. คำนวณค่าต่างๆ
accuracy = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
cm = confusion_matrix(y_true, y_pred, labels=labels)

# 4. สร้างข้อความรายงานผล
output_text = "=" * 60 + "\n"
output_text += f"🎯 OVERALL ACCURACY (ความแม่นยำรวม): {accuracy * 100:.2f}%\n"
output_text += "=" * 60 + "\n"
output_text += "📊 CLASSIFICATION REPORT (รายงานแยกตามตัวเลือก):\n"
output_text += "-" * 60 + "\n"
output_text += report + "\n"
output_text += "=" * 60 + "\n"

# 🌟 พิมพ์ลง Terminal และ **บันทึกเป็นไฟล์ Text** 🌟
print(output_text)
with open("accuracy_report.txt", "w", encoding="utf-8") as f:
    f.write(output_text)

# 5. วาด Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=labels, yticklabels=labels, 
            annot_kws={"size": 14})
plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Predicted Answer', fontsize=12, labelpad=10)
plt.ylabel('Correct Answer', fontsize=12, labelpad=10)

plt.savefig('confusion_matrix.png', bbox_inches='tight', dpi=300)

print("✅ สร้างรายงานเสร็จสมบูรณ์!")
print("1. เปิดดูผลตัวเลขได้ที่ไฟล์: accuracy_report.txt")
print("2. เปิดดูกราฟได้ที่ไฟล์: confusion_matrix.png")

















# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# # 1. โหลดไฟล์ผลลัพธ์
# df = pd.read_csv("Results_Qwen2VL.csv")

# # 2. ทำความสะอาดข้อมูล (ลบช่องว่างส่วนเกินเผื่อไว้) และแปลงเป็น String
# y_true = df['Correct Answer'].astype(str).str.strip().tolist()
# y_pred = df['Predicted Answer'].astype(str).str.strip().tolist()

# # หาตัวเลือกทั้งหมดที่มี (เช่น A, B, C, D) แล้วเอามาเรียงลำดับ
# labels = sorted(list(set(y_true + y_pred)))

# # 3. ให้ sklearn คำนวณค่าต่างๆ
# accuracy = accuracy_score(y_true, y_pred)
# report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
# cm = confusion_matrix(y_true, y_pred, labels=labels)

# # ----------------------------------------------------
# # 📌 ส่วนที่ 4: พิมพ์ผลลัพธ์แบบ Text ให้ดูง่าย
# # ----------------------------------------------------
# print("=" * 60)
# print(f"🎯 OVERALL ACCURACY (ความแม่นยำรวม): {accuracy * 100:.2f}%")
# print("=" * 60)
# print("\n📊 CLASSIFICATION REPORT (รายงานแยกตามตัวเลือก):")
# print("-" * 60)
# print(report)
# print("=" * 60)

# # ----------------------------------------------------
# # 📌 ส่วนที่ 5: วาด Confusion Matrix ให้ออกมาเป็นรูปภาพสีสวยงาม
# # ----------------------------------------------------
# plt.figure(figsize=(8, 6))

# # ใช้ไลบรารี seaborn วาดตาราง Heatmap
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
#             xticklabels=labels, yticklabels=labels, 
#             annot_kws={"size": 14}) # ขยายขนาดตัวเลข

# plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=15)
# plt.xlabel('Predicted Answer', fontsize=12, labelpad=10)
# plt.ylabel('Correct Answer', fontsize=12, labelpad=10)

# # บันทึกเป็นรูปภาพและแสดงผล
# plt.savefig('confusion_matrix.png', bbox_inches='tight', dpi=300)
# plt.show()

# print("\n กราฟ Confusion Matrix เสร็จแล้ว! (รูปภาพถูกบันทึกเป็นไฟล์ confusion_matrix.png)")


















# import pandas as pd
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix

# # 1. โหลดไฟล์ผลลัพธ์
# df = pd.read_csv("Results_Qwen2VL.csv")

# # 2. ดึงคอลัมน์เฉลย และ คำตอบของ AI ออกมา
# y_true = df['Correct Answer'].tolist()  # เฉลย
# y_pred = df['Predicted Answer'].tolist()  # AI ตอบ

# # 3. ให้ sklearn คำนวณความแม่นยำ!
# accuracy = accuracy_score(y_true, y_pred)

# print(f"accuracy: {accuracy * 100:.2f}%")
# print(classification_report(y_true, y_pred))
# print(confusion_matrix(y_true, y_pred))
