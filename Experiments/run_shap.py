# ==========================================
# ไฟล์: Experiments/run_shap.py (รันบน Local Windows)
# ==========================================
import os
import json
import torch
import numpy as np
import shap
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F

import clip
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from expert_generator import expert_generator

# 1. ตั้งค่า Config 
CONFIG = {
    "seed": 123,
    "device": "cuda",
    "clip_model": "ViT-B/16",
    "qwen_path": "Qwen/Qwen2-VL-2B-Instruct", 
    "reference_index": r"D:\AI_Projects\MoXpert\Memory\memory.index",
    "reference_images": r"D:\AI_Projects\MoXpert\Memory\reference_image_locations.txt",
    "annotation_file": r"D:\AI_Projects\MoXpert\Annotation\DS-MVTec.json",
    "domain_knowledge": r"D:\AI_Projects\MoXpert\Knowledge Guide\domain_knowledge_detection.json",
    "results_csv": r"Results_Qwen2VL.csv"
}

# ---------------------------------------------------------
# สเตป 2: ฟังก์ชันสำหรับดึงค่าความน่าจะเป็นของ A, B, C, D
# ---------------------------------------------------------
def get_answer_probabilities(model, inputs, processor):
    tokenizer = processor.tokenizer
    target_ids = [
        tokenizer.convert_tokens_to_ids("A"),
        tokenizer.convert_tokens_to_ids("B"),
        tokenizer.convert_tokens_to_ids("C"),
        tokenizer.convert_tokens_to_ids("D")
    ]
    
    with torch.no_grad():
        outputs = model(**inputs, output_logits=True, max_new_tokens=1)
        first_token_logits = outputs.logits[0, -1, :] 
        
        abcd_logits = first_token_logits[target_ids]
        abcd_probs = F.softmax(abcd_logits, dim=-1).float().cpu().numpy()
        
    return abcd_probs

# ---------------------------------------------------------
# สเตป 3: ฟังก์ชัน Wrapper ให้ SHAP โยนรูปเข้ามาประมวลผลได้
# ---------------------------------------------------------
def qwen_predict_wrapper(images):
    batch_probs = []
    
    for img_array in images:
        img_pil = Image.fromarray(img_array.astype(np.uint8))
        
        # ⚠️ หมายเหตุ: ฟังก์ชัน expert_generator อาจจะต้องรองรับการรับ PIL Image ตรงๆ 
        messages = expert_generator(
            reference_image_path, 
            img_pil, 
            question_type, 
            question, 
            options_text, 
            domain_knowledge
        )
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        
        inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to(CONFIG["device"])
        
        probs = get_answer_probabilities(model, inputs, processor)
        batch_probs.append(probs)
        
    return np.array(batch_probs)

# ---------------------------------------------------------
# สเตป 4: ส่วนรันหลัก (Main Execution)
# ---------------------------------------------------------
if __name__ == "__main__":
    print("Loading models... (ใช้เวลาสักครู่)")
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        CONFIG["qwen_path"], torch_dtype=torch.bfloat16, attn_implementation="sdpa", trust_remote_code=True
    ).to(CONFIG["device"])
    processor = AutoProcessor.from_pretrained(CONFIG["qwen_path"], trust_remote_code=True, use_fast=False)

    # 📌 4.1 กำหนด "ข้อสอบ 1 ข้อ" ที่อยากเอามาทำ SHAP
    # [!] Path โฟลเดอร์ Dataset 
    query_image_path = r"D:\AI_Projects\MoXpert\Dataset\MMAD\MVTec-AD\bottle\test\broken_large\000.png"
    reference_image_path = r"D:\AI_Projects\MoXpert\Dataset\MMAD\MVTec-AD\bottle\train\good\000.png"
    
    question = "Is there any defect in the object?"
    question_type = "Anomaly Detection"
    options_text = "A: Yes, there is a large crack.\nB: Yes, there is a scratch.\nC: Yes, there is a stain.\nD: No, it is normal."
    
    # จำลอง Domain Knowledge ของขวด
    domain_knowledge = {"object_name": "bottle", "descriptions": "A standard glass bottle. Defects include broken parts, contamination, and structural damage."}

    # 📌 4.2 โหลดภาพออริจินัล
    if not os.path.exists(query_image_path):
        print(f"❌ หาไฟล์ภาพทดสอบไม่เจอ: {query_image_path}")
        print("กรุณาแก้ไขตัวแปร query_image_path ให้ชี้ไปยังไฟล์รูปภาพที่มีอยู่จริง")
        exit()

    original_image = np.array(Image.open(query_image_path).convert("RGB"))

    print("เริ่มกระบวนการ SHAP Explainer...")
    masker = shap.maskers.Image("inpaint_telea", original_image.shape)
    
    explainer = shap.Explainer(qwen_predict_wrapper, masker, output_names=["A", "B", "C", "D"])

    # 📌 4.3 สั่งคำนวณ SHAP (max_evals=50 เพื่อทดสอบความเร็วก่อน)
    print("กำลังประมวลผล SHAP (อาจใช้เวลา 1-3 นาทีขึ้นอยู่กับการ์ดจอ)...")
    shap_values = explainer(np.expand_dims(original_image, axis=0), max_evals=50, batch_size=1)

    # 📌 4.4 บันทึกรูป Heatmap ออกมา
    shap.image_plot(shap_values, show=False)
    
    # บันทึกไฟล์ลงในโฟลเดอร์ MoXpert
    output_img_path = r"D:\AI_Projects\MoXpert\shap_result_bottle.png"
    plt.savefig(output_img_path, bbox_inches='tight', dpi=300)
    print(f"✅ รัน SHAP เสร็จสิ้น! บันทึกรูปภาพไว้ที่: {output_img_path}")