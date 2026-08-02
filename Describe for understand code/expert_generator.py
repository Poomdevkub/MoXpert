def expert_generator(image1, image2, question_type, question, options_text, domain_knowledge):

    descriptions_text = ""
    if isinstance(domain_knowledge, dict):
        descriptions_text = "\n".join([f"{key.capitalize()}: {value}" for key, value in domain_knowledge.items()])
        # descriptions_text = ข้อมูลที่เตรียมไว้ให้ Knowledge Guide ใช้ (ยังไม่ activate จนกว่าจะถูกแทรกใน prompt ด้านล่าง)
    else:
        descriptions_text = domain_knowledge

    object_name = domain_knowledge['object_name']  # ไม่ใช่ expert ใด ๆ, ไม่ได้ถูกใช้ต่อ

    if question_type == "Anomaly Detection":
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image1},   # ← Reference Extractor
                {"type": "image", "image": image2},    # ← query image (ไม่ใช่ expert)
                {"type": "text", "text": f"The first image is a normal sample... Question: {question}\nOptions:\n{options_text}..."},
                # ← ข้อความอธิบาย Reference Extractor (ไม่มี Knowledge Guide, ไม่มี Reasoning Expert)
            ],
        }]

    elif question_type == "Defect Classification":
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image1},   # ← Reference Extractor
                {"type": "image", "image": image2},    # ← query image
                {"type": "text", "text": (
                    f"Question: {question}\nOptions:\n{options_text}\n"
                    "The first image is a normal reference sample..."   # ← ส่วนอธิบาย Reference Extractor
                    f"Following is the domain knowledge...:\n{descriptions_text}\n"   # ← Knowledge Guide
                    "Please respond with the letter of the correct option only."
                )},
            ],
        }]

    elif question_type == "Defect Localization":
        messages = [{
            "role": "user",
            "content": [
                #{"type": "image", "image": image1},  # ← Reference Extractor (ปิด/comment ออก)
                {"type": "image", "image": image2},    # ← query image
                {"type": "text", "text": f"\nQuestion: {question}\nOptions:\n{options_text}..."},
                # ← ไม่มี Knowledge Guide, ไม่มี Reasoning Expert
            ],
        }]

    elif question_type == "Defect Description":
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image1},   # ← Reference Extractor
                {"type": "image", "image": image2},    # ← query image
                {"type": "text", "text": 
                    "Following is the domain knowledge...\n"
                    f"Domain Knowledge:\n{descriptions_text}\n"    # ← Knowledge Guide
                    "The first image is a normal reference sample..."   # ← ส่วนอธิบาย Reference Extractor
                    "Let's approach this systematically:\n"
                    "1. **Observe**...\n2. **Compare**...\n3. **Decide**...\n"   # ← Reasoning Expert (CoT)
                    f"Question: {question}\nOptions:\n{options_text}..."
                },
            ],
        }]

    elif question_type == "Defect Analysis":
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image1},   # ← Reference Extractor
                {"type": "image", "image": image2},    # ← query image
                {"type": "text", "text": 
                    "The first image is a normal reference sample..."   # ← ส่วนอธิบาย Reference Extractor
                    "Let's approach this systematically:\n"
                    "1. **Observe**...\n2. **Compare**...\n3. **Decide**...\n"   # ← Reasoning Expert (CoT)
                    # ไม่มี descriptions_text ส่วนนี้ → Knowledge Guide ปิด
                    f"Question: {question}\nOptions:\n{options_text}..."
                },
            ],
        }]

    else:
        raise ValueError(...)   # ไม่ใช่ expert, เป็น error handling

    return messages
    # messages ทั้งชุด → ส่งเข้า Qwen2-VL.generate() ใน Qwen2-VL.py = Decision Maker (รันเสมอทุก question_type)