"""
สคริปต์สำหรับสร้าง subset annotation JSON จากไฟล์เต็ม
ใช้สำหรับทดสอบความสมบูรณ์ของการทำงานของโค้ด
"""
import json
import argparse
from pathlib import Path
from typing import Dict, Any


def create_subset(
    input_json: str,
    output_json: str,
    items_per_category: int = 2,
    categories: list = None
) -> None:
    """
    สร้าง subset จากไฟล์ JSON เต็ม
    
    Args:
        input_json: Path ไปยังไฟล์ annotation เต็ม
        output_json: Path ไปยังไฟล์ output
        items_per_category: จำนวนรูปต่อคลาส
        categories: List ของคลาสที่ต้องการ (None = ทั้งหมด)
    """
    
    with open(input_json, 'r', encoding='utf-8') as f:
        full_data = json.load(f)
    
    subset_data = {}
    category_count = {}
    
    print(f"📖 อ่านไฟล์: {input_json}")
    print(f"📊 ทั้งหมด: {len(full_data)} รูป")
    
    for img_path, item_value in full_data.items():
        # ดึงชื่อคลาส (category) จาก path
        # เช่น "DS-MVTec/bottle/image/..." → "bottle"
        parts = img_path.split('/')
        if len(parts) >= 2:
            category = parts[1]
        else:
            continue
        
        # ถ้ากำหนด categories ให้เฉพาะบางคลาส
        if categories and category not in categories:
            continue
        
        # นับจำนวนรูปต่อคลาส
        if category not in category_count:
            category_count[category] = 0
        
        if category_count[category] < items_per_category:
            subset_data[img_path] = item_value
            category_count[category] += 1
    
    # บันทึก subset ใหม่
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(subset_data, f, indent=4, ensure_ascii=False)
    
    print(f"\n✅ สร้างเสร็จ: {output_json}")
    print(f"📈 ขนาด: {len(subset_data)} รูป")
    print("\n📋 จำนวนรูปต่อคลาส:")
    for cat, count in sorted(category_count.items()):
        print(f"  - {cat}: {count}")


def create_reference_subset(
    full_reference_file: str,
    output_reference_file: str,
    max_images: int = 50
) -> None:
    """
    สร้าง subset reference images list
    
    Args:
        full_reference_file: Path ไปยังไฟล์ reference เต็ม
        output_reference_file: Path ไปยังไฟล์ output
        max_images: จำนวนรูปสูงสุด
    """
    
    with open(full_reference_file, 'r') as f:
        all_refs = [line.strip() for line in f if line.strip()]
    
    # เลือก reference จาก bottle (first category)
    bottle_refs = [ref for ref in all_refs if 'bottle' in ref]
    subset_refs = bottle_refs[:min(max_images, len(bottle_refs))]
    
    with open(output_reference_file, 'w') as f:
        for ref in subset_refs:
            f.write(ref + '\n')
    
    print(f"\n✅ สร้างเสร็จ: {output_reference_file}")
    print(f"📈 จำนวนรูป: {len(subset_refs)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="สร้าง subset annotation JSON")
    parser.add_argument(
        "--input",
        type=str,
        default=r"D:\AI_Projects\MoXpert\Annotation\DS-MVTec.json",
        help="Path ไปยังไฟล์ annotation เต็ม"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=r"D:\AI_Projects\MoXpert\Annotation\DS-MVTec-small.json",
        help="Path ไปยังไฟล์ output"
    )
    parser.add_argument(
        "--items",
        type=int,
        default=2,
        help="จำนวนรูปต่อคลาส"
    )
    parser.add_argument(
        "--categories",
        type=str,
        default="bottle",
        help="คลาสที่ต้องการ (comma-separated) หรือ 'all'"
    )
    parser.add_argument(
        "--reference-input",
        type=str,
        default=r"D:\AI_Projects\MoXpert\Memory\reference_image_locations.txt",
        help="Path ไปยังไฟล์ reference เต็ม"
    )
    parser.add_argument(
        "--reference-output",
        type=str,
        default=r"D:\AI_Projects\MoXpert\Memory\reference_image_locations_small.txt",
        help="Path ไปยังไฟล์ reference output"
    )
    parser.add_argument(
        "--max-ref",
        type=int,
        default=50,
        help="จำนวนรูป reference สูงสุด"
    )
    
    args = parser.parse_args()
    
    # Parse categories
    if args.categories.lower() == "all":
        categories = None
    else:
        categories = [cat.strip() for cat in args.categories.split(",")]
    
    # สร้าง annotation subset
    create_subset(
        args.input,
        args.output,
        args.items,
        categories
    )
    
    # สร้าง reference subset
    create_reference_subset(
        args.reference_input,
        args.reference_output,
        args.max_ref
    )
    
    print("\n" + "="*50)
    print("⚙️ การตั้งค่า CONFIG ใน Qwen2-VL.py:")
    print("="*50)
    print(f"""
# สำหรับทดสอบกับ subset:
CONFIG = {{
    "annotation_file": r"{args.output}",
    "reference_images": r"{args.reference_output}",
    ...
}}

# สำหรับใช้ full dataset:
CONFIG = {{
    "annotation_file": r"{args.input}",
    "reference_images": r"{args.reference_input}",
    ...
}}
""")
