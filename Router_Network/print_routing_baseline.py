#!/usr/bin/env python3
"""print ค่าที่ได้จากขั้นตอนเลือก expert แบบ BASELINE (MoXpert เดิม)

baseline ใช้ `expert_generator()` ที่ผูกกับ `question_type` แบบ hardcode (if/elif)
สคริปต์นี้เรียกโค้ดเดิมจากข้างนอก (ไม่แก้ไฟล์ต้นฉบับ) แล้ว print ค่าที่ได้ออกมาต่อคำถาม

*** จุดที่ต้องสังเกต ***
baseline ไม่มี "ค่าความน่าจะเป็น" ให้ print เลย มีแค่เวกเตอร์ 0/1 ที่ได้จากการ lookup
ด้วย question_type ต่างจากไฟล์คู่กัน (print_routing_router.py) ที่มีค่า p ต่อเนื่อง 0-1

รัน:
    python Router_Network/print_routing_baseline.py --limit 300
    python Router_Network/print_routing_baseline.py --limit 5 --show-prompt
    python Router_Network/print_routing_baseline.py --limit 50 --no-retrieval   # เร็ว ไม่ใช้ CLIP
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # กันชน libomp บน macOS

_THIS_DIR = Path(__file__).resolve().parent
REPO = _THIS_DIR.parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(REPO / "Experiments"))

from expert_generator import expert_generator                      # noqa: E402  baseline (ห้ามแก้)
from analyze_original_expert_generator import (                    # noqa: E402  ตัว detect ที่ทดสอบแล้ว
    EXPERT_NAMES, SHORT, detect_experts, split_messages,
)
from Qwen2VL_router import (                                       # noqa: E402  helper เดิม
    find_descriptions, resolve_reference, select_device,
)

PLACEHOLDER_REF = "__PLACEHOLDER_REFERENCE_IMAGE__"

md_lines: list[str] = []


def emit(line: str = "") -> None:
    print(line)
    md_lines.append(line)


def descriptions_text_of(dk: dict) -> str:
    """สร้างสตริง domain knowledge แบบเดียวกับที่ expert_generator สร้างเป๊ะ ๆ

    ตรงกับ expert_generator.py:7 -> "\\n".join(f"{key.capitalize()}: {value}" ...)
    ใช้เป็น marker ตรวจว่า Knowledge Guide ถูกเปิดไหม โดยไม่ต้องฉีด sentinel เข้าข้อมูลจริง
    """
    return "\n".join(f"{k.capitalize()}: {v}" for k, v in dk.items())


def build_retrieval(device: str):
    """โหลด CLIP + FAISS index สำหรับหา reference image (เหมือน baseline ตัวจริง)"""
    import faiss
    import platform
    import numpy as np
    import moxpert_lite as ml

    if platform.system() == "Darwin":
        faiss.omp_set_num_threads(1)
    print("[โหลด] CLIP encoder ...")
    encoder = ml.MoXpertEncoder(clip_model_name="ViT-B/16", device=device)
    print("[โหลด] FAISS index ...")
    index_img = faiss.read_index(str(REPO / "Memory" / "memory.index"))
    ref_paths = [l.strip() for l in open(REPO / "Memory" / "reference_image_locations.txt") if l.strip()]

    def retrieve(query_image_path: str) -> str:
        raw = encoder.encode_image_raw(query_image_path)
        _, idx = index_img.search(np.expand_dims(raw.astype(np.float32), 0), 1)
        return resolve_reference(ref_paths[int(idx[0][0])], REPO)

    return retrieve


def main():
    ap = argparse.ArgumentParser(description="print ค่าจากขั้นตอน routing แบบ baseline")
    ap.add_argument("--limit", type=int, default=300,
                    help="จำนวนรูปที่จะประมวลผล (default 300 = เท่ากับตอนสร้าง features_real.npz)")
    ap.add_argument("--show-prompt", action="store_true", help="โชว์ prompt เต็มที่ expert_generator สร้าง")
    ap.add_argument("--no-retrieval", action="store_true",
                    help="ไม่ใช้ CLIP+FAISS ใช้ path placeholder แทน (เร็วกว่า ผลเวกเตอร์เท่ากัน)")
    ap.add_argument("--outdir", default=str(REPO / "Router_Network" / "artifacts"))
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = json.load(open(REPO / "Annotation" / "DS-MVTec.json"))
    domain_data = json.load(open(REPO / "Knowledge Guide" / "domain_knowledge_detection.json",
                                 encoding="utf-8"))
    data_root = REPO / "Dataset" / "MMAD"

    retrieve = None
    if not args.no_retrieval:
        retrieve = build_retrieval(select_device())

    emit("# ค่าจากขั้นตอน routing — แบบ BASELINE (`expert_generator` เดิม)")
    emit()
    emit(f"สร้างเมื่อ {ts} | `--limit {args.limit}` | "
         f"retrieval: {'placeholder' if args.no_retrieval else 'CLIP + FAISS จริง'}")
    emit()
    emit("กลไก: `if/elif question_type == ...` แบบ hardcode ที่ "
         "[expert_generator.py:14-105](Experiments/expert_generator.py#L14) "
         "— **ไม่มีค่าความน่าจะเป็น ไม่มี threshold**")
    emit()
    emit("---")
    emit()
    emit("## ค่าที่ได้ต่อคำถาม")
    emit()
    emit("| # | question_type | เวกเตอร์ | expert ที่เปิด | #รูป | image |")
    emit("|--:|---|:--:|---|:--:|---|")

    rows = []
    vec_by_type: dict[str, set[str]] = defaultdict(set)
    type_counter: Counter = Counter()
    expert_on: Counter = Counter()
    unsupported: Counter = Counter()
    warn_same_ref = 0
    n_img = 0
    n_q = 0

    for img_key, item in data.items():
        if n_img >= args.limit:
            break
        query_image_path = str(data_root / img_key)
        if not os.path.exists(query_image_path):      # ข้ามรูปที่ไฟล์หาย
            continue
        n_img += 1

        if retrieve is not None:
            ref_path = retrieve(query_image_path)
            if os.path.abspath(ref_path) == os.path.abspath(query_image_path):
                # FAISS คืนรูป query เอง -> detect Reference จะกำกวม ต้องเตือน ไม่ใช่เงียบ
                warn_same_ref += 1
                print(f"[เตือน] reference = query image ({img_key}) -> ข้ามคำถามของรูปนี้")
                continue
        else:
            ref_path = PLACEHOLDER_REF

        dk = find_descriptions(domain_data, img_key)
        dk_marker = descriptions_text_of(dk)

        for conv in item["conversation"]:
            qtype = conv["type"]
            question = conv["Question"]
            options_text = "\n".join(f"{k}: {v}" for k, v in conv["Options"].items())

            try:
                messages = expert_generator(ref_path, query_image_path, qtype,
                                            question, options_text, dk)
            except ValueError:
                unsupported[qtype] += 1     # question_type ที่โค้ดเดิมไม่รองรับ
                continue

            idx = n_q          # ดัชนีแถวฐาน 0 ให้ตรงกับ print_routing_router.py (join กันได้)
            n_q += 1           # ตัวนับจำนวนคำถามทั้งหมด
            act = detect_experts(messages, ref_marker=ref_path, dk_marker=dk_marker)
            images, text = split_messages(messages)
            vec = "".join(str(act[n]) for n in EXPERT_NAMES)
            active = [n for n in EXPERT_NAMES if act[n] == 1]

            vec_by_type[qtype].add(vec)
            type_counter[qtype] += 1
            for n in active:
                expert_on[n] += 1

            if idx < 40:       # จำกัดความยาวตาราง ไม่ให้รายงานบวม
                emit(f"| {idx} | {qtype} | `{vec}` | {', '.join(SHORT[n] for n in active)} | "
                     f"{len(images)} | `{img_key.split('/')[-1]}` |")

            if args.show_prompt and idx < 5:
                emit()
                emit(f"<details><summary>prompt เต็มของแถวที่ {idx}</summary>")
                emit()
                emit("```")
                emit(f"images: {images}")
                emit(text)
                emit("```")
                emit()
                emit("</details>")
                emit()

            rows.append({
                "idx": idx, "image_key": img_key, "question_type": qtype,
                "question": question, "vector": vec,
                **{f"p_{SHORT[n]}": "" for n in EXPERT_NAMES},        # baseline ไม่มีค่า p
                **{f"y_{SHORT[n]}": act[n] for n in EXPERT_NAMES},
                "active_experts": "|".join(active), "n_images": len(images),
            })

    if n_q > 40:
        emit(f"| ... | *(แสดง 40 แถวแรกจาก {n_q} แถว — ดูครบใน CSV)* | | | | |")
    emit()

    # ---------------- สรุป ----------------
    emit("---")
    emit()
    emit("## สรุป")
    emit()
    emit(f"ประมวลผล **{n_img} รูป / {n_q} คำถาม**")
    if warn_same_ref:
        emit(f"> ข้าม {warn_same_ref} รูป เพราะ FAISS คืน reference = รูป query เอง (detect กำกวม)")
    emit()

    emit("### ตาราง lookup ทั้งหมด — \"router\" ของ baseline อยู่ในตารางนี้ทั้งตัว")
    emit()
    emit("| question_type | เวกเตอร์ที่เป็นไปได้ | expert ที่เปิด | จำนวนคำถาม |")
    emit("|---|:--:|---|--:|")
    for qtype in sorted(vec_by_type):
        vs = sorted(vec_by_type[qtype])
        names = ", ".join(SHORT[n] for n, c in zip(EXPERT_NAMES, vs[0]) if c == "1")
        flag = "" if len(vs) == 1 else "  ⚠ มากกว่า 1 แบบ"
        emit(f"| {qtype} | {', '.join(f'`{v}`' for v in vs)}{flag} | {names} | {type_counter[qtype]} |")
    emit()
    if unsupported:
        for qtype, c in unsupported.items():
            emit(f"> `{qtype}` — โค้ดเดิมไม่รองรับ (`raise ValueError`) ข้ามไป {c} คำถาม")
        emit()

    emit("### อัตราการเปิดของแต่ละ expert")
    emit()
    emit("| expert | เปิดกี่ครั้ง | อัตรา |")
    emit("|---|--:|--:|")
    for n in EXPERT_NAMES:
        pct = expert_on[n] / n_q * 100 if n_q else 0
        emit(f"| {n} | {expert_on[n]} | {pct:.1f}% |")
    emit()

    all_vecs = {v for s in vec_by_type.values() for v in s}
    emit(f"### จำนวนชุด expert ที่ต่างกันทั้งหมด: **{len(all_vecs)}**")
    emit()
    emit(f"ชุดที่พบ: {', '.join(f'`{v}`' for v in sorted(all_vecs))}")
    emit()
    emit(f"> จาก {n_q} คำถาม ได้ผลลัพธ์ที่ต่างกันเพียง **{len(all_vecs)} แบบ** เท่ากับจำนวน "
         f"`question_type` ที่รองรับพอดี — ต่อให้รันครบ 6,507 คำถามก็ยังได้ {len(all_vecs)} แบบเท่าเดิม "
         f"เพราะผลลัพธ์ขึ้นกับ `question_type` อย่างเดียว")
    emit()

    emit("### ข้อจำกัดที่ต้องระบุใน report")
    emit()
    emit("1. **ไม่มีค่า `p` ให้ print** — คอลัมน์ `p_*` ใน CSV จึงว่างทั้งหมด baseline ตัดสินใจ "
         "แบบ 0/1 ตรง ๆ ไม่ผ่าน sigmoid และไม่มี threshold")
    emit("2. **เทียบกับไฟล์ `print_routing_router.py` แบบแถวต่อแถวไม่ได้** เพราะ "
         "`features_real.npz` ไม่ได้เก็บ image path / คำถามไว้ ถ้าต้องการเทียบตรง ๆ "
         "ให้ใช้โหมด `--live` ของไฟล์นั้นบนตัวอย่างชุดเดียวกัน")
    emit("3. เกณฑ์ detect expert อิงตาม `guide-rt.md` (ชิ้นส่วนใดของ prompt เป็นของ expert ตัวไหน) "
         "ถ้านิยามต่างออกไป ตัวเลขในตารางอาจเปลี่ยน")
    emit()

    # ---------------- เขียนไฟล์ ----------------
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_md = outdir / "routing_values_baseline.md"
    out_csv = outdir / "routing_values_baseline.csv"
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    if rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print()
    print("=" * 68)
    print(f"เขียนแล้ว: {out_md.relative_to(REPO)}")
    print(f"เขียนแล้ว: {out_csv.relative_to(REPO)}  ({len(rows)} แถว)")
    print("=" * 68)


if __name__ == "__main__":
    main()
