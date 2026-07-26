"""Qwen2VL_router.py — เปรียบเทียบ (A/B) การเลือก expert แบบเดิม vs Router Network

รันได้ทั้ง macOS และ Google Colab ด้วยโค้ดชุดเดียวกัน (เลือก device อัตโนมัติ +
ค้นหา repo root อัตโนมัติ ไม่ผูกกับโฟลเดอร์ที่รัน)

โหมด:
  baseline : ใช้ expert_generator เดิม (ผูกกับ question_type แบบ hardcode)
  router   : ใช้ RouterMLP (เฟส A) เลือกชุด expert จาก V_fuse แล้ว build_expert_prompt
  both     : รันทั้งสองบน "ตัวอย่างชุดเดียวกัน" แล้วเทียบ accuracy

ตัวอย่างการใช้งาน:
  # ทดสอบเล็ก ๆ ให้แน่ใจว่ารันครบ (โมเดลเล็ก เร็ว)
  python Qwen2VL_router.py --limit 3 --qwen-model Qwen/Qwen2-VL-2B-Instruct
  # รันเต็ม (เช่นบน Colab GPU)
  python Qwen2VL_router.py --limit 0 --qwen-model Qwen/Qwen2-VL-7B-Instruct
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # กันชน libomp (torch/faiss/sklearn) บน macOS

import argparse
import csv
import json
import logging
import platform
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix

# ให้ import โมดูลข้าง ๆ ได้ ไม่ว่าจะรันจากที่ใด (Experiments/ หรือ repo root บน Colab)
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from expert_generator import expert_generator          # baseline (ของเดิม)
import moxpert_lite as ml                                # helper ของ router


# ==========================================================================
# 0) ตัวช่วยทั่วไป: หา repo root, เลือก device, resolve path ของ reference
# ==========================================================================
def find_repo_root(start: Path) -> Path:
    """ไต่ขึ้นไปจนเจอโฟลเดอร์ที่มี Annotation/DS-MVTec.json = repo root"""
    for cand in [start, *start.parents]:
        if (cand / "Annotation" / "DS-MVTec.json").exists():
            return cand
    raise FileNotFoundError("หา repo root ไม่เจอ (ต้องมี Annotation/DS-MVTec.json)")


def select_device() -> str:
    """cuda (Colab/cloud GPU) -> mps (Mac) -> cpu"""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_reference(stored: str, repo: Path) -> str:
    """แปลง path ที่เก็บไว้ (เช่น ../Dataset/MMAD/...) ให้เป็น path เต็มจาก repo root"""
    return stored.strip().replace("\\", "/").replace("../", str(repo) + "/")


def knowledge_to_text(descriptions) -> str:
    """แปลง domain knowledge (dict/str) -> ข้อความ สำหรับ Knowledge Guide ของ router"""
    if isinstance(descriptions, dict):
        return "\n".join(f"{k.capitalize()}: {v}" for k, v in descriptions.items())
    return str(descriptions)


# ==========================================================================
# 1) โหลดโมเดล/ดัชนี/ข้อมูล
# ==========================================================================
def build_context(args, repo: Path, device: str):
    """โหลด encoder (CLIP), FAISS index, reference list, domain knowledge, router, Qwen"""
    import faiss
    if platform.system() == "Darwin":
        faiss.omp_set_num_threads(1)  # กัน segfault จาก OpenMP บน macOS
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info

    logging.info("โหลด CLIP encoder (moxpert_lite) ...")
    encoder = ml.MoXpertEncoder(clip_model_name=args.clip_model, device=device)  # ใช้ทั้ง raw + V_fuse

    logging.info("โหลด FAISS index + reference locations ...")
    index_img = faiss.read_index(str(repo / "Memory" / "memory.index"))
    ref_paths = [l.strip() for l in open(repo / "Memory" / "reference_image_locations.txt") if l.strip()]

    dk_path = repo / "Knowledge Guide" / "domain_knowledge_detection.json"
    domain_data = json.load(open(dk_path, encoding="utf-8"))

    # router (เฉพาะโหมดที่ต้องใช้)
    router, tau = None, None
    if args.mode in ("router", "both"):
        logging.info(f"โหลด router checkpoint: {args.router_ckpt}")
        router, tau_ckpt = ml.load_router(args.router_ckpt, device=device)
        tau = args.tau if args.tau is not None else tau_ckpt   # ใช้ τ ที่สั่ง หรือค่าที่เซฟไว้
        logging.info(f"router พร้อม (τ = {tau:.3f})")

    logging.info(f"โหลด Qwen2-VL: {args.qwen_model} ...")
    if device == "cuda":
        dtype, attn = torch.bfloat16, "sdpa"
    elif device == "mps":
        dtype, attn = torch.float16, "sdpa"
    else:
        dtype, attn = torch.float32, "sdpa"
    qwen = Qwen2VLForConditionalGeneration.from_pretrained(
        args.qwen_model, torch_dtype=dtype, attn_implementation=attn, trust_remote_code=True,
    ).to(device)
    qwen.eval()
    processor = AutoProcessor.from_pretrained(args.qwen_model, trust_remote_code=True, use_fast=False)

    return {
        "encoder": encoder, "index_img": index_img, "ref_paths": ref_paths,
        "domain_data": domain_data, "router": router, "tau": tau,
        "qwen": qwen, "processor": processor, "process_vision_info": process_vision_info,
        "device": device,
    }


def find_descriptions(domain_data, img_key: str) -> dict:
    """หา domain knowledge ของ object จาก key ของรูป (เลียนแบบ find_all_descriptions เดิม)"""
    object_name = img_key.split("/")[1] if "/" in img_key else img_key
    for _, sub in domain_data.items():
        if isinstance(sub, dict) and object_name in sub:
            return {"object_name": object_name, "descriptions": sub[object_name]}
    return {"object_name": object_name, "descriptions": "No descriptions found."}


# ==========================================================================
# 2) เรียก Qwen ให้ตอบ + ดึงตัวอักษรคำตอบ (A-D)
# ==========================================================================
def extract_letter(text: str) -> str:
    """ดึงตัวอักษรคำตอบ A-D จากข้อความ (พอร์ตจาก Analysis_Results_Mac.ipynb)

    ใช้ตัวเดียวกับ pipeline เดิมตอนวิเคราะห์ เพื่อให้เทียบ baseline อย่างยุติธรรม:
    - ลอง match ตัวอักษรตัวแรก ^([A-D])   เช่น "A." / "A) ..." / "A"
    - ถ้าไม่เจอ ลองหา A-D ตัวแรกในข้อความ เช่น "The answer is A"
    - parse ไม่ได้ -> "N/A"
    """
    if not text:
        return "N/A"
    s = text.strip()
    m = re.match(r"^\(?([A-D])\b", s)          # ขึ้นต้นด้วยตัวเลือก
    if m:
        return m.group(1)
    m = re.search(r"\b([A-D])\b", s)           # เจอ A-D ที่ไหนสักที่
    if m:
        return m.group(1)
    return "N/A"


def qwen_answer(ctx, messages) -> str:
    """ส่ง messages เข้า Qwen2-VL แล้วคืนตัวอักษรคำตอบ (A-D หรือ N/A)"""
    proc = ctx["processor"]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = ctx["process_vision_info"](messages)
    inputs = proc(text=[text], images=image_inputs, return_tensors="pt").to(ctx["device"])
    with torch.no_grad():
        out = ctx["qwen"].generate(**inputs, max_new_tokens=10)   # คำตอบเป็นตัวอักษรเดียว
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
    resp = proc.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return extract_letter(resp[0] if resp else "")   # ตัดอักษรแบบเดียวกับ pipeline เดิม


# ==========================================================================
# 3) การวัดผล: accuracy + confusion matrix (count + percentage, text อย่างเดียว)
# ==========================================================================
QTYPE_ORDER = ["Anomaly Detection", "Defect Classification", "Defect Localization",
               "Defect Description", "Defect Analysis"]   # ลำดับคงที่ตอนพิมพ์


def _cm_labels_acc(y_true, y_pred):
    """คำนวณ (cm, labels, acc) ครั้งเดียว ใช้ร่วมทั้ง text และรูป PNG

    - labels = ตัวอักษรที่ปรากฏจริง (Anomaly Detection ได้ 2x2 A/B, type อื่นได้ 4x4)
    - cm = confusion_matrix (แถว=True, คอลัมน์=Pred)
    """
    labels = sorted(set(y_true) | set(y_pred))
    import warnings
    with warnings.catch_warnings():                     # กัน warning ตอนมี label เดียว (sample เล็ก)
        warnings.simplefilter("ignore")
        cm = confusion_matrix(y_true, y_pred, labels=labels)
    return cm, labels, accuracy_score(y_true, y_pred)


def format_confusion(y_true, y_pred, title: str) -> str:
    """สร้างตาราง confusion matrix แบบข้อความ 2 บล็อก: by count และ by percentage

    - แถว = True (เฉลย), คอลัมน์ = Predicted (โมเดลทาย)
    - label = ตัวอักษรที่ปรากฏจริง (Anomaly Detection ได้ 2x2 A/B, type อื่นได้ 4x4)
    - percentage = normalize ตามแถว (หารด้วยผลรวมของแต่ละ true class x 100)
    """
    y_true = list(y_true); y_pred = list(y_pred)
    if not y_true:
        return f"\n[{title}] (ไม่มีข้อมูล)\n"
    cm, labels, acc = _cm_labels_acc(y_true, y_pred)
    n = len(y_true)

    lines = [f"\n[{title}]  n={n}  accuracy={acc:.3f}"]
    header = "true\\pred |" + "".join(f"{l:>6s}" for l in labels)

    # (a) by count — จำนวนเต็ม
    lines.append("  -- by count --")
    lines.append("  " + header)
    for i, lt in enumerate(labels):
        lines.append("  " + f"{lt:>8s} |" + "".join(f"{int(cm[i, j]):>6d}" for j in range(len(labels))))

    # (b) by percentage — หารตามผลรวมแถว (กันหารศูนย์)
    row_sum = cm.sum(axis=1, keepdims=True)
    pct = np.divide(cm * 100.0, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum != 0)
    lines.append("  -- by percentage (row-normalized) --")
    lines.append("  " + header)
    for i, lt in enumerate(labels):
        lines.append("  " + f"{lt:>8s} |" + "".join(f"{pct[i, j]:>6.1f}" for j in range(len(labels))))
    return "\n".join(lines)


def report_metrics(rows, mode: str) -> None:
    """พิมพ์ accuracy + confusion matrix (overall แล้วตามด้วยแยกราย question_type)"""
    print("\n" + "#" * 68)
    print(f"# CONFUSION MATRIX + ACCURACY — mode = {mode}")
    print("#" * 68)
    yt = [r["Correct"] for r in rows]
    yp = [r["Predicted"] for r in rows]

    # overall (รวมทุก type)
    print(format_confusion(yt, yp, f"OVERALL / {mode}"))

    # แยกราย question_type (เรียงตามลำดับคงที่ ตามด้วย type อื่นที่อาจโผล่)
    seen = {r["Question Type"] for r in rows}
    ordered = [t for t in QTYPE_ORDER if t in seen] + sorted(seen - set(QTYPE_ORDER))
    for t in ordered:
        ryt = [r["Correct"] for r in rows if r["Question Type"] == t]
        ryp = [r["Predicted"] for r in rows if r["Question Type"] == t]
        print(format_confusion(ryt, ryp, f"{t} / {mode}"))


def save_confusion_png(rows, mode: str, outdir) -> None:
    """เซฟ confusion matrix เป็นรูป PNG 2 ไฟล์: by count และ by percentage

    แต่ละไฟล์มี subplot: overall + แยกราย question_type (สไตล์ Analysis_Results_Mac.ipynb)
    ใช้ backend Agg เพื่อรัน headless ได้ทั้ง Mac และ Colab. ถ้าไม่มี matplotlib -> ข้าม
    """
    try:
        import matplotlib
        matplotlib.use("Agg")               # ไม่ต้องมีหน้าจอ (headless / Colab)
        import matplotlib.pyplot as plt
    except Exception as e:
        logging.warning(f"[png] ข้าม (import matplotlib ไม่ได้: {e}) — ยังมี text ครบ")
        return

    # เตรียมชุด panel: overall + แต่ละ question_type ที่มี (เรียงตามลำดับคงที่)
    seen = {r["Question Type"] for r in rows}
    ordered = [t for t in QTYPE_ORDER if t in seen] + sorted(seen - set(QTYPE_ORDER))
    panels = [("OVERALL", rows)] + [(t, [r for r in rows if r["Question Type"] == t]) for t in ordered]

    from math import ceil
    outdir = Path(outdir)

    # ทำ 2 เวอร์ชัน: count (จำนวนเต็ม) และ percent (row-normalized)
    for kind in ("count", "percent"):
        ncols = min(3, len(panels))
        nrows = ceil(len(panels) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows), squeeze=False)
        for ax in axes.flat:                 # ปิดแกนที่ไม่ได้ใช้ไว้ก่อน
            ax.axis("off")

        for k, (title, prows) in enumerate(panels):
            ax = axes[k // ncols][k % ncols]
            ax.axis("on")
            yt = [r["Correct"] for r in prows]
            yp = [r["Predicted"] for r in prows]
            if not yt:
                ax.set_title(f"{title}\n(ไม่มีข้อมูล)", fontsize=9); ax.axis("off"); continue
            cm, labels, acc = _cm_labels_acc(yt, yp)

            if kind == "count":
                mat = cm.astype(float); fmt = lambda v: f"{int(v)}"
            else:  # percent: หารตามผลรวมแถว x 100
                rs = cm.sum(axis=1, keepdims=True)
                mat = np.divide(cm * 100.0, rs, out=np.zeros_like(cm, dtype=float), where=rs != 0)
                fmt = lambda v: f"{v:.1f}"

            im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=(100 if kind == "percent" else mat.max() or 1))
            ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
            ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
            ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            ax.set_title(f"{title}\nacc={acc:.3f} (n={len(yt)})", fontsize=9)
            thr = mat.max() / 2 if mat.max() > 0 else 0.5
            for i in range(len(labels)):
                for j in range(len(labels)):
                    ax.text(j, i, fmt(mat[i, j]), ha="center", va="center", fontsize=8,
                            color="white" if mat[i, j] > thr else "black")   # ตัวเลขในเซลล์

        suffix = "by count" if kind == "count" else "by percentage (row-normalized)"
        fig.suptitle(f"Confusion Matrix — {mode} — {suffix}", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out_png = outdir / f"Confusion_{mode}_{kind}.png"
        fig.savefig(out_png, dpi=120, bbox_inches="tight")
        plt.close(fig)
        logging.info(f"เซฟรูป {out_png}")


def report_routing_agreement(rows) -> None:
    """พิมพ์สัดส่วนที่ router เลือก expert ตรงกับ heuristic (และจำนวนที่ route ต่าง)"""
    flagged = [r for r in rows if r.get("Routing Agree") in ("yes", "no")]
    if not flagged:
        return
    agree = sum(1 for r in flagged if r["Routing Agree"] == "yes")
    total = len(flagged)
    diff = total - agree
    print("\n" + "-" * 68)
    print(f"routing-agreement (router เลือกตรง heuristic): {agree}/{total} = {agree/total*100:.1f}%")
    print(f"router route ต่างจาก baseline: {diff} ตัวอย่าง  "
          f"(accuracy ที่ต่างจาก baseline มาจากกลุ่มนี้เท่านั้น เพราะ prompt-parity)")
    print("-" * 68)


# ==========================================================================
# 4) ลูปประเมินผล A/B
# ==========================================================================
def evaluate(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    repo = find_repo_root(_THIS_DIR)
    device = select_device()
    logging.info(f"repo={repo} | device={device} | mode={args.mode}")

    ctx = build_context(args, repo, device)
    data_root = repo / "Dataset" / "MMAD"
    data = json.load(open(repo / "Annotation" / "DS-MVTec.json"))

    modes = ["baseline", "router"] if args.mode == "both" else [args.mode]
    # เก็บ correct/total แยกตาม (mode, question_type)
    stats = {m: defaultdict(lambda: [0, 0]) for m in modes}
    rows = {m: [] for m in modes}

    processed = 0
    for idx, (img_key, item) in enumerate(data.items()):
        if args.limit and processed >= args.limit:
            break
        query_image_path = str(data_root / img_key)
        if not os.path.exists(query_image_path):     # ข้ามรูปที่ไฟล์หาย (เช่น good บางรูป)
            continue
        processed += 1
        logging.info(f"[{processed}] {img_key}")

        # --- retrieval: ฟีเจอร์ดิบ -> FAISS -> reference image (ใช้ร่วมทั้งสองโหมด) ---
        raw = ctx["encoder"].encode_image_raw(query_image_path)
        _, I = ctx["index_img"].search(np.expand_dims(raw.astype(np.float32), 0), 1)
        ref_stored = ctx["ref_paths"][int(I[0][0])]
        ref_abs = resolve_reference(ref_stored, repo)

        dk = find_descriptions(ctx["domain_data"], img_key)          # domain knowledge (dict)
        dk_text = knowledge_to_text(dk["descriptions"])              # เวอร์ชันข้อความ (สำหรับ router)

        # เข้ารหัส V_img ครั้งเดียวต่อรูป (ใช้ซ้ำทุกคำถามของรูปนี้) เมื่อโหมดใช้ router
        v_img = ctx["encoder"].encode_image(query_image_path) if ctx["router"] is not None else None

        for conv in item["conversation"]:
            question = conv["Question"]
            correct = conv["Answer"]
            qtype = conv["type"]
            options_text = "\n".join(f"{k}: {v}" for k, v in conv["Options"].items())

            for m in modes:
                active = None
                agree = ""       # ('' = baseline, 'yes'/'no' = router เลือกตรง heuristic ไหม)
                realized = ""    # template ที่ใช้จริง (baseline-template / router-template)
                if m == "baseline":
                    # เลือก prompt แบบเดิม (ผูก question_type) — ข้าม type ที่ไม่รองรับ
                    try:
                        messages = expert_generator(ref_abs, query_image_path, qtype,
                                                    question, options_text, dk)
                    except ValueError:
                        continue
                else:  # router
                    # V_fuse -> p -> ชุด expert ที่ router เลือก
                    v_text = ctx["encoder"].encode_text(question)
                    v_fuse = ml.MoXpertEncoder.fuse(v_img, v_text)
                    p = ctx["router"].predict_proba(v_fuse)[0]
                    active = ml.experts_from_vector(p, tau=ctx["tau"])

                    # --- Prompt-parity: ถ้า router เลือกตรง heuristic ของ type นี้ ---
                    #     ให้สร้าง prompt ด้วย expert_generator เดิม (เหมือน baseline เป๊ะ)
                    #     เพื่อคุมถ้อยคำ prompt ให้คงที่ เหลือความต่างเฉพาะ "การ routing"
                    prior = ml.default_prior(qtype)
                    if set(active) == set(prior):
                        agree = "yes"
                        try:
                            messages = expert_generator(ref_abs, query_image_path, qtype,
                                                        question, options_text, dk)
                            realized = "baseline-template"
                        except ValueError:
                            # type ที่ของเดิมไม่รองรับ -> ใช้ build_expert_prompt แทน
                            messages = ml.build_expert_prompt(active, ml.Query(
                                query_image_path=query_image_path, question=question,
                                options_text=options_text, object_name=dk["object_name"],
                                question_type=qtype, reference_image_path=ref_abs, domain_knowledge=dk_text))
                            realized = "router-template"
                    else:
                        # router เลือกต่างจาก heuristic จริง -> ใช้ prompt ตามชุดที่เลือก
                        agree = "no"
                        realized = "router-template"
                        messages = ml.build_expert_prompt(active, ml.Query(
                            query_image_path=query_image_path, question=question,
                            options_text=options_text, object_name=dk["object_name"],
                            question_type=qtype,
                            reference_image_path=ref_abs if ml.REFERENCE_EXTRACTOR in active else None,
                            domain_knowledge=dk_text if ml.KNOWLEDGE_GUIDE in active else None,
                        ))

                pred = qwen_answer(ctx, messages)
                ok = (pred == correct)
                stats[m][qtype][0] += int(ok)
                stats[m][qtype][1] += 1
                rows[m].append({
                    "Image Path": img_key, "Question Type": qtype, "Question": question,
                    "Predicted": pred, "Correct": correct,
                    "Active Experts": "|".join(active) if active else "(baseline)",
                    "Routing Agree": agree, "Realized": realized,
                })

    # --- เขียน CSV ต่อโหมด ---
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    for m in modes:
        out_csv = outdir / f"Results_{m}.csv"
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["Image Path", "Question Type", "Question",
                                              "Predicted", "Correct", "Active Experts",
                                              "Routing Agree", "Realized"])
            w.writeheader(); w.writerows(rows[m])
        logging.info(f"เขียน {out_csv} ({len(rows[m])} แถว)")

    # --- สรุปตารางเทียบ accuracy ---
    print("\n" + "=" * 68)
    print(f"{'question_type':24s}  " + "  ".join(f"{m:>12s}" for m in modes))
    print("-" * 68)
    qtypes = sorted({t for m in modes for t in stats[m]})
    for t in qtypes:
        cells = []
        for m in modes:
            c, n = stats[m][t]
            cells.append(f"{(c/n if n else 0):.3f}({n})")
        print(f"{t:24s}  " + "  ".join(f"{c:>12s}" for c in cells))
    print("-" * 68)
    overall = []
    for m in modes:
        c = sum(v[0] for v in stats[m].values()); n = sum(v[1] for v in stats[m].values())
        overall.append(f"{(c/n if n else 0):.3f}({n})")
    print(f"{'OVERALL':24s}  " + "  ".join(f"{o:>12s}" for o in overall))
    print("=" * 68)

    # --- routing-agreement (โหมด router) + confusion matrix ต่อโหมด (text + PNG) ---
    if "router" in rows:
        report_routing_agreement(rows["router"])
    for m in modes:
        report_metrics(rows[m], m)              # text
        save_confusion_png(rows[m], m, outdir)  # รูป PNG (count + percentage)


def analyze_csv(path: str) -> None:
    """โหมด --analyze-csv: อ่านผลที่รันไว้แล้วมาพิมพ์ accuracy + CM โดยไม่โหลดโมเดล

    CSV ต้องมีคอลัมน์ Question Type, Predicted, Correct (เช่น Results_router.csv)
    """
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"[analyze-csv] ไม่มีข้อมูลใน {path}"); return
    mode = Path(path).stem.replace("Results_", "") or "csv"
    print(f"[analyze-csv] อ่าน {len(rows)} แถวจาก {path}")
    report_routing_agreement(rows)
    report_metrics(rows, mode)                          # text
    save_confusion_png(rows, mode, Path(path).parent)   # เซฟ PNG ข้างไฟล์ CSV


def parse_args():
    p = argparse.ArgumentParser(description="A/B: baseline expert_generator vs Router Network")
    p.add_argument("--limit", type=int, default=3,
                   help="จำนวนรูปที่ประเมิน (0 = ทั้งหมด) ; ค่าน้อย ๆ ไว้ทดสอบว่ารันครบ")
    p.add_argument("--mode", choices=["baseline", "router", "both"], default="both")
    p.add_argument("--qwen-model", default="Qwen/Qwen2-VL-7B-Instruct",
                   help="โมเดลเล็ก (2B) ไว้ทดสอบเร็ว ๆ / 7B ไว้รันจริง")
    p.add_argument("--clip-model", default="ViT-B/16")
    p.add_argument("--router-ckpt", default=None,
                   help="ค่าเริ่มต้น = <repo>/Router_Network/artifacts/router_real.pt")
    p.add_argument("--tau", type=float, default=None, help="override threshold (ไม่ใส่ = ใช้ค่าจาก ckpt)")
    p.add_argument("--outdir", default=None, help="โฟลเดอร์เก็บ CSV (ค่าเริ่มต้น = โฟลเดอร์สคริปต์)")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--analyze-csv", default=None,
                   help="อ่านผลจาก CSV ที่รันไว้แล้วมาพิมพ์ accuracy+CM (ไม่โหลดโมเดล)")
    args = p.parse_args()
    if args.analyze_csv:      # โหมดวิเคราะห์อย่างเดียว ไม่ต้องหา router ckpt/outdir
        return args
    repo = find_repo_root(_THIS_DIR)
    if args.router_ckpt is None:
        args.router_ckpt = str(repo / "Router_Network" / "artifacts" / "router_real.pt")
    if args.outdir is None:
        args.outdir = str(_THIS_DIR)
    return args


if __name__ == "__main__":
    _args = parse_args()
    if _args.analyze_csv:
        analyze_csv(_args.analyze_csv)     # วิเคราะห์จาก CSV อย่างเดียว
    else:
        evaluate(_args)                    # รันจริง (baseline/router)
