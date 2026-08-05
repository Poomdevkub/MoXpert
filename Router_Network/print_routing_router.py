#!/usr/bin/env python3
"""print ค่าที่ได้จากขั้นตอน ROUTER NETWORK — เดินตามเส้นทางจริงของ paper เส้นทางเดียว

    Eq.1-2  รูป + คำถาม --CLIP (frozen)--> V_img(576), V_text(576)
    Eq.3    fuse ----------------------->  V_fuse(1152)
    Eq.4    p = sigmoid(MLP(V_fuse)) --->  p(4)
    Alg.1   y_i = 1 ถ้า p_i > τ -------->  ชุด expert --> build_expert_prompt

ไม่มีโหมดให้เลือกแล้ว ทุกตัวอย่างเดินครบเส้นเสมอ ส่วนแคช `V_fuse` เป็นเรื่องภายใน
(ข้ามขั้น CLIP เมื่อเคยเข้ารหัสไว้แล้ว) ไม่ได้เปลี่ยนเส้นทางที่รายงาน

รัน:
    python Router_Network/print_routing_router.py
    python Router_Network/print_routing_router.py --limit-images 5 --detail 2
    python Router_Network/print_routing_router.py --refresh        # เข้ารหัสใหม่ทั้งหมด
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np                                                 # noqa: E402
import torch                                                       # noqa: E402

_THIS_DIR = Path(__file__).resolve().parent
REPO = _THIS_DIR.parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(REPO / "Experiments"))

import moxpert_lite as ml                                          # noqa: E402
from expert_generator import expert_generator                      # noqa: E402
from Qwen2VL_router import (                                       # noqa: E402
    find_descriptions, knowledge_to_text, resolve_reference, select_device,
)

EXPERT_NAMES = ml.EXPERT_NAMES
SHORT = {ml.REFERENCE_EXTRACTOR: "Ref", ml.KNOWLEDGE_GUIDE: "Know",
         ml.REASONING_EXPERT: "Reason", ml.DECISION_MAKER: "Decide"}
CACHE_PATH = REPO / "Router_Network" / "artifacts" / "routing_cache.npz"
FEATURES_REAL = REPO / "Router_Network" / "artifacts" / "features_real.npz"

md_lines: list[str] = []


def emit(line: str = "") -> None:
    print(line)
    md_lines.append(line)


# ==========================================================================
# ตัวช่วย
# ==========================================================================
def forward_all(model, X: np.ndarray):
    """Eq.4 — รัน router คืน (logits, p)

    `RouterMLP` ใน moxpert_lite ไม่มีเมธอด .logits() (มีเฉพาะเวอร์ชันในโน้ตบุ๊ก)
    จึงเรียก model.net(x) ตรง ๆ ซึ่งคือ output ก่อนเข้า sigmoid พอดี
    (forward() = sigmoid(self.net(x)) ที่ moxpert_lite.py:185)
    """
    model.eval()
    device = next(model.parameters()).device
    xt = torch.as_tensor(np.atleast_2d(X), dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model.net(xt).cpu().numpy()
    return logits, 1.0 / (1.0 + np.exp(-logits))


def per_expert_f1(y_true: np.ndarray, y_pred: np.ndarray) -> list[float]:
    """F1 รายตัว expert (สูตรเดียวกับ macro_f1 ในโน้ตบุ๊ก แต่คืนแยกราย column)"""
    out = []
    for j in range(y_true.shape[1]):
        tp = int(np.sum((y_pred[:, j] == 1) & (y_true[:, j] == 1)))
        fp = int(np.sum((y_pred[:, j] == 1) & (y_true[:, j] == 0)))
        fn = int(np.sum((y_pred[:, j] == 0) & (y_true[:, j] == 1)))
        denom = 2 * tp + fp + fn
        out.append(1.0 if denom == 0 else (2 * tp) / denom)
    return out


def text_histogram(values: np.ndarray, tau: float, bins: int = 10, width: int = 40) -> list[str]:
    counts, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    top = max(1, counts.max())
    return [f"`{lo:.1f}-{hi:.1f}` | `{'#' * int(c / top * width):<{width}}` | {c}"
            f"{' <- τ' if lo <= tau < hi else ''}"
            for c, lo, hi in zip(counts, edges[:-1], edges[1:])]


def vec_of(y_row) -> str:
    return "".join(str(int(v)) for v in y_row)


def prompt_of(messages) -> tuple[list[str], str]:
    """แกะ messages -> (รายการรูป, ข้อความรวม)"""
    content = messages[0]["content"]
    return ([c["image"] for c in content if c["type"] == "image"],
            "\n".join(c["text"] for c in content if c["type"] == "text"))


# ==========================================================================
# เก็บตัวอย่างจาก dataset ตามลำดับมาตรฐาน (ลำดับเดียวกับที่ features_real.npz ใช้)
# ==========================================================================
def collect_records(limit_images: int):
    """วน DS-MVTec.json ตามลำดับ ข้ามรูปที่ไฟล์หาย -> list ของ record ต่อคำถาม"""
    data = json.load(open(REPO / "Annotation" / "DS-MVTec.json"))
    domain_data = json.load(open(REPO / "Knowledge Guide" / "domain_knowledge_detection.json",
                                 encoding="utf-8"))
    data_root = REPO / "Dataset" / "MMAD"

    records, n_img = [], 0
    for img_key, item in data.items():
        if n_img >= limit_images:
            break
        query_image_path = str(data_root / img_key)
        if not os.path.exists(query_image_path):
            continue
        n_img += 1
        dk = find_descriptions(domain_data, img_key)
        for conv in item["conversation"]:
            records.append({
                "image_key": img_key,
                "query_image_path": query_image_path,
                "question": conv["Question"],
                "options_text": "\n".join(f"{k}: {v}" for k, v in conv["Options"].items()),
                "question_type": conv["type"],
                "dk": dk,
                "dk_text": knowledge_to_text(dk["descriptions"]),
            })
    return records, n_img


# ==========================================================================
# Eq.1-3 — เข้ารหัส V_fuse (ใช้แคชถ้ามี) + หา reference image ด้วย FAISS
# ==========================================================================
def encode_records(records, device, use_cache: bool, refresh: bool):
    """คืน (X, ref_paths, from_cache) — X คือ V_fuse (n,1152)

    แคชเก็บ image_key + question ไว้ด้วย จึงตรวจได้ว่าตรงกับชุดที่กำลังจะรันหรือไม่
    ถ้าไม่ตรง (เช่นเปลี่ยน --limit-images) จะเข้ารหัสใหม่ ไม่ใช้แคชผิดชุด
    """
    keys = [r["image_key"] for r in records]
    qs = [r["question"] for r in records]

    if use_cache and not refresh and CACHE_PATH.exists():
        c = np.load(CACHE_PATH, allow_pickle=True)
        if (len(c["image_keys"]) >= len(records)
                and list(c["image_keys"][:len(records)]) == keys
                and list(c["questions"][:len(records)]) == qs):
            print(f"[แคช] ใช้ V_fuse ที่เข้ารหัสไว้แล้วจาก {CACHE_PATH.name} "
                  f"({len(records)} คำถาม) -> ข้ามขั้น CLIP")
            return (c["X"][:len(records)].astype(np.float32),
                    list(c["ref_paths"][:len(records)]), True)
        print("[แคช] มีไฟล์แคชแต่ไม่ตรงกับชุดที่จะรัน -> เข้ารหัสใหม่")

    import faiss
    import platform
    if platform.system() == "Darwin":
        faiss.omp_set_num_threads(1)

    print("[โหลด] CLIP encoder (frozen ViT-B/16) ...")
    enc = ml.MoXpertEncoder(clip_model_name="ViT-B/16", device=device)
    print("[โหลด] FAISS index สำหรับหา reference image ...")
    index_img = faiss.read_index(str(REPO / "Memory" / "memory.index"))
    ref_list = [l.strip() for l in open(REPO / "Memory" / "reference_image_locations.txt") if l.strip()]

    X = np.zeros((len(records), 1152), dtype=np.float32)
    ref_paths: list[str] = []
    v_img_cache: dict[str, np.ndarray] = {}
    ref_cache: dict[str, str] = {}
    t0 = time.time()
    for i, r in enumerate(records):
        path = r["query_image_path"]
        if path not in v_img_cache:                       # เข้ารหัสรูปครั้งเดียวต่อรูป
            raw = enc.encode_image_raw(path)              # ฟีเจอร์ดิบ 512 -> FAISS
            _, idx = index_img.search(np.expand_dims(raw.astype(np.float32), 0), 1)
            ref_cache[path] = resolve_reference(ref_list[int(idx[0][0])], REPO)
            v_img_cache[path] = enc.encode_image(path)    # Eq.1
        v_text = enc.encode_text(r["question"])           # Eq.2
        X[i] = ml.MoXpertEncoder.fuse(v_img_cache[path], v_text)   # Eq.3
        ref_paths.append(ref_cache[path])
        if (i + 1) % 200 == 0:
            print(f"      เข้ารหัสแล้ว {i + 1}/{len(records)} คำถาม")
    print(f"[เข้ารหัส] เสร็จ {len(records)} คำถาม ใน {time.time() - t0:.1f}s")

    if use_cache:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE_PATH, X=X,
                            image_keys=np.array(keys, dtype=object),
                            questions=np.array(qs, dtype=object),
                            qtypes=np.array([r["question_type"] for r in records], dtype=object),
                            ref_paths=np.array(ref_paths, dtype=object))
        print(f"[แคช] เซฟ {CACHE_PATH.name} แล้ว (รันครั้งหน้าจะข้ามขั้น CLIP)")
    return X, ref_paths, False


# ==========================================================================
# กู้ is_train จาก features_real.npz โดยจับคู่ตามตำแหน่ง + ตรวจสอบก่อนเชื่อ
# ==========================================================================
def recover_split(X: np.ndarray):
    """คืน array ของ 'train'/'eval'/'unknown' ยาวเท่า X

    features_real.npz เก็บ is_train ไว้แต่ไม่มี key ให้จับคู่ จึงจับตามตำแหน่งแถว
    (สร้างด้วยการวน DS-MVTec.json ลำดับเดียวกัน) แล้ว **ตรวจด้วยการเทียบ V_fuse**
    ถ้าไม่ตรงจะตั้งเป็น unknown ทั้งหมด ไม่แปะ label มั่ว
    """
    split = np.array(["unknown"] * len(X), dtype=object)
    if not FEATURES_REAL.exists():
        return split, "ไม่พบ features_real.npz -> ไม่มีข้อมูล train/eval"

    d = np.load(FEATURES_REAL, allow_pickle=True)
    Xr, is_train = d["X"], d["is_train"]
    n = min(len(X), len(Xr))
    probe = list(range(0, n, max(1, n // 20)))[:20]        # สุ่มตรวจ ~20 จุด
    diffs = [float(np.abs(X[i] - Xr[i]).max()) for i in probe]
    worst = max(diffs) if diffs else float("inf")

    if worst > 1e-3:
        return split, (f"⚠ จับคู่แถวกับ features_real.npz **ไม่ผ่าน** "
                       f"(ต่างสูงสุด {worst:.2e} > 1e-3) -> ตั้ง split เป็น unknown ทั้งหมด")
    split[:n] = np.where(is_train[:n], "train", "eval")
    note = (f"จับคู่แถวกับ features_real.npz ผ่าน (ต่างสูงสุด {worst:.2e} จาก {len(probe)} จุดที่ตรวจ) "
            f"-> กู้ train/eval ได้ {n}/{len(X)} แถว")
    if n < len(X):
        # โน้ตบุ๊กตอนสร้าง npz ตัดที่ "ลำดับ entry ใน JSON >= 300" (มีไฟล์จริง 282 รูป -> 1006 คำถาม)
        # ส่วนสคริปต์นี้นับ "รูปที่มีไฟล์จริง" ให้ครบตาม --limit-images จึงได้แถวมากกว่า
        note += (f" ส่วนอีก {len(X) - n} แถวเป็น `unknown` เพราะเกินช่วงที่ npz ครอบคลุม "
                 f"(ใช้ `--limit-images 282` ถ้าต้องการชุดที่ตรงกับตอนเทรนเป๊ะ ๆ = {n} แถว ไม่มี unknown)")
    return split, note


# ==========================================================================
# main
# ==========================================================================
def main():
    ap = argparse.ArgumentParser(
        description="print ค่าจากขั้นตอน router network (RouterMLP) — เส้นทางเดียวตาม paper")
    ap.add_argument("--limit-images", type=int, default=300,
                    help="จำนวนรูป (300 = ช่วงที่ features_real.npz มี is_train ให้กู้)")
    ap.add_argument("--detail", type=int, default=3, help="จำนวนตัวอย่างที่โชว์เส้นทางเต็ม + prompt")
    ap.add_argument("--rows", type=int, default=20, help="จำนวนแถวในตารางค่าต่อคำถาม")
    ap.add_argument("--refresh", action="store_true", help="เข้ารหัสใหม่ ไม่ใช้แคช")
    ap.add_argument("--no-cache", action="store_true", help="ไม่อ่านไม่เขียนแคช")
    ap.add_argument("--ckpt", default=str(REPO / "Router_Network" / "artifacts" / "router_real.pt"))
    ap.add_argument("--tau", type=float, default=None, help="override τ (default = ค่าจาก checkpoint)")
    ap.add_argument("--outdir", default=str(REPO / "Router_Network" / "artifacts"))
    args = ap.parse_args()

    device = select_device()
    model, tau_ckpt = ml.load_router(args.ckpt, device=device)
    tau = args.tau if args.tau is not None else tau_ckpt
    n_param = sum(p.numel() for p in model.parameters())

    records, n_img = collect_records(args.limit_images)
    X, ref_paths, from_cache = encode_records(records, device, not args.no_cache, args.refresh)
    split, split_note = recover_split(X)

    logits, p = forward_all(model, X)                     # Eq.4
    y = ml.apply_threshold(p, tau)                        # Algorithm 1
    margin = p - tau

    # ---------------- หัวรายงาน ----------------
    emit("# ค่าจากขั้นตอน ROUTER NETWORK (`RouterMLP`)")
    emit()
    emit(f"สร้างเมื่อ {datetime.now():%Y-%m-%d %H:%M:%S} | device `{device}` | "
         f"{n_img} รูป / {len(records)} คำถาม")
    emit()
    emit("## เส้นทางที่เดิน (เส้นทางเดียวตาม paper)")
    emit()
    emit("```")
    emit("Eq.1-2  รูป + คำถาม --CLIP (frozen ViT-B/16)--> V_img(576), V_text(576)")
    emit("Eq.3    fuse ------------------------------->  V_fuse(1152)")
    emit("Eq.4    p = sigmoid(MLP(V_fuse)) ----------->  p(4)")
    emit(f"Alg.1   y_i = 1 ถ้า p_i > τ ({tau:.3f}) ---------->  ชุด expert --> build_expert_prompt")
    emit("```")
    emit()
    emit(f"- **checkpoint:** `{Path(args.ckpt).name}`")
    emit(f"- **สถาปัตยกรรม:** `{model.in_dim} → {' → '.join(str(h) for h in model.hidden)} → "
         f"{model.n_experts}` + sigmoid | dropout {model.dropout}")
    emit(f"- **จำนวนพารามิเตอร์ที่เรียนรู้ได้:** {n_param:,}")
    emit(f"- **τ ที่ใช้:** {tau:.3f}"
         f"{' (ค่าจาก checkpoint — optimize มาแล้ว ตรงตามที่ paper กำหนด)' if args.tau is None else ' (override จาก --tau)'}")
    emit(f"- **ขั้น CLIP:** {'ใช้ V_fuse จากแคช (ข้ามการเข้ารหัสซ้ำ)' if from_cache else 'เข้ารหัสใหม่ในรอบนี้'}"
         f" — ผลลัพธ์เหมือนกัน แคชเป็นแค่การเร่งความเร็ว ไม่ได้เปลี่ยนเส้นทาง")
    emit(f"- **train/eval:** {split_note}")
    emit()
    emit("---")
    emit()

    # ---------------- เส้นทางละเอียด N ตัวอย่างแรก ----------------
    if args.detail > 0:
        emit(f"## เส้นทางเต็มของ {min(args.detail, len(records))} ตัวอย่างแรก")
        emit()
        for i in range(min(args.detail, len(records))):
            r = records[i]
            active = ml.experts_from_vector(p[i], tau=tau)
            emit(f"### ตัวอย่างที่ {i + 1} — `{r['image_key']}`")
            emit()
            emit(f"- **คำถาม:** {r['question']}")
            emit(f"- **question_type:** {r['question_type']} | **split:** {split[i]}")
            emit(f"- **V_img:** (576,) | **V_text:** (576,) | **V_fuse:** ({X.shape[1]},)")
            emit(f"- **reference image (FAISS):** `{Path(ref_paths[i]).name}`")
            emit()
            emit("| expert | logit | p = sigmoid(logit) | τ | p − τ | y |")
            emit("|---|--:|--:|--:|--:|:--:|")
            for j, n in enumerate(EXPERT_NAMES):
                emit(f"| {n} | {logits[i][j]:+.4f} | {p[i][j]:.4f} | {tau:.3f} | "
                     f"{margin[i][j]:+.4f} | {int(y[i][j])} |")
            emit()
            prior = ml.default_prior(r["question_type"])
            emit(f"- **ชุด expert ที่ router เลือก:** `{vec_of(y[i])}` = {', '.join(active) or '—'}")
            emit(f"- **เทียบ `HEURISTIC_PRIORS`:** {', '.join(SHORT[n] for n in prior)} "
                 f"→ {'ตรงกัน' if set(active) == set(prior) else '**ต่างกัน**'}")
            emit()

            q = ml.Query(query_image_path=r["query_image_path"], question=r["question"],
                         options_text=r["options_text"], object_name=r["dk"]["object_name"],
                         question_type=r["question_type"],
                         reference_image_path=ref_paths[i] if ml.REFERENCE_EXTRACTOR in active else None,
                         domain_knowledge=r["dk_text"] if ml.KNOWLEDGE_GUIDE in active else None)
            imgs, text = prompt_of(ml.build_expert_prompt(active, q))
            emit("<details><summary>prompt ที่ <code>build_expert_prompt</code> ประกอบ (router)</summary>")
            emit()
            emit("```")
            emit(f"images ({len(imgs)}): {[Path(x).name for x in imgs]}")
            emit(text)
            emit("```")
            emit()
            emit("</details>")
            emit()
            try:
                b_imgs, b_text = prompt_of(expert_generator(
                    ref_paths[i], r["query_image_path"], r["question_type"],
                    r["question"], r["options_text"], r["dk"]))
                emit("<details><summary>prompt ของ baseline ตัวเดียวกัน (<code>expert_generator</code>) ไว้เทียบ</summary>")
                emit()
                emit("```")
                emit(f"images ({len(b_imgs)}): {[Path(x).name for x in b_imgs]}")
                emit(b_text)
                emit("```")
                emit()
                emit("</details>")
            except ValueError:
                emit("> baseline ไม่รองรับ question_type นี้ (`raise ValueError`)")
            emit()
        emit("---")
        emit()

    # ---------------- ตารางค่าต่อคำถาม ----------------
    emit("## ค่าที่ได้ต่อคำถาม")
    emit()
    emit("`logits` = ก่อน sigmoid | `p` = หลัง sigmoid (Eq.4) | `y` = หลัง threshold (Alg.1)")
    emit()
    hdr = " | ".join(f"p_{SHORT[n]}" for n in EXPERT_NAMES)
    emit(f"| # | question_type | split | {hdr} | y | expert ที่เปิด | ตรง prior? |")
    emit("|--:|---|:--:|--:|--:|--:|--:|:--:|---|:--:|")

    rows = []
    n_show = min(args.rows, len(records))
    prior_vecs = np.zeros_like(y)
    for i, r in enumerate(records):
        prior = ml.default_prior(r["question_type"])
        prior_vecs[i] = [1 if n in prior else 0 for n in EXPERT_NAMES]
        active = [n for n, v in zip(EXPERT_NAMES, y[i]) if v == 1]
        match = np.array_equal(y[i], prior_vecs[i])
        if i < n_show:
            ps = " | ".join(f"{p[i][j]:.3f}" for j in range(len(EXPERT_NAMES)))
            emit(f"| {i} | {r['question_type']} | {split[i]} | {ps} | `{vec_of(y[i])}` | "
                 f"{', '.join(SHORT[n] for n in active) or '—'} | {'✓' if match else '✗'} |")
        rows.append({
            "idx": i, "image_key": r["image_key"], "question": r["question"],
            "question_type": r["question_type"], "split": split[i],
            "reference_image": Path(ref_paths[i]).name,
            **{f"logit_{SHORT[n]}": f"{logits[i][j]:.6f}" for j, n in enumerate(EXPERT_NAMES)},
            **{f"p_{SHORT[n]}": f"{p[i][j]:.6f}" for j, n in enumerate(EXPERT_NAMES)},
            **{f"margin_{SHORT[n]}": f"{margin[i][j]:+.6f}" for j, n in enumerate(EXPERT_NAMES)},
            **{f"y_{SHORT[n]}": int(y[i][j]) for j, n in enumerate(EXPERT_NAMES)},
            "vector": vec_of(y[i]), "active_experts": "|".join(active),
            "prior_vector": vec_of(prior_vecs[i]), "match_prior": int(match),
        })
    if len(records) > n_show:
        emit(f"| ... | *(แสดง {n_show} แถวแรกจาก {len(records)} — ดูครบใน CSV)* | | | | | | | | |")
    emit()

    # ---------------- สถิติรวม ----------------
    emit("---")
    emit()
    emit("## สรุปค่าทางสถิติของ `p`")
    emit()
    emit("| expert | mean | std | min | max | อัตราเปิด | ใกล้ τ (\\|p−τ\\|<0.05) |")
    emit("|---|--:|--:|--:|--:|--:|--:|")
    for j, n in enumerate(EXPERT_NAMES):
        col = p[:, j]
        emit(f"| {n} | {col.mean():.4f} | {col.std():.4f} | {col.min():.4f} | {col.max():.4f} | "
             f"{y[:, j].mean() * 100:.1f}% | {int(np.sum(np.abs(col - tau) < 0.05))} |")
    emit()
    emit("### การกระจายของ `p` (รวมทุก expert)")
    emit()
    emit("| ช่วง | | จำนวน |")
    emit("|---|---|--:|")
    for line in text_histogram(p.reshape(-1), tau):
        emit(f"| {line} |")
    emit()

    vecs = Counter(vec_of(row) for row in y)
    emit(f"### จำนวนชุด expert ที่ต่างกันที่ router ผลิตได้จริง: **{len(vecs)}**")
    emit()
    emit("| ชุด | expert | จำนวน | สัดส่วน |")
    emit("|:--:|---|--:|--:|")
    for vec, c in vecs.most_common():
        names = ", ".join(SHORT[n] for n, ch in zip(EXPERT_NAMES, vec) if ch == "1") or "—"
        emit(f"| `{vec}` | {names} | {c} | {c / len(records) * 100:.1f}% |")
    emit()

    # ---------------- เทียบ prior + F1 ----------------
    emit("### เทียบกับ `HEURISTIC_PRIORS` (ชุด label ที่ใช้เทรน)")
    emit()
    emit(f"- ตรงกันทั้งเวกเตอร์: **{np.mean(np.all(y == prior_vecs, axis=1)) * 100:.1f}%**")
    emit()
    masks = [("ทั้งหมด", np.ones(len(records), dtype=bool))]
    for s in ("train", "eval", "unknown"):
        m = split == s
        if m.sum():
            masks.append((s, m))
    emit("| expert | " + " | ".join(f"F1 ({nm})" for nm, _ in masks) + " |")
    emit("|---" + "|--:" * len(masks) + "|")
    f1s = [per_expert_f1(prior_vecs[m], y[m]) for _, m in masks]
    for j, n in enumerate(EXPERT_NAMES):
        emit(f"| {n} | " + " | ".join(f"{f[j]:.3f}" for f in f1s) + " |")
    emit("| **macro** | " + " | ".join(f"**{np.mean(f):.3f}**" for f in f1s) + " |")
    emit()
    emit("> **ระวังการตีความ:** `HEURISTIC_PRIORS` ไม่ตรงกับพฤติกรรมจริงของ `expert_generator` "
         "3/6 type (Defect Localization, Defect Analysis, Anomaly Discrimination) ตัวเลข F1 นี้จึงบอกว่า "
         "router เลียนแบบ **label ที่ใช้เทรน** ได้ดีแค่ไหน **ไม่ใช่** ว่าตรงกับ baseline จริงแค่ไหน")
    emit()

    emit("### แยกตาม question_type")
    emit()
    emit("| question_type | n | ชุดที่พบบ่อยสุด | #ชุดที่ต่างกัน |")
    emit("|---|--:|---|--:|")
    by_type = defaultdict(list)
    for i, r in enumerate(records):
        by_type[r["question_type"]].append(vec_of(y[i]))
    for qt in sorted(by_type):
        c = Counter(by_type[qt])
        top, n_top = c.most_common(1)[0]
        names = ", ".join(SHORT[n] for n, ch in zip(EXPERT_NAMES, top) if ch == "1") or "—"
        emit(f"| {qt} | {len(by_type[qt])} | `{top}` = {names} ({n_top}) | {len(c)} |")
    emit()

    # ---------------- ข้อจำกัด ----------------
    emit("---")
    emit()
    emit("## ข้อจำกัดที่ต้องระบุใน report")
    emit()
    emit(f"1. **τ = {tau:.3f} ปรับบนชุด eval ไม่ใช่ validation set แยก** — paper (Algorithm 1) ระบุว่า τ "
         f"เป็นค่าเดียวร่วมกันทุก expert ที่ optimize ไว้ล่วงหน้าบน validation set ซึ่งค่านี้ก็ได้มาแบบนั้น "
         f"(grid-search 33 จุดใน 0.1–0.9) **แต่** โน้ตบุ๊กแบ่งข้อมูล 70/30 สองทางเท่านั้น "
         f"`select_threshold()` จึงเลือก τ บน `~is_train` ซึ่งเป็นชุดเดียวกับที่รายงานผล "
         f"→ **ตัวเลข eval มี optimistic bias** ถ้าจะเคลมตาม paper ต้องแบ่ง train/val/test สามทาง")
    emit("2. **router ถูกเทรนให้ทำนาย `HEURISTIC_PRIORS`** ดังนั้น F1 สูง ๆ แปลว่าเลียนแบบกฎ hardcode "
         "ได้ดี ไม่ได้แปลว่าเลือก expert ได้ดีกว่ากฎนั้น")
    emit("3. **`train`/`eval` กู้มาจาก `features_real.npz` โดยจับคู่ตามตำแหน่งแถว** "
         "(ตรวจด้วยการเทียบ `V_fuse` แล้ว) ถ้า dataset หรือลำดับเปลี่ยน การจับคู่จะไม่ผ่าน "
         "และสคริปต์จะตั้ง split เป็น `unknown` แทนการเดา")
    emit("4. เทียบกับ [print_routing_baseline.py](Router_Network/print_routing_baseline.py) ได้แบบแถวต่อแถว "
         "เมื่อใช้ `--limit-images` ค่าเดียวกัน เพราะทั้งสองไฟล์วน `DS-MVTec.json` ลำดับเดียวกัน")
    emit()

    # ---------------- เขียนไฟล์ ----------------
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_md, out_csv = outdir / "routing_values_router.md", outdir / "routing_values_router.csv"
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
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
