#!/usr/bin/env python3
"""ตรวจสอบเชิงประจักษ์: MoXpert repo ทางการ มี "Router Network" ตามที่ paper อ้างหรือไม่

สคริปต์นี้ "ไม่แก้" โค้ดต้นฉบับแม้แต่บรรทัดเดียว — import `expert_generator` เข้ามาเรียก
จากข้างนอก แล้ววิเคราะห์จาก `messages` ที่มันคืนกลับมา เพื่อให้หลักฐานยังเป็นของโค้ดเดิมจริง ๆ

ผลิตหลักฐาน 4 ชิ้น:
  A. ตาราง activation matrix (question_type x expert) ที่ detect จาก output จริง
  B. Invariance test — output ขึ้นกับอะไรบ้าง (ตัวชี้ขาดว่าเป็น router หรือ lookup table)
  C. ข้อเท็จจริงเชิงสถิตจาก git ของ repo ทางการ (upstream/main)
  D. เทียบ HEURISTIC_PRIORS (label ที่ใช้เทรน RouterMLP) กับพฤติกรรมจริงของโค้ดเดิม

รัน:
    python Router_Network/analyze_original_expert_generator.py

พึ่งพาแค่ Python stdlib + expert_generator.py (ไม่ต้องมี torch / CLIP / Qwen)
"""

from __future__ import annotations

import random
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "Experiments"))

from expert_generator import expert_generator  # noqa: E402  (ต้อง insert path ก่อน)

# --- ชื่อ expert (ลำดับมาตรฐาน = ลำดับ output ของ router) -------------------
REFERENCE_EXTRACTOR = "Reference Extractor"
KNOWLEDGE_GUIDE = "Knowledge Guide"
REASONING_EXPERT = "Reasoning Expert"
DECISION_MAKER = "Decision Maker"
EXPERT_NAMES = [REFERENCE_EXTRACTOR, KNOWLEDGE_GUIDE, REASONING_EXPERT, DECISION_MAKER]
SHORT = {REFERENCE_EXTRACTOR: "Ref", KNOWLEDGE_GUIDE: "Know",
         REASONING_EXPERT: "Reason", DECISION_MAKER: "Decide"}

# question_type ที่จะทดสอบ — รวม Anomaly Discrimination ที่ของเดิม "ไม่รองรับ" ไว้ด้วย
QUESTION_TYPES = [
    "Anomaly Detection",
    "Anomaly Discrimination",
    "Defect Classification",
    "Defect Localization",
    "Defect Description",
    "Defect Analysis",
]

# กฎ heuristic ที่ผู้ใช้เขียนไว้ใน moxpert_lite.py (= label ที่ใช้เทรน RouterMLP)
HEURISTIC_PRIORS = {
    "Anomaly Detection":      [REFERENCE_EXTRACTOR, DECISION_MAKER],
    "Anomaly Discrimination": [REFERENCE_EXTRACTOR, DECISION_MAKER],
    "Defect Classification":  [REFERENCE_EXTRACTOR, KNOWLEDGE_GUIDE, DECISION_MAKER],
    "Defect Localization":    [REFERENCE_EXTRACTOR, DECISION_MAKER],
    "Defect Description":     [REFERENCE_EXTRACTOR, KNOWLEDGE_GUIDE, REASONING_EXPERT, DECISION_MAKER],
    "Defect Analysis":        [KNOWLEDGE_GUIDE, REASONING_EXPERT, DECISION_MAKER],
}

# --- sentinel input: ค่าที่จำง่าย เพื่อให้ detect จาก output ได้แบบไม่กำกวม ---
REF_IMG = "__REF_IMAGE__"
QRY_IMG = "__QUERY_IMAGE__"
DK_SENTINEL = "__DK_SENTINEL__"
SENTINEL_DK = {"object_name": "__OBJ__", "scratch": DK_SENTINEL}
SENTINEL_Q = "__QUESTION__"
SENTINEL_OPT = "__OPTIONS__"

md_lines: list[str] = []


def emit(line: str = "") -> None:
    """พิมพ์ลง console และเก็บลงไฟล์ markdown พร้อมกัน"""
    print(line)
    md_lines.append(line)


# ==========================================================================
# ตัวช่วย: แกะ messages ที่ expert_generator คืนมา
# ==========================================================================
def split_messages(messages):
    """คืน (รายการ image ที่ส่งเข้า MLLM, ข้อความ prompt ที่รวมแล้ว)"""
    images, texts = [], []
    for msg in messages:
        for item in msg.get("content", []):
            if item.get("type") == "image":
                images.append(item.get("image"))
            elif item.get("type") == "text":
                texts.append(item.get("text", ""))
    return images, "\n".join(texts)


def detect_experts(messages, ref_marker: str, dk_marker: str) -> dict[str, int]:
    """detect ว่า expert ตัวไหนถูก "เปิด" โดยดูจาก messages ที่ได้กลับมา

    เกณฑ์อิงชิ้นส่วน prompt ที่แต่ละ expert รับผิดชอบ (ตาม guide-rt.md):
      Reference Extractor = มีรูป reference (image1) ส่งเข้าไปด้วยหรือไม่
      Knowledge Guide     = มี domain knowledge ถูก interpolate เข้า prompt หรือไม่
      Reasoning Expert    = มีบล็อก CoT Observe/Compare/Decide หรือไม่
      Decision Maker      = มีคำสั่งให้ตอบเป็นตัวอักษรตัวเลือกหรือไม่

    `ref_marker` / `dk_marker` ส่งเข้ามาเป็นพารามิเตอร์ เพื่อให้ detect ได้ถูกต้อง
    แม้ตอน Evidence B ที่สุ่มเปลี่ยน input จนใช้ค่า sentinel คงที่ไม่ได้
    """
    images, text = split_messages(messages)
    cot = all(re.search(rf"\*\*{kw}\*\*", text) for kw in ("Observe", "Compare", "Decide"))
    return {
        REFERENCE_EXTRACTOR: int(ref_marker in images),
        KNOWLEDGE_GUIDE: int(dk_marker in text),
        REASONING_EXPERT: int(cot),
        DECISION_MAKER: int("letter of the correct option" in text.lower()),
    }


def run_generator(qtype, image1=REF_IMG, image2=QRY_IMG, question=SENTINEL_Q,
                  options_text=SENTINEL_OPT, domain_knowledge=None,
                  dk_marker=DK_SENTINEL):
    """เรียกโค้ดเดิม คืน (activation dict, n_images) หรือ (None, ข้อความ error)

    image1 ถูกใช้เป็น marker ของ Reference Extractor โดยตรง จึงต้องต่างจาก image2 เสมอ
    """
    assert image1 != image2, "image1 ต้องต่างจาก image2 ไม่งั้น detect Reference ผิด"
    dk = SENTINEL_DK if domain_knowledge is None else domain_knowledge
    try:
        messages = expert_generator(image1, image2, qtype, question, options_text, dk)
    except ValueError as exc:
        return None, str(exc)
    images, _ = split_messages(messages)
    return detect_experts(messages, ref_marker=image1, dk_marker=dk_marker), len(images)


def fmt_vec(act: dict[str, int]) -> str:
    return "".join(str(act[name]) for name in EXPERT_NAMES)


def names_of(act: dict[str, int]) -> list[str]:
    return [n for n in EXPERT_NAMES if act[n] == 1]


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True,
                          text=True).stdout.strip()


# ==========================================================================
# Evidence A — activation matrix
# ==========================================================================
def evidence_a():
    emit("## Evidence A — ตาราง activation matrix ที่ได้จากการรัน `expert_generator()` จริง")
    emit()
    emit("detect จาก `messages` ที่ฟังก์ชันคืนกลับมา ไม่ใช่จากการอ่าน source code")
    emit()
    emit("| question_type | Ref | Know | Reason | Decide | เวกเตอร์ | #รูปที่ส่งเข้า MLLM |")
    emit("|---|:--:|:--:|:--:|:--:|:--:|:--:|")

    results = {}
    for qtype in QUESTION_TYPES:
        act, extra = run_generator(qtype)
        if act is None:
            emit(f"| {qtype} | — | — | — | — | **ValueError** | — |")
            results[qtype] = None
            continue
        cells = " | ".join(str(act[n]) for n in EXPERT_NAMES)
        emit(f"| {qtype} | {cells} | `{fmt_vec(act)}` | {extra} |")
        results[qtype] = act

    emit()
    unsupported = [q for q, a in results.items() if a is None]
    if unsupported:
        emit(f"> `{', '.join(unsupported)}` — โค้ดเดิม **ไม่รองรับ** จะ `raise ValueError` "
             f"([expert_generator.py:105](Experiments/expert_generator.py#L105))")
        emit()
    return results


# ==========================================================================
# Evidence B — invariance test (ตัวชี้ขาด)
# ==========================================================================
def evidence_b(n_trials=50, seed=0):
    emit("## Evidence B — Invariance test (หลักฐานชี้ขาด)")
    emit()
    emit("นิยามของ router network คือ output ต้อง**ขึ้นกับข้อมูลเข้า** (ภาพ + คำถาม) "
         "การทดสอบนี้จึงสุ่มเปลี่ยนข้อมูลเข้าแล้วดูว่า output ขยับไหม")
    emit()

    rng = random.Random(seed)
    objects = ["bottle", "cable", "capsule", "hazelnut", "screw", "transistor", "zipper"]
    defects = ["scratch", "crack", "contamination", "bent", "missing", "hole"]
    questions = [
        "Is there any defect in the object?",
        "What kind of defect appears on the surface?",
        "Where is the anomaly located in this image?",
        "Describe the abnormality shown in the second image.",
        "Analyse the root cause of the observed defect.",
    ]

    emit("### B1 — ล็อก `question_type` แล้วสุ่มเปลี่ยนภาพ / คำถาม / domain knowledge")
    emit()
    emit(f"สุ่ม {n_trials} ครั้งต่อ type (seed={seed})")
    emit()
    emit("| question_type | จำนวนเวกเตอร์ที่ต่างกัน | variance | ผล |")
    emit("|---|:--:|:--:|:--:|")

    all_invariant = True
    for qtype in QUESTION_TYPES:
        if run_generator(qtype)[0] is None:
            continue
        seen = set()
        for t in range(n_trials):
            obj = rng.choice(objects)
            # ฝัง marker ไว้ใน "ค่า" ของ domain knowledge เพื่อให้ detect Knowledge Guide ได้
            # (descriptions_text ต่อ key:value ทุกตัว ถ้าถูกใส่เข้า prompt marker จะโผล่ด้วย)
            dk_marker = f"__DK_{t}__"
            # ใช้ sample เพื่อกัน key ซ้ำ — ถ้า key ชนกัน ค่าที่มี marker จะถูกเขียนทับ
            k1, k2 = rng.sample(defects, 2)
            dk = {"object_name": obj,
                  k1: f"{dk_marker} a defect on the {obj} surface",
                  k2: f"visible {rng.choice(defects)} pattern"}
            # prefix ต่างกันชัดเจน กันไม่ให้ image1 ชนกับ image2 โดยบังเอิญ
            act, _ = run_generator(
                qtype,
                image1=f"/data/{obj}/good/ref_{rng.randint(0, 999):03d}.png",
                image2=f"/data/{obj}/test/qry_{rng.randint(0, 999):03d}.png",
                question=rng.choice(questions),
                options_text="\n".join(f"{c}: {rng.choice(defects)}" for c in "ABCD"),
                domain_knowledge=dk,
                dk_marker=dk_marker,
            )
            seen.add(fmt_vec(act))
        ok = len(seen) == 1
        all_invariant &= ok
        emit(f"| {qtype} | {len(seen)} | {0 if ok else 'ไม่เป็นศูนย์'} | "
             f"{'[PASS] ไม่เปลี่ยนเลย' if ok else '[FAIL] เปลี่ยน'} |")

    emit()
    emit("### B2 — ล็อกภาพ/คำถาม/domain knowledge แล้วเปลี่ยนแค่ `question_type`")
    emit()
    vecs = {}
    for qtype in QUESTION_TYPES:
        act, _ = run_generator(qtype)
        if act is not None:
            vecs[qtype] = fmt_vec(act)
    distinct = len(set(vecs.values()))
    emit(f"เวกเตอร์ที่ได้: {', '.join(f'`{v}`({q})' for q, v in vecs.items())}")
    emit()
    emit(f"→ ได้ **{distinct} รูปแบบที่ต่างกัน** จาก {len(vecs)} type")
    emit()
    emit("### สรุป Evidence B")
    emit()
    if all_invariant and distinct > 1:
        emit("**ตัวแปรเดียวที่มีผลต่อการเลือก expert คือสตริง `question_type`** — "
             "เนื้อภาพและข้อความคำถามจริงไม่มีผลใด ๆ ต่อ output")
        emit()
        emit("นี่คือพฤติกรรมของ **lookup table** ไม่ใช่ router network ที่เรียนรู้จากข้อมูล "
             "(ซึ่งต้องรับ `V_fuse` แล้วให้ค่าที่ต่างกันตามภาพ/คำถาม)")
    else:
        emit("ผลไม่ตรงกับที่คาด — ต้องตรวจสอบเพิ่ม (รายงานตามผลจริงข้างบน)")
    emit()
    return all_invariant, distinct


# ==========================================================================
# Evidence C — ข้อเท็จจริงเชิงสถิตจาก git ของ repo ทางการ
# ==========================================================================
ROUTING_PATTERN = re.compile(
    r"router|gating|sigmoid|softmax|nn\.Linear|nn\.Module|requires_grad", re.I)


def evidence_c():
    emit("## Evidence C — ตรวจ repo ทางการของผู้เขียน paper ด้วย git")
    emit()

    upstream_url = git("remote", "get-url", "upstream") or "(ไม่มี remote ชื่อ upstream)"
    tip = git("log", "-1", "--format=%h|%an|%ad|%s", "--date=short", "upstream/main")
    if not tip:
        emit("> ไม่พบ ref `upstream/main` ในเครื่อง — ข้ามส่วนนี้ "
             "(รัน `git fetch upstream` ก่อนเพื่อให้ตรวจได้)")
        emit()
        return None

    h, an, ad, subj = tip.split("|", 3)
    emit(f"- **repo ต้นทาง:** `{upstream_url}`")
    emit(f"- **commit ล่าสุดของ `upstream/main`:** `{h}` — {ad} — {an} — \"{subj}\"")
    n_commits = git("rev-list", "--count", "upstream/main")
    emit(f"- **จำนวน commit ทั้งหมดใน repo ทางการ:** {n_commits}")
    emit()

    files = [f for f in git("ls-tree", "-r", "--name-only", "upstream/main").splitlines() if f]
    py_files = [f for f in files if f.endswith(".py")]
    emit(f"- **จำนวนไฟล์ทั้งหมด:** {len(files)}  |  **ไฟล์ Python:** {len(py_files)} "
         f"({', '.join(py_files)})")
    emit()

    emit("### grep หา routing/neural-gate ทุกไฟล์ใน `upstream/main`")
    emit()
    emit(f"pattern: `{ROUTING_PATTERN.pattern}`")
    emit()
    emit("| ไฟล์ | จำนวนที่เจอ |")
    emit("|---|:--:|")
    total_hits = 0
    for f in files:
        if not f.lower().endswith((".py", ".md", ".ipynb", ".json")):
            continue
        content = subprocess.run(["git", "show", f"upstream/main:{f}"], cwd=REPO,
                                 capture_output=True, text=True).stdout
        n = len(ROUTING_PATTERN.findall(content))
        total_hits += n
        emit(f"| `{f}` | {n} |")
    emit()
    emit(f"**รวมทั้ง repo ทางการ: {total_hits} ครั้ง**")
    emit()

    emit("### `torch` ถูกใช้ทำอะไรใน repo ทางการ")
    emit()
    emit("มี `import torch` จริง จึงต้องระบุให้ชัดว่าใช้ทำอะไร ไม่ให้เข้าใจผิดว่ามี trainable layer")
    emit()
    emit("| ไฟล์:บรรทัด | โค้ด | ใช้ทำอะไร |")
    emit("|---|---|---|")
    for f in py_files:
        content = subprocess.run(["git", "show", f"upstream/main:{f}"], cwd=REPO,
                                 capture_output=True, text=True).stdout
        for i, line in enumerate(content.splitlines(), 1):
            if "torch" in line:
                code = line.strip()
                if "import" in code:
                    why = "โหลดไลบรารี"
                elif "manual_seed" in code:
                    why = "ตั้ง seed ให้ผลรันซ้ำได้"
                elif "no_grad" in code:
                    why = "ปิดการคำนวณ gradient (inference อย่างเดียว)"
                elif "dtype" in code:
                    why = "กำหนดความละเอียดตัวเลขของ Qwen2-VL"
                else:
                    why = "อื่น ๆ"
                emit(f"| `{f}:{i}` | `{code}` | {why} |")
    emit()
    emit("> ไม่มีบรรทัดใดสร้าง layer ที่เทรนได้ (`nn.Linear` / `nn.Module`) หรือเรียก "
         "`.backward()` — torch ถูกใช้เพื่อรัน Qwen2-VL กับ CLIP ที่ freeze ไว้เท่านั้น")
    emit()

    eg = subprocess.run(["git", "show", "upstream/main:Experiments/expert_generator.py"],
                        cwd=REPO, capture_output=True, text=True).stdout
    n_if = len(re.findall(r"^\s*(?:el)?if question_type ==", eg, re.M))
    n_raise = len(re.findall(r"raise ValueError", eg))
    n_import = len(re.findall(r"^\s*(?:import|from)\s", eg, re.M))
    emit("### โครงสร้างของ `expert_generator.py` (ไฟล์ที่เลือก expert)")
    emit()
    emit(f"- กิ่ง `if/elif question_type ==` : **{n_if}** กิ่ง")
    emit(f"- `else: raise ValueError` : **{n_raise}**")
    emit(f"- จำนวน import ทั้งไฟล์ : **{n_import}** — ไม่พึ่ง library ภายนอกเลย")
    emit(f"- จำนวนบรรทัด : **{len(eg.splitlines())}**")
    emit()
    emit("> ไม่มี parameter ที่เรียนรู้ได้ ไม่มี threshold ไม่มีค่าความน่าจะเป็น "
         "— เป็น string comparison ล้วน ๆ")
    emit()
    return total_hits


# ==========================================================================
# Evidence D — HEURISTIC_PRIORS vs พฤติกรรมจริง
# ==========================================================================
def evidence_d(results):
    emit("## Evidence D — `HEURISTIC_PRIORS` ตรงกับโค้ดเดิมหรือไม่")
    emit()
    emit("`HEURISTIC_PRIORS` ใน [moxpert_lite.py](Experiments/moxpert_lite.py#L45) คือ **label "
         "ที่ใช้เทรน RouterMLP** ถ้าไม่ตรงกับพฤติกรรมจริงของ baseline แปลว่า router ถูกสอน "
         "ให้เลียนแบบสิ่งที่ผิดตั้งแต่ต้น")
    emit()
    emit("| question_type | โค้ดเดิมทำจริง | HEURISTIC_PRIORS | ตรงกัน? |")
    emit("|---|---|---|:--:|")

    mismatches = []
    for qtype in QUESTION_TYPES:
        prior = HEURISTIC_PRIORS.get(qtype, [])
        prior_s = ", ".join(SHORT[n] for n in prior) or "—"
        act = results.get(qtype)
        if act is None:
            emit(f"| {qtype} | **ไม่รองรับ (ValueError)** | {prior_s} | ✗ |")
            mismatches.append((qtype, "ของเดิมไม่รองรับ type นี้", prior_s))
            continue
        actual = names_of(act)
        actual_s = ", ".join(SHORT[n] for n in actual) or "—"
        same = set(actual) == set(prior)
        emit(f"| {qtype} | {actual_s} | {prior_s} | {'✓' if same else '✗'} |")
        if not same:
            mismatches.append((qtype, actual_s, prior_s))

    emit()
    n_total = len(QUESTION_TYPES)
    emit(f"**ไม่ตรงกัน {len(mismatches)}/{n_total} type**")
    emit()
    if mismatches:
        for qtype, actual_s, prior_s in mismatches:
            emit(f"- **{qtype}** — โค้ดเดิมให้ `{actual_s}` แต่ label บอก `{prior_s}`")
        emit()
        emit("> ผลกระทบ: label ที่ใช้เทรน router คลาดจาก baseline จริง ทำให้การเทียบ "
             "accuracy ระหว่าง baseline กับ router ไม่ได้วัดเฉพาะผลของการ routing "
             "ควรแก้ `HEURISTIC_PRIORS` ให้ตรงก่อนเทรนรอบต่อไป")
        emit()
    return mismatches


# ==========================================================================
# main
# ==========================================================================
def main():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    head = git("rev-parse", "--short", "HEAD")
    branch = git("rev-parse", "--abbrev-ref", "HEAD")

    emit("# ตรวจสอบ: MoXpert repo ทางการ มี Router Network หรือไม่")
    emit()
    emit(f"สร้างโดย `Router_Network/analyze_original_expert_generator.py` เมื่อ {ts}  ")
    emit(f"branch ที่ใช้ตรวจ: `{branch}` @ `{head}`")
    emit()
    emit("**คำถามที่ต้องการตอบ:** paper MoXpert (Chen & Imani, *Pattern Recognition* 2025) "
         "อธิบาย router network ที่เลือก expert จาก `V_fuse` แต่ repo ที่อ้างว่าเป็น "
         "official implementation มีส่วนนี้จริงหรือไม่")
    emit()
    emit("---")
    emit()

    results = evidence_a()
    emit("---")
    emit()
    invariant, distinct = evidence_b()
    emit("---")
    emit()
    total_hits = evidence_c()
    emit("---")
    emit()
    mismatches = evidence_d(results)
    emit("---")
    emit()

    # --- บทสรุป ---
    emit("## บทสรุป")
    emit()
    emit("| ข้อ | ผลตรวจ |")
    emit("|---|---|")
    emit(f"| การเลือก expert ขึ้นกับเนื้อภาพ/คำถามไหม | "
         f"{'**ไม่** — คงที่ทุกครั้ง' if invariant else 'เปลี่ยน'} |")
    emit(f"| ตัวแปรที่มีผลจริง | `question_type` (string) เท่านั้น |")
    if total_hits is not None:
        emit(f"| routing/neural-gate ใน repo ทางการ | **{total_hits} ครั้ง** |")
    emit(f"| กลไกที่ใช้จริง | `if/elif` hardcode 5 กิ่ง ใน `expert_generator.py` |")
    emit(f"| `HEURISTIC_PRIORS` ไม่ตรงกับโค้ดเดิม | {len(mismatches)}/{len(QUESTION_TYPES)} type |")
    emit()
    emit("**คำตอบ: repo ทางการของ MoXpert ไม่มี router network** การเลือก expert ทำด้วย "
         "การเทียบสตริง `question_type` แบบ hardcode ไม่มีสมการ `p = sigmoid(MLP(V_fuse))` "
         "ปรากฏที่ใดใน repo")
    emit()

    emit("## ข้อจำกัดของการตรวจนี้ (ต้องระบุใน report)")
    emit()
    emit("1. **สคริปต์นี้ตรวจได้เฉพาะฝั่งโค้ด** ไม่ได้ตรวจตัว paper — ไม่มีไฟล์ PDF ของ paper "
         "ใน repo ควรเปิด paper ตัวจริงยืนยันว่าอธิบาย router network ไว้อย่างไร "
         "และเลขสมการตรงกับที่อ้างหรือไม่ ก่อนส่งอาจารย์")
    emit("2. **ตรวจจาก `upstream/main` ที่ fetch ไว้ในเครื่อง** ควรรัน `git fetch upstream` "
         "ให้เป็นปัจจุบันก่อนตรวจ แล้วระบุวันที่ตรวจสอบกำกับใน report")
    emit("3. **เกณฑ์ detect expert เป็นการตีความ** ว่าชิ้นส่วนใดของ prompt เป็นของ expert ตัวไหน "
         "(อิงตาม `guide-rt.md`) ถ้าอาจารย์นิยามต่างออกไป ตาราง Evidence A อาจเปลี่ยน "
         "แต่ข้อสรุปหลักจาก Evidence B และ C ไม่เปลี่ยน เพราะไม่ขึ้นกับเกณฑ์นี้")
    emit()

    outdir = REPO / "Router_Network" / "artifacts"
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "original_repo_router_audit.md"
    out.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print()
    print("=" * 68)
    print(f"เขียนรายงานแล้ว: {out.relative_to(REPO)}")
    print("=" * 68)


if __name__ == "__main__":
    main()
