"""moxpert_lite — ตัวช่วยแบบ self-contained สำหรับเสียบ Router Network เข้า pipeline

รวมชิ้นส่วนที่จำเป็นจาก branch `test1` (moxpert/experts.py, encoder.py, router.py)
มาไว้ในไฟล์เดียว เพื่อให้ `Qwen2VL_router.py` เรียกใช้ได้ทั้งบน macOS และ Google Colab
โดยไม่ต้อง merge ทั้งแพ็กเกจ

ต้องการเพียง: numpy, torch, และ clip (openai CLIP)  —  ไม่ผูกกับ faiss/json
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = object


# ==========================================================================
# 1) นิยาม Expert (ลำดับต้องตรงกับ output ของ Router)
# ==========================================================================
REFERENCE_EXTRACTOR = "Reference Extractor"
KNOWLEDGE_GUIDE = "Knowledge Guide"
REASONING_EXPERT = "Reasoning Expert"
DECISION_MAKER = "Decision Maker"
EXPERT_NAMES: List[str] = [REFERENCE_EXTRACTOR, KNOWLEDGE_GUIDE, REASONING_EXPERT, DECISION_MAKER]
N_EXPERTS = len(EXPERT_NAMES)


def activation_vector(active: Iterable[str]) -> np.ndarray:
    """แปลงรายชื่อ expert ที่เปิด -> เวกเตอร์ 0/1 ยาว 4 ตามลำดับ EXPERT_NAMES"""
    active = set(active)
    return np.array([1.0 if n in active else 0.0 for n in EXPERT_NAMES], dtype=np.float64)


# --- กฎ heuristic: question_type -> ชุด expert ที่ "ควร" เปิด -------------
# (พอร์ตจาก moxpert/labeling.py ; Decision Maker เปิดเสมอ)
# ใช้เทียบว่า router เลือกตรงกับ heuristic ของ type นั้นไหม (สำหรับ prompt-parity)
HEURISTIC_PRIORS = {
    "Anomaly Detection":      [REFERENCE_EXTRACTOR, DECISION_MAKER],
    "Anomaly Discrimination": [REFERENCE_EXTRACTOR, DECISION_MAKER],
    "Defect Classification":  [REFERENCE_EXTRACTOR, KNOWLEDGE_GUIDE, DECISION_MAKER],
    "Defect Localization":    [REFERENCE_EXTRACTOR, DECISION_MAKER],
    "Defect Description":     [REFERENCE_EXTRACTOR, KNOWLEDGE_GUIDE, REASONING_EXPERT, DECISION_MAKER],
    "Defect Analysis":        [KNOWLEDGE_GUIDE, REASONING_EXPERT, DECISION_MAKER],
}
DEFAULT_PRIOR = [REFERENCE_EXTRACTOR, DECISION_MAKER]   # ใช้เมื่อไม่รู้จัก question_type


def default_prior(question_type: str) -> List[str]:
    """คืนชุด expert ตามกฎ heuristic ของ question_type (fallback = DEFAULT_PRIOR)"""
    return HEURISTIC_PRIORS.get(question_type, DEFAULT_PRIOR)


# --- Decision threshold τ (แปลง p -> y) -----------------------------------
# ตาม paper (Algorithm 1): ใช้ τ "ค่าเดียวร่วมกันทุก expert" เป็นเกณฑ์กลาง และเป็นค่าที่
# ปรับหาไว้ล่วงหน้า (predefined) จาก validation set — นี่คือพฤติกรรม default ที่ใช้จริง
# หมายเหตุ: การส่ง τ เป็นเวกเตอร์แยกราย expert เป็น "ส่วนขยายนอก paper" (เตรียมไว้สำหรับ
# งาน SHAP ภายหลัง) ไม่ใช่พฤติกรรมมาตรฐานของ MoXpert
def _as_tau_vector(tau) -> np.ndarray:
    """รับ τ เป็น scalar (ตาม paper) หรือเวกเตอร์ยาว 4 (extension) -> เวกเตอร์ยาว N_EXPERTS"""
    if torch is not None and isinstance(tau, torch.Tensor):
        tau = tau.detach().cpu().numpy()
    tau = np.asarray(tau, dtype=np.float64)
    if tau.ndim == 0:
        tau = np.full(N_EXPERTS, float(tau))   # scalar -> ใช้เกณฑ์เดียวกันทุก expert (paper)
    if tau.shape != (N_EXPERTS,):
        raise ValueError(f"tau ต้องเป็น scalar หรือ shape ({N_EXPERTS},) แต่ได้ {tau.shape}")
    return tau


def apply_threshold(probs, tau=0.5) -> np.ndarray:
    """แปลงความน่าจะเป็น p_i -> การตัดสินใจ y_i : y_i = 1 ถ้า p_i > τ ไม่งั้น 0

    ใช้ strictly greater (>) ตาม Algorithm 1 ของ paper
    """
    if torch is not None and isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    probs = np.asarray(probs, dtype=np.float64)
    return (probs > _as_tau_vector(tau)).astype(int)


def experts_from_vector(vec, tau=0.5) -> List[str]:
    """คืนรายชื่อ expert ที่ถูกเปิด (y_i == 1) หลัง threshold"""
    y = apply_threshold(np.asarray(vec, dtype=np.float64).reshape(-1), tau)
    return [name for name, v in zip(EXPERT_NAMES, y) if v == 1]


# ==========================================================================
# 2) การประกอบ prompt จากชุด expert (แทน expert_generator เดิมที่ผูกกับ question_type)
# ==========================================================================
@dataclass
class Query:
    """ข้อมูล 1 คำถามสำหรับสร้าง prompt"""
    query_image_path: str
    question: str
    options_text: str
    object_name: str
    question_type: str
    reference_image_path: Optional[str] = None
    domain_knowledge: Optional[str] = None


# โครง Chain-of-Thought ที่ Reasoning Expert เพิ่มเข้าไป
_COT_SCAFFOLD = (
    "Let's approach this systematically:\n"
    "1. **Observe** the normal sample's key characteristics (color, texture, shape, etc.).\n"
    "2. **Compare** these features to those in the second image, noting any visible differences.\n"
    "3. **Decide** based on these differences which option best answers the question.\n"
)


def build_expert_prompt(active: Iterable[str], query: Query) -> List[dict]:
    """ประกอบ messages payload ของ Qwen2-VL จากชุด expert ที่ Router เปิด

    - Reference Extractor  -> ใส่รูป reference (normal) เป็นภาพแรก
    - Knowledge Guide      -> ใส่ domain knowledge เป็นข้อความ
    - Reasoning Expert     -> ใส่โครง CoT
    - Decision Maker       -> บังคับให้ตอบเป็นตัวอักษรเดียว (เปิดเสมอ)
    """
    active = set(active)
    content: List[dict] = []
    text_parts: List[str] = []

    # ภาพ reference (ถ้าเปิด Reference Extractor และมี path)
    use_reference = REFERENCE_EXTRACTOR in active and query.reference_image_path
    if use_reference:
        content.append({"type": "image", "image": query.reference_image_path})
        text_parts.append(
            "The first image is a normal reference sample. Use it as a baseline to answer "
            "the question about the query image."
        )
    # ภาพ query เสมอ
    content.append({"type": "image", "image": query.query_image_path})

    text_parts.append(f"Question: {query.question}")
    text_parts.append(f"Options:\n{query.options_text}")

    # domain knowledge (ถ้าเปิด Knowledge Guide)
    if KNOWLEDGE_GUIDE in active and query.domain_knowledge:
        text_parts.append(
            "Relevant domain knowledge (defect characteristics and tolerances):\n"
            f"{query.domain_knowledge}"
        )
    # โครง CoT (ถ้าเปิด Reasoning Expert)
    if REASONING_EXPERT in active:
        text_parts.append(_COT_SCAFFOLD)

    # Decision Maker: บังคับตอบตัวอักษรเดียว
    text_parts.append("Please respond with the letter of the correct option only.")

    content.append({"type": "text", "text": "\n".join(text_parts)})
    return [{"role": "user", "content": content}]


# ==========================================================================
# 3) Router Network (Eq. 4)  p = sigmoid(MLP(V_fuse))  + ตัวโหลด checkpoint
# ==========================================================================
class RouterMLP(nn.Module):
    """MLP หลายป้ายกำกับ: V_fuse (1152) -> p (N) โดย p = sigmoid(MLP(V_fuse))"""

    def __init__(self, in_dim: int = 1152, n_experts: int = N_EXPERTS,
                 hidden=(512, 256), dropout: float = 0.2) -> None:
        if torch is None:
            raise ImportError("ต้องมี PyTorch")
        super().__init__()
        self.in_dim = int(in_dim)
        self.n_experts = int(n_experts)
        self.hidden = tuple(hidden)
        self.dropout = float(dropout)
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_experts))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.net(x))

    def predict_proba(self, v_fuse: np.ndarray) -> np.ndarray:
        """numpy V_fuse -> ความน่าจะเป็น (โหมด eval, ปิด dropout -> deterministic)"""
        self.eval()
        device = next(self.parameters()).device
        x = torch.as_tensor(np.atleast_2d(v_fuse), dtype=torch.float32, device=device)
        with torch.no_grad():
            return self.forward(x).cpu().numpy()


def load_router(ckpt_path: str, device: str = "cpu"):
    """โหลด router_real.pt (จากเฟส A) -> (model, threshold)"""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = RouterMLP(in_dim=cfg["in_dim"], n_experts=cfg["n_experts"],
                      hidden=tuple(cfg["hidden"]), dropout=cfg["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, float(ckpt.get("threshold", 0.5))


# ==========================================================================
# 4) CLIP encoder (Eq. 1-3): รูป/ข้อความ -> V_fuse (1152)  + ฟีเจอร์ดิบ 512 สำหรับ FAISS
# ==========================================================================
CLIP_NATIVE_DIM = 512   # ขนาด embedding ของ CLIP ViT-B/16
PAPER_DIM = 576         # d ต่อ modality ตาม paper


class MoXpertEncoder:
    """CLIP ViT-B/16 (แช่แข็ง) + projection 512->576 แบบสุ่มคงที่ -> V_fuse (1152)

    ต้องใช้ proj_seed เดียวกับตอนเทรน router (ค่า default 1234) เพื่อให้ฟีเจอร์ตรงกัน
    """

    def __init__(self, clip_model_name="ViT-B/16", target_dim=PAPER_DIM,
                 device="cpu", proj_seed=1234):
        import clip
        self._clip = clip
        self.device = device
        self.model, self.preprocess = clip.load(clip_model_name, device=device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.native_dim = int(self.model.visual.output_dim)
        self.target_dim = int(target_dim)
        g = torch.Generator().manual_seed(proj_seed)   # โปรเจกชันคงที่ (deterministic)
        self.img_proj = self._make_projection(self.native_dim, self.target_dim, g)
        self.text_proj = self._make_projection(self.native_dim, self.target_dim, g)

    def _make_projection(self, in_dim, out_dim, gen):
        if in_dim == out_dim:
            return nn.Identity()
        proj = nn.Linear(in_dim, out_dim, bias=False)
        with torch.no_grad():
            w = torch.empty(out_dim, in_dim)
            w.normal_(0.0, 1.0 / np.sqrt(in_dim), generator=gen)
            proj.weight.copy_(w)
        for p in proj.parameters():
            p.requires_grad_(False)
        return proj.to(self.device)

    def _to_pil(self, image):
        from PIL import Image
        if hasattr(image, "size") and not hasattr(image, "ndim"):
            return image
        if isinstance(image, (str, bytes)):
            return Image.open(image).convert("RGB")
        return image

    def encode_image_raw(self, image) -> np.ndarray:
        """ฟีเจอร์ดิบ 512-d (L2-normalized) — พื้นที่เดียวกับ FAISS index"""
        x = self.preprocess(self._to_pil(image)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model.encode_image(x)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.float().cpu().numpy().reshape(-1)

    def encode_image(self, image) -> np.ndarray:
        """V_img หลัง projection (576-d) — พื้นที่ของ router"""
        x = self.preprocess(self._to_pil(image)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model.encode_image(x)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            v = self.img_proj(feat.float())
        return v.detach().cpu().numpy().reshape(-1)

    def encode_text(self, text: str) -> np.ndarray:
        """V_text หลัง projection (576-d)"""
        tokens = self._clip.tokenize([text], truncate=True).to(self.device)
        with torch.no_grad():
            feat = self.model.encode_text(tokens)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            v = self.text_proj(feat.float())
        return v.detach().cpu().numpy().reshape(-1)

    @staticmethod
    def fuse(v_img, v_text) -> np.ndarray:
        """V_fuse = [V_img ; V_text] (Eq. 3)"""
        return np.concatenate([np.asarray(v_img).reshape(-1), np.asarray(v_text).reshape(-1)])

    def encode_vfuse(self, image, text: str) -> np.ndarray:
        """สะดวก: รูป+ข้อความ -> V_fuse (1152)"""
        return self.fuse(self.encode_image(image), self.encode_text(text))
