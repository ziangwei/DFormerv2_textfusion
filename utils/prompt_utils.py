# prompt_utils.py (unified)
import os, re, json, math
from pathlib import Path
from typing import List, Sequence, Union, Optional, Tuple

import torch
import torch.nn.functional as F

# --- Optional backends ---
from transformers import AutoModel, AutoTokenizer, CLIPModel

# =========================
# Global cache (models & embeds)
# =========================
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_GLOBAL = {
    "model": None,
    "tokenizer": None,
    "name": None,    # "<encoder_name>:<device>"
    "device": _DEVICE,
}

PROMPT_EMBEDS = None
PROMPT_CACHE = {}
ACTIVE_PROMPT_SET = "train"

# =========================
# Encoder loader
# =========================
def _load_encoder(encoder: str = "jinaclip",
                  encoder_name: Optional[str] = None,
                  device: Optional[torch.device] = None):
    """
    encoder: "clip" | "jinaclip"
    encoder_name:
      - clip:     default "openai/clip-vit-base-patch16"
      - jinaclip: default "jinaai/jina-clip-v2"
    """
    if device is None:
        device = _DEVICE
    if encoder_name is None:
        encoder_name = "openai/clip-vit-base-patch16" if encoder == "clip" else "jinaai/jina-clip-v2"

    cache_key = f"{encoder}:{encoder_name}:{device}"
    if _GLOBAL.get("name") != cache_key:
        if encoder == "clip":
            model = CLIPModel.from_pretrained(encoder_name)
            tok = AutoTokenizer.from_pretrained(encoder_name)
            model.eval().to(device)
            for p in model.parameters(): p.requires_grad = False
            _GLOBAL.update({"model": model, "tokenizer": tok, "name": cache_key, "device": device})
        else:  # jinaclip
            model = AutoModel.from_pretrained(encoder_name, trust_remote_code=True)
            model.eval().to(device)
            for p in model.parameters(): p.requires_grad = False
            _GLOBAL.update({"model": model, "tokenizer": None, "name": cache_key, "device": device})
    return _GLOBAL["model"], _GLOBAL["tokenizer"], _GLOBAL["device"]

def unload_clip_model():
    _GLOBAL.update({"model": None, "tokenizer": None, "name": None})
    torch.cuda.empty_cache()

# =========================
# Encoding primitives
# =========================
def _encode_texts(texts: List[str],
                  encoder: str = "jinaclip",
                  encoder_name: Optional[str] = None,
                  device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Returns [N, D] float tensor (L2-normalized).
    """
    model, tok, device = _load_encoder(encoder, encoder_name, device)
    with torch.no_grad():
        if encoder == "clip":
            inputs = tok(texts, padding=True, truncation=True, return_tensors="pt").to(device)
            feats = model.get_text_features(**inputs)           # [N, D]
        else:
            # Jina-CLIP 提供 encode_text(list[str], truncate_dim=?)
            truncate_dim = int(os.getenv("JINA_CLIP_DIM", "512"))
            feats = model.encode_text(texts, truncate_dim=truncate_dim)
            if not isinstance(feats, torch.Tensor):
                feats = torch.from_numpy(feats).to(device)
    feats = feats.to(device)
    feats = F.normalize(feats, dim=-1)
    return feats

def _postproject(feats: torch.Tensor, target_dim: Optional[int]) -> torch.Tensor:
    if target_dim is None:
        return feats
    D = feats.shape[-1]
    if D == target_dim:
        return feats
    if D > target_dim:
        return feats[..., :target_dim]
    pad = torch.zeros(feats.size(0), target_dim - D, device=feats.device, dtype=feats.dtype)
    return torch.cat([feats, pad], dim=-1)

def encode_prompts(prompts: List[Union[str, List[str]]],
                   encoder: str = "jinaclip",
                   encoder_name: Optional[str] = None,
                   target_dim: Optional[int] = None) -> torch.Tensor:
    """
    输入：
      - List[str]：直接编码 -> [N, D]
      - List[List[str]]：对同一组内句子编码并平均 -> [G, D]
    输出：
      Tensor [N_or_G, target_dim or D]
    """
    if len(prompts) == 0:
        td = target_dim or int(os.getenv("JINA_CLIP_DIM", "512"))
        return torch.zeros(0, td)

    # case 1: List[str]
    if isinstance(prompts[0], str):
        feats = _encode_texts(prompts, encoder, encoder_name)
        return _postproject(feats, target_dim)

    # case 2: List[List[str]] (group average)
    outs = []
    for group in prompts:
        feats = _encode_texts(group, encoder, encoder_name)  # [k, D]
        feats = feats.mean(dim=0, keepdim=True)              # [1, D]
        outs.append(feats)
    feats = torch.cat(outs, dim=0)                           # [G, D]
    return _postproject(feats, target_dim)

def encode_prompt(prompt: str,
                  encoder: str = "jinaclip",
                  encoder_name: Optional[str] = None,
                  target_dim: Optional[int] = None) -> torch.Tensor:
    return encode_prompts([prompt], encoder=encoder, encoder_name=encoder_name, target_dim=target_dim)[0]

# ========= global prompt cache helpers =========
def set_prompt_embeds(text_embeds: torch.Tensor):
    global PROMPT_EMBEDS
    PROMPT_EMBEDS = text_embeds

def register_prompt_embeds(set_name: str, text_embeds: torch.Tensor):
    PROMPT_CACHE[set_name] = text_embeds

def switch_prompt_set(set_name: str):
    global PROMPT_EMBEDS, ACTIVE_PROMPT_SET
    assert set_name in PROMPT_CACHE, f"Prompt set {set_name} not registered!"
    PROMPT_EMBEDS = PROMPT_CACHE[set_name]
    ACTIVE_PROMPT_SET = set_name

# =========================
# Label → prompt variants
# =========================
PROMPT_TEMPLATES_CLIP = [
    "a photo of a {}.",
    "this is a photo of a {}.",
    "an image of a {}.",
]

_LABEL_ALIAS = {
    "refridgerator": "refrigerator",
    "night stand": "nightstand",
    "floor mat": "rug",
    "television": "tv",
    "books": "book",
    "clothes": "clothing",
    "bathtub": "bath tub",
    "bookshelf": "bookcase",
    "shelves": "shelf",
}

def _normalize_label(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return _LABEL_ALIAS.get(s, s)

def _pick_article(noun: str) -> str:
    return "an" if noun[:1] in "aeiou" else "a"

def build_prompt_variants_for_label(label: str,
                                    template_set: str = "clip",
                                    max_templates_per_label: int = 4) -> List[str]:
    lbl = _normalize_label(label)
    if template_set == "clip":
        out = []
        for t in PROMPT_TEMPLATES_CLIP[:max_templates_per_label]:
            if "{}" in t:
                if "a {}" in t:
                    out.append(t.replace("a {}", f"{_pick_article(lbl)} {lbl}"))
                elif "an {}" in t:
                    out.append(t.replace("an {}", f"{_pick_article(lbl)} {lbl}"))
                else:
                    out.append(t.format(lbl))
            else:
                out.append(f"{t} {lbl}")
        return out
    return [lbl]

def build_prompt_groups_from_labels(labels: List[str],
                                    L: int,
                                    template_set: str = "clip",
                                    max_templates_per_label: int = 4) -> List[List[str]]:
    # 去重保序
    seen, norm = set(), []
    for lb in labels:
        nlb = _normalize_label(lb)
        if nlb and nlb not in seen:
            norm.append(nlb); seen.add(nlb)
    PAD = "object"
    if len(norm) >= L: norm = norm[:L]
    else: norm = norm + [PAD] * (L - len(norm))
    return [build_prompt_variants_for_label(lb, template_set, max_templates_per_label) for lb in norm]

def prepare_classbank_prompts(labels_txt_path: str,
                              max_templates_per_label: int = 3,
                              template_set: str = "clip",
                              encoder: str = "jinaclip",
                              encoder_name: Optional[str] = None,
                              target_dim: Optional[int] = None,
                              register_set_name: str = "classbank"):
    labels = [ln.strip() for ln in Path(labels_txt_path).read_text().splitlines() if ln.strip()]
    K = len(labels)
    groups = build_prompt_groups_from_labels(labels, L=K, template_set=template_set,
                                             max_templates_per_label=max_templates_per_label)
    text_feats_KD = encode_prompts(groups, encoder=encoder, encoder_name=encoder_name, target_dim=target_dim).cpu()
    register_prompt_embeds(register_set_name, text_feats_KD)  # (K,D)
    switch_prompt_set(register_set_name)
    unload_clip_model()
    return {"labels": labels, "embeds": text_feats_KD, "K": K, "D": text_feats_KD.shape[-1], "set_name": register_set_name}

# =========================
# Caption sentence Top-K
# =========================
_SENT_SPLIT = re.compile(r"[。！？!?\.]\s+")

def split_into_sentences(text: str, max_sentences: int = 8) -> List[str]:
    if not text: return []
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]
    return parts[:max_sentences]

def select_caption_topk(description: str,
                        K: int,
                        mode: str = "class_sim",
                        labels_txt_path: Optional[str] = None,
                        template_set: str = "clip",
                        max_templates_per_label: int = 3,
                        encoder: str = "jinaclip",
                        encoder_name: Optional[str] = None,
                        target_dim: Optional[int] = None) -> Tuple[List[str], torch.Tensor]:
    """
    将单条描述切分成句子 → 选择 Top-K 句子 → 返回 (句子列表, 特征矩阵[K,D])
    mode:
      - class_sim: 每句和“全类原型库”的最大 cos 相似度
      - firstk   : 取前 K 句
      - lenk     : 取长度前 K
    """
    sents = split_into_sentences(description, max_sentences=K * 2 if K > 0 else 8)
    if len(sents) == 0:
        return [], torch.zeros(0, target_dim or 512)

    if K <= 0 or len(sents) <= K:
        feats = encode_prompts(sents, encoder=encoder, encoder_name=encoder_name, target_dim=target_dim)
        return sents, feats

    if mode == "firstk":
        chosen = sents[:K]
        feats = encode_prompts(chosen, encoder=encoder, encoder_name=encoder_name, target_dim=target_dim)
        return chosen, feats

    if mode == "lenk":
        idx = sorted(range(len(sents)), key=lambda i: len(sents[i]), reverse=True)[:K]
        chosen = [sents[i] for i in idx]
        feats = encode_prompts(chosen, encoder=encoder, encoder_name=encoder_name, target_dim=target_dim)
        return chosen, feats

    # class_sim
    assert labels_txt_path is not None, "class_sim 需要 labels_txt_path"
    labels = [ln.strip() for ln in Path(labels_txt_path).read_text().splitlines() if ln.strip()]
    groups = build_prompt_groups_from_labels(labels, L=len(labels), template_set=template_set,
                                             max_templates_per_label=max_templates_per_label)
    classbank = encode_prompts(groups, encoder=encoder, encoder_name=encoder_name, target_dim=target_dim)  # [C,D]
    sent_feats = encode_prompts(sents, encoder=encoder, encoder_name=encoder_name, target_dim=target_dim) # [N,D]
    # cos 已归一化，直接点乘
    scores = (sent_feats @ classbank.T).amax(dim=1)  # [N]
    topk_idx = torch.topk(scores, k=min(K, len(sents)), dim=0).indices.tolist()
    chosen = [sents[i] for i in topk_idx]
    feats = sent_feats[topk_idx]
    unload_clip_model()
    return chosen, feats

# =========================
# Eval helpers (兼容原先 API)
# =========================
def prepare_eval_prompts(eval_txt_path: str,
                         prompt_json_path: str,
                         encoder: str = "jinaclip",
                         encoder_name: Optional[str] = None,
                         target_dim: Optional[int] = None,
                         register_set_name: str = "eval"):
    eval_list = Path(eval_txt_path).read_text().splitlines()
    fnames = [Path(l.split()[0]).name for l in eval_list]
    prompt_dict = json.loads(Path(prompt_json_path).read_text())
    all_prompts = [prompt_dict.get(fn, "") for fn in fnames]
    prompt_embeds = encode_prompts(all_prompts, encoder=encoder, encoder_name=encoder_name, target_dim=target_dim)
    prompt_embeds = prompt_embeds.cpu()
    register_prompt_embeds(register_set_name, prompt_embeds)
    switch_prompt_set(register_set_name)
    unload_clip_model()
    return {"filenames": fnames, "embeds": prompt_embeds, "D": prompt_embeds.shape[-1], "set_name": register_set_name}
