# prompt_utils.py (based on your original version; CLIP -> open_clip, Jina-CLIP via HF)
import os, re, json
from pathlib import Path
from typing import List, Union, Optional, Tuple

import torch
import torch.nn.functional as F

# --- Optional backends ---
from transformers import AutoModel, AutoTokenizer  # Jina-CLIP 走 HF
# 注意：不再从 transformers 加载 CLIPModel，CLIP 统一改走 open_clip

# =========================
# Global cache (models & embeds)
# =========================
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_GLOBAL = {
    "model": None,
    "tokenizer": None,   # open_clip 才会用到
    "name": None,        # "<encoder>:<encoder_name>:<device>"
    "device": _DEVICE,
}

PROMPT_EMBEDS = None
PROMPT_CACHE = {}
ACTIVE_PROMPT_SET = "train"

# =========================
# Utils
# =========================
def _l2norm(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1) if x.numel() else x

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

# =========================
# open_clip name resolver
# =========================
def _resolve_openclip_name(encoder_name: Optional[str]) -> Tuple[str, str]:
    """
    将常见的 HF/口语化名字映射到 open_clip 的 (model_name, pretrained_tag)
    """
    if not encoder_name:
        return "ViT-B-16", "openai"
    n = encoder_name.lower()
    if "vit-b-16" in n or "b/16" in n or "base-patch16" in n:
        return "ViT-B-16", "openai"
    if "vit-l-14" in n or "l/14" in n or "large-patch14" in n:
        return "ViT-L-14", "openai"
    if "vit-h-14" in n or "h/14" in n:
        return "ViT-H-14", "laion2b_s32b_b79k"
    # 兜底
    return "ViT-B-16", "openai"

# =========================
# Encoder loader
# =========================
def _load_encoder(encoder: str = "jinaclip",
                  encoder_name: Optional[str] = None,
                  device: Optional[torch.device] = None):
    """
    encoder: "clip" | "jinaclip"
      - clip:     走 open_clip（避免 transformers 的 .bin/weights_only 问题）
      - jinaclip: 走 transformers（jinaai/jina-clip-v2）
    """
    if device is None:
        device = _DEVICE
    if encoder_name is None:
        encoder_name = "jinaai/jina-clip-v2" if encoder == "jinaclip" else "ViT-B-16"

    if _GLOBAL["model"] is not None:
        return _GLOBAL["model"], _GLOBAL["tokenizer"], _GLOBAL["device"]

    if encoder == "clip":
        # 使用 open_clip
        import open_clip
        model_name, pretrained_tag = _resolve_openclip_name(encoder_name)
        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_tag, device=device)
        tok = open_clip.get_tokenizer(model_name)
        model.eval().to(device)
        for p in model.parameters(): p.requires_grad = False
        _GLOBAL.update({"model": model, "tokenizer": tok, "name": f"{encoder}:{model_name}", "device": device})
        return model, tok, device

    # 默认：Jina-CLIP via HF
    model_id = encoder_name or "jinaai/jina-clip-v2"
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    model.eval().to(device)
    for p in model.parameters(): p.requires_grad = False
    _GLOBAL.update({"model": model, "tokenizer": None, "name": f"{encoder}:{model_id}", "device": device})
    return model, None, device

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
            # open_clip：tokenizer 返回 callable；也可用 open_clip.tokenize
            import open_clip
            toks = open_clip.tokenize(texts).to(device)
            feats = model.encode_text(toks)       # [N, D]
        else:
            # Jina-CLIP 提供 encode_text(list[str], truncate_dim=?)
            truncate_dim = int(os.getenv("JINA_CLIP_DIM", "512"))
            out = model.encode_text(texts, truncate_dim=truncate_dim)
            feats = out if isinstance(out, torch.Tensor) else torch.from_numpy(out).to(device)
    feats = feats.to(device)
    feats = _l2norm(feats)

    # Check for NaN/Inf and fix if necessary
    if torch.isnan(feats).any() or torch.isinf(feats).any():
        print(f"WARNING: NaN/Inf detected in text encoding, replacing with zeros")
        feats = torch.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=-1.0)
        # Re-normalize after fixing
        feats = _l2norm(feats)

    return feats

def encode_prompts(prompts: List[Union[str, List[str]]],
                   encoder: str = "jinaclip",
                   encoder_name: Optional[str] = None,
                   target_dim: Optional[int] = None) -> torch.Tensor:
    """
    输入：
      - List[str]：直接编码 -> [N, D]
      - List[List[str]]：同组平均 -> [G, D]
    输出：Tensor [N_or_G, target_dim or D]
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
        outs.append(feats.mean(dim=0, keepdim=True))         # [1, D]
    feats = torch.cat(outs, dim=0)                           # [G, D]
    return _postproject(feats, target_dim)

def encode_prompt(prompt: str,
                  encoder: str = "jinaclip",
                  encoder_name: Optional[str] = None,
                  target_dim: Optional[int] = None) -> torch.Tensor:
    return encode_prompts([prompt], encoder=encoder, encoder_name=encoder_name, target_dim=target_dim)[0]

# =========================
# Batch encoding for label deduplication
# =========================
def encode_labels_batch(labels: List[str],
                        template_set: str = "clip",
                        max_templates_per_label: int = 3,
                        encoder: str = "jinaclip",
                        encoder_name: Optional[str] = None,
                        target_dim: Optional[int] = None,
                        batch_size: int = 512) -> dict:
    """
    批量编码标签，返回标签到embedding的映射字典（去重优化）

    Args:
        labels: 标签列表（可能包含重复）
        template_set: 模板集合（"clip" / "none"）
        max_templates_per_label: 每个标签的最大模板数
        encoder: 编码器类型
        encoder_name: 编码器名称
        target_dim: 目标维度
        batch_size: 批量编码的batch大小

    Returns:
        dict: {label_name: tensor[D]} 标签到embedding的映射
    """
    # 去重保序
    unique_labels = []
    seen = set()
    for lb in labels:
        lb_norm = _normalize_label(lb)
        if lb_norm and lb_norm not in seen:
            unique_labels.append(lb_norm)
            seen.add(lb_norm)

    if len(unique_labels) == 0:
        return {}

    # 为每个标签生成模板变体
    label_to_prompts = {}
    all_prompts = []
    prompt_to_label = []  # 记录每个prompt属于哪个标签

    for lb in unique_labels:
        variants = build_prompt_variants_for_label(lb, template_set, max_templates_per_label)
        label_to_prompts[lb] = variants
        all_prompts.extend(variants)
        prompt_to_label.extend([lb] * len(variants))

    # 批量编码所有prompts
    if len(all_prompts) == 0:
        return {}

    all_embeds = []
    for i in range(0, len(all_prompts), batch_size):
        batch_prompts = all_prompts[i:i+batch_size]
        batch_embeds = _encode_texts(batch_prompts, encoder, encoder_name)
        # 立即移到CPU，避免设备不匹配
        all_embeds.append(batch_embeds.cpu())

    all_embeds = torch.cat(all_embeds, dim=0)  # [total_prompts, D]
    all_embeds = _postproject(all_embeds, target_dim)

    # 按标签聚合（平均多个模板的embedding）
    label_embeds = {}
    idx = 0
    for lb in unique_labels:
        num_variants = len(label_to_prompts[lb])
        label_feats = all_embeds[idx:idx+num_variants]  # [num_variants, D]
        # 确保返回的embedding在CPU上
        label_embeds[lb] = label_feats.mean(dim=0).cpu()  # [D]
        idx += num_variants

    return label_embeds

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
    labels = [ln.strip() for ln in Path(labels_txt_path).read_text(encoding="utf-8").splitlines() if ln.strip()]
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
    parts = [p.strip() for p in _SENT_SPLIT.split(text.replace("\n", " ")) if p.strip()]
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
      - class_sim: 每句与“类原型库”的最大 cos 相似度
      - firstk   : 取前 K 句
      - lenk     : 取长度前 K
    """
    sents = split_into_sentences(description, max_sentences=K * 2 if K > 0 else 8)
    if len(sents) == 0:
        td = target_dim or 512
        return [], torch.zeros(0, td)

    if K <= 0 or len(sents) <= K:
        feats = encode_prompts(sents, encoder=encoder, encoder_name=encoder_name, target_dim=target_dim)
        unload_clip_model()
        return sents, feats

    if mode == "firstk":
        chosen = sents[:K]
        feats = encode_prompts(chosen, encoder=encoder, encoder_name=encoder_name, target_dim=target_dim)
        unload_clip_model()
        return chosen, feats

    if mode == "lenk":
        idx = sorted(range(len(sents)), key=lambda i: len(sents[i]), reverse=True)[:K]
        chosen = [sents[i] for i in idx]
        feats = encode_prompts(chosen, encoder=encoder, encoder_name=encoder_name, target_dim=target_dim)
        unload_clip_model()
        return chosen, feats

    # class_sim
    assert labels_txt_path is not None, "class_sim 需要 labels_txt_path"
    labels = [ln.strip() for ln in Path(labels_txt_path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    groups = build_prompt_groups_from_labels(labels, L=len(labels), template_set=template_set,
                                             max_templates_per_label=max_templates_per_label)
    classbank = encode_prompts(groups, encoder=encoder, encoder_name=encoder_name, target_dim=target_dim)  # [C,D]
    sent_feats = encode_prompts(sents, encoder=encoder, encoder_name=encoder_name, target_dim=target_dim) # [N,D]
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
    eval_list = Path(eval_txt_path).read_text(encoding="utf-8").splitlines()
    fnames = [Path(l.split()[0]).name for l in eval_list]
    prompt_dict = json.loads(Path(prompt_json_path).read_text(encoding="utf-8"))
    all_prompts = [prompt_dict.get(fn, "") for fn in fnames]
    prompt_embeds = encode_prompts(all_prompts, encoder=encoder, encoder_name=encoder_name, target_dim=target_dim).cpu()
    register_prompt_embeds(register_set_name, prompt_embeds)
    switch_prompt_set(register_set_name)
    unload_clip_model()
    return {"filenames": fnames, "embeds": prompt_embeds, "D": prompt_embeds.shape[-1], "set_name": register_set_name}