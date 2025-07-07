import clip
import torch
from typing import List

# 最简单的模板：只保留场景信息，其它 filler
BASE_TEMPLATE = "this is a {} image"
PROMPT_EMBEDS = None

def load_scene_list(path):
    """
    读取 sceneTypes.txt，每行一个场景，返回一个 Python list。
    """
    with open(path, 'r') as f:
        scenes = [line.strip() for line in f if line.strip()]
    return scenes

def sample_prompt(scene_label: str) -> str:
    """
    根据单个 scene_label（如 "kitchen"）生成 prompt。
    """
    return BASE_TEMPLATE.format(scene_label)


# 全局加载一次 CLIP 模型并冻结参数
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_CLIP_MODEL = None


def _get_clip_model():
    """Lazy-load CLIP on first use to avoid unnecessary GPU memory usage."""
    global _CLIP_MODEL
    if _CLIP_MODEL is None:
        _CLIP_MODEL, _ = clip.load("ViT-B/32", device=_DEVICE)
        _CLIP_MODEL.eval()
        for p in _CLIP_MODEL.parameters():
            p.requires_grad = False
    return _CLIP_MODEL


def unload_clip_model():
    """Free the globally loaded CLIP model and release GPU memory."""
    global _CLIP_MODEL
    if _CLIP_MODEL is not None:
        del _CLIP_MODEL
        _CLIP_MODEL = None
        torch.cuda.empty_cache()

def encode_prompts(prompts: List[str]) -> torch.Tensor:
    """
    批量将文本列表编码为 CLIP 文本特征向量。
    返回 shape = (len(prompts), feature_dim)，已在 DEVICE 上。
    """
    model = _get_clip_model()
    tokens = clip.tokenize(prompts, truncate=True).to(_DEVICE)  # (N, token_len)
    with torch.no_grad():
        text_feats = model.encode_text(tokens)  # (N, D)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)  # L2 归一化（可选）
    return text_feats

def encode_prompt(prompt: str) -> torch.Tensor:
    """
    单条编码，返回 shape = (feature_dim,)。
    内部调用 encode_prompts，仅做语法糖。
    """
    return encode_prompts([prompt])[0]

def set_prompt_embeds(text_embeds: torch.Tensor):
    """
    在 train.py 中调用，将 encode_prompts 返回的特征存入全局变量。
    """
    global PROMPT_EMBEDS
    PROMPT_EMBEDS = text_embeds
