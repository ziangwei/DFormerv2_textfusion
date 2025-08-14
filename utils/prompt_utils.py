import clip
import torch
from typing import List, Union

# 最简单的模板：只保留场景信息，其它 filler
BASE_TEMPLATE = "this is a {} image"
PROMPT_EMBEDS = None
PROMPT_CACHE = {}
ACTIVE_PROMPT_SET = "train"

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

def encode_prompts(prompts: List[Union[str, List[str]]]):
    """
    支持列表中的元素为字符串或字符串列表。如果为列表，将对其中
    每个字符串分别编码并对特征取平均。返回 text_feats
    """

    # # ---- flatten prompts and record lengths ----
    # flat_prompts: List[str] = []
    # lens: List[int] = []
    # for p in prompts:
    #     if isinstance(p, list):
    #         flat_prompts.extend(p)
    #         lens.append(len(p))
    #     else:
    #         flat_prompts.append(p)
    #         lens.append(1)
    #
    # model = _get_clip_model()
    # tokens = clip.tokenize(flat_prompts, truncate=True).to(_DEVICE)  # (M, token_len)
    #
    # # 2) forward through CLIP text encoder
    # with torch.no_grad():
    #     text_feats = model.encode_text(tokens)  # (M, D)
    #     text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    #
    # out_text_feats = []
    # idx = 0
    # for ln in lens:
    #     out_text_feats.append(text_feats[idx: idx + ln].mean(dim=0))
    #     idx += ln
    #
    # return torch.stack(out_text_feats)

    processed_prompts: List[str] = []
    for p in prompts:
        if isinstance(p, list):
            processed_prompts.append("a picture with " + ", ".join(p))
        else:
            processed_prompts.append(p)

    model = _get_clip_model()
    tokens = clip.tokenize(processed_prompts, truncate=True).to(_DEVICE)

    with torch.no_grad():
        text_feats = model.encode_text(tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    return text_feats



def encode_prompt(prompt: str):
    """
    单条编码，返回 shape = (feature_dim,)。
    内部调用 encode_prompts，仅做语法糖。
    """
    feats = encode_prompts([prompt])
    return feats[0]

def set_prompt_embeds(text_embeds: torch.Tensor):
    """
    在 train.py 中调用，将 encode_prompts 返回的特征存入全局变量。
    """
    global PROMPT_EMBEDS
    PROMPT_EMBEDS = text_embeds

def register_prompt_embeds(set_name: str, text_embeds: torch.Tensor):
    global PROMPT_CACHE
    PROMPT_CACHE[set_name] = text_embeds


def switch_prompt_set(set_name: str):
    global PROMPT_EMBEDS, PROMPT_CACHE, ACTIVE_PROMPT_SET
    assert set_name in PROMPT_CACHE, f"Prompt set {set_name} not registered!"
    PROMPT_EMBEDS = PROMPT_CACHE[set_name]
    ACTIVE_PROMPT_SET = set_name

def prepare_eval_prompts(eval_txt_path, prompt_json_path):
    from pathlib import Path
    import json
    eval_list = Path(eval_txt_path).read_text().splitlines()
    fnames = [Path(l.split()[0]).name for l in eval_list]
    prompt_dict = json.loads(Path(prompt_json_path).read_text())
    all_prompts = [prompt_dict.get(fn, "") for fn in fnames]
    prompt_embeds = encode_prompts(all_prompts)
    prompt_embeds = prompt_embeds.cpu()
    register_prompt_embeds("eval", prompt_embeds)
    switch_prompt_set("eval")
    unload_clip_model()
