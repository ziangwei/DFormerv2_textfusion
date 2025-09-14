import clip
import torch
import open_clip
from typing import List, Union
from transformers import AutoModel
import os
from pathlib import Path
import json
import itertools
import re

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
_TOKENIZER = None
_TRUNCATE_DIM = int(os.getenv("JINA_CLIP_DIM", "512"))  # 用 Jina 的 Matryoshka 截断维度，默认 512

def _get_clip_model():
    global _CLIP_MODEL, _TOKENIZER
    if _CLIP_MODEL is None:
        # # 标准 OpenAI CLIP
        # _CLIP_MODEL, _ = clip.load("ViT-B/32", device=_DEVICE)  # D=512
        # _CLIP_MODEL.eval()
        # for p in _CLIP_MODEL.parameters():
        #     p.requires_grad = False
        # # 统一一个与旧代码兼容的分词器接口
        # _TOKENIZER = lambda texts: clip.tokenize(texts, truncate=True)
        # Jina-CLIP v2（通过 transformers 加载）

        _CLIP_MODEL = AutoModel.from_pretrained(
            "jinaai/jina-clip-v2", trust_remote_code=True
        ).to(_DEVICE)
        _CLIP_MODEL.eval()
        for p in _CLIP_MODEL.parameters():
            p.requires_grad = False
        _TOKENIZER = None  # 不再需要手工 tokenizer

    return _CLIP_MODEL


def unload_clip_model():

    # """Free the globally loaded CLIP model and release GPU memory."""
    # global _CLIP_MODEL

    """Free the globally loaded EVA-CLIP model and release GPU memory."""
    global _CLIP_MODEL, _TOKENIZER

    if _CLIP_MODEL is not None:
        del _CLIP_MODEL
        _CLIP_MODEL = None
        _TOKENIZER = None
        torch.cuda.empty_cache()

def encode_prompts(prompts: List[Union[str, List[str]]]):
    """
    支持列表中的元素为字符串或字符串列表。如果为列表，将对其中
    每个字符串分别编码并对特征取平均。返回 text_feats
    """

    # ---- flatten prompts and record lengths ----
    flat_prompts: List[str] = []
    lens: List[int] = []
    for p in prompts:
        if isinstance(p, list):
            flat_prompts.extend(p)
            lens.append(len(p))
        else:
            flat_prompts.append(p)
            lens.append(1)

    # model = _get_clip_model()
    # # tokens = clip.tokenize(flat_prompts, truncate=True).to(_DEVICE)  # (M, token_len)
    # tokens = _TOKENIZER(flat_prompts).to(_DEVICE)  # (M, token_len)
    #
    # # 2) forward through CLIP text encoder
    # with torch.no_grad():
    #     text_feats = model.encode_text(tokens)  # (M, D)
    #     text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    model = _get_clip_model()

    # 直接传字符串列表给 Jina-CLIP v2；用 Matryoshka 截断到 512 维，保持与你模型 text_dim 对齐
    with torch.no_grad():
        text_feats = model.encode_text(flat_prompts, truncate_dim=_TRUNCATE_DIM)
        # 兼容返回 numpy 的实现
        if not isinstance(text_feats, torch.Tensor):
            text_feats = torch.from_numpy(text_feats)
        text_feats = text_feats.to(_DEVICE)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)


    out_text_feats = []
    idx = 0
    for ln in lens:
        out_text_feats.append(text_feats[idx: idx + ln].mean(dim=0))
        idx += ln

    return torch.stack(out_text_feats)

    # processed_prompts: List[str] = []
    # for p in prompts:
    #     if isinstance(p, list):
    #         processed_prompts.append("a picture with " + ", ".join(p))
    #     else:
    #         processed_prompts.append(p)
    #
    # model = _get_clip_model()
    # tokens = clip.tokenize(processed_prompts, truncate=True).to(_DEVICE)
    #
    # with torch.no_grad():
    #     text_feats = model.encode_text(tokens)
    #     text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    #
    # return text_feats



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


# =========================
# 多标签 → CLIP 风格短句 → (L, D) 特征 的工具集
# =========================

# 1) 一组“CLIP 式”轻模板（参考原论文风格，尽量通用室内分割）
#    会从这些模板里为每个标签构造若干句子，最后对同一标签的句子做平均得到一个向量。
PROMPT_TEMPLATES_CLIP = [
    "a photo of a {}.",
    "this is a photo of a {}.",
    "an image of a {}.",
]

# 2) 常见拼写/同义词归一化（可按需扩充）
#    目的：减少不必要的词形噪声，提升文本编码的稳定性。
_LABEL_ALIAS = {
    "refridgerator": "refrigerator",
    "night stand": "nightstand",
    "floor mat": "rug",
    "television": "tv",
    "picture": "picture",
    "blinds": "window blinds",
    "sofa": "couch",
    "books": "book",
    "clothes": "clothing",
    "towel": "bath towel",
    "sink": "washbasin",
    "bathtub": "bath tub",
    "toilet": "flush toilet",
    "counter": "countertop",
    "paper": "paper sheet",
    "floor": "flooring",
    "ceiling": "ceiling",
    "lamp": "lamp",
    "bag": "bag",
    "box": "cardboard box",
    "desk": "desk",
    "chair": "chair",
    "table": "table",
    "window": "window",
    "door": "door",
    "mirror": "mirror",
    "cabinet": "cabinet",
    "bookshelf": "bookcase",
    "whiteboard": "whiteboard",
    "dresser": "dresser",
    "curtain": "curtain",
    "bed": "bed",
    "pillow": "pillow",
    "shelves": "shelf",
    "picture frame": "frame",
}

def _normalize_label(label: str) -> str:
    """
    将数据集标签做最小限度的词形清洗/归一化，避免明显拼写问题。
    """
    s = label.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return _LABEL_ALIAS.get(s, s)

def _pick_article(noun: str) -> str:
    """
    简单选择 a/an（不追求完美，足够好看就行）。
    """
    return "an" if noun[:1] in "aeiou" else "a"

def build_prompt_variants_for_label(
    label: str,
    template_set: str = "clip",
    max_templates_per_label: int = 4,
) -> list[str]:
    """
    给单个标签构造若干条 CLIP 风格短句。
    - label: 数据集里的原始标签（例如 "bookshelf"）
    - template_set: 暂时只支持 "clip"，后续可扩展不同风格
    - max_templates_per_label: 限制每个标签使用前 N 条模板（节省算力）

    返回: 对应该标签的一组句子字符串列表（长度 <= max_templates_per_label）
    """
    lbl = _normalize_label(label)
    if template_set == "clip":
        # 将模板里 a/an 做个小修饰，让句子自然一点
        variants = []
        for t in PROMPT_TEMPLATES_CLIP[:max_templates_per_label]:
            if "{}" in t:
                # 把模板里的 "a {}" / "an {}" 替换得更自然
                if "a {}" in t:
                    variants.append(t.replace("a {}", f"{_pick_article(lbl)} {lbl}"))
                elif "an {}" in t:
                    variants.append(t.replace("an {}", f"{_pick_article(lbl)} {lbl}"))
                else:
                    variants.append(t.format(lbl))
            else:
                # 兜底：若模板没占位符，直接拼在后面
                variants.append(f"{t} {lbl}")
        return variants
    else:
        # 兜底：不用任何模板，直接用原词
        return [lbl]

def build_prompt_groups_from_labels(
    labels: list[str],
    L: int = 5,
    template_set: str = "clip",
    max_templates_per_label: int = 4,
) -> list[list[str]]:
    """
    将“若干标签”转为“句子组”的列表（**列表的列表**）：
    - 外层长度 = L（即要使用的标签数，超出会截断，不足会补齐）
    - 内层每个元素 = 针对该标签的一组短句（会在 encode 时求平均）

    之所以返回“列表的列表”，是为了对接你现有的 encode_prompts：
    它会对**每个内层列表**做编码并**平均**，最终为每个标签产出一个 512 维向量。:contentReference[oaicite:1]{index=1}
    """
    # 1) 归一化 + 去重（保持顺序）
    norm_labels = []
    seen = set()
    for lb in labels:
        nlb = _normalize_label(lb)
        if nlb and nlb not in seen:
            norm_labels.append(nlb)
            seen.add(nlb)

    # 2) 截断/补齐到 L
    PAD_TOKEN = "object"  # 补齐用一个中性占位词，避免空字符串扰动 tokenizer
    if len(norm_labels) >= L:
        norm_labels = norm_labels[:L]
    else:
        norm_labels = norm_labels + [PAD_TOKEN] * (L - len(norm_labels))

    # 3) 为每个标签构造若干短句
    prompt_groups = [
        build_prompt_variants_for_label(lb, template_set, max_templates_per_label)
        for lb in norm_labels
    ]
    return prompt_groups  # 形如 [[句1,句2,...], [句1,句2,...], ...], 长度 = L

def encode_label_set_to_matrix(
    labels: list[str],
    L: int = 5,
    template_set: str = "clip",
    max_templates_per_label: int = 4,
):
    """
    针对**一张图**的若干标签（通常 5 个），生成 (L, D) 的文本特征矩阵。
    - 外层每个标签先扩写成 2~4 条短句
    - 对同一标签的句子向量**做平均**得到一条 512-D（利用 encode_prompts 的分组平均）:contentReference[oaicite:2]{index=2}
    - 将 L 个标签依次堆叠，得到 (L, D)

    返回：
      text_feats_LD: torch.Tensor, shape = (L, D=512)
      used_labels:   list[str],     长度 L（归一化后的标签序列；含可能补齐的占位词）
    """
    groups = build_prompt_groups_from_labels(
        labels, L=L, template_set=template_set, max_templates_per_label=max_templates_per_label
    )
    # 利用你已有的 encode_prompts：内层列表会被平均成“一条向量”
    text_feats_LD = encode_prompts(groups)  # (L, D)，不会把 L 进一步平均！:contentReference[oaicite:3]{index=3}
    return text_feats_LD, [_normalize_label(lb) for lb in labels[:L]]

def load_topk_labels_dict(json_path: str) -> dict[str, list[str]]:
    """
    读取你已经给出的 top5 标签字典（文件名 → [label1..label5]）。
    """
    return json.loads(Path(json_path).read_text())

def prepare_eval_prompts_multilabel(
    eval_txt_path: str,
    topk_json_path: str,
    K: int = 5,
    max_templates_per_label: int = 4,
    template_set: str = "clip",
    register_set_name: str = "eval-ml",
):
    """
    给一份列表文件（可以是 train_source 或 eval_source）准备 (N,K,D) 文本特征。
    """
    from pathlib import Path
    import json

    eval_list = Path(eval_txt_path).read_text().splitlines()
    fnames = [Path(l.split()[0]).name for l in eval_list]  # e.g. '228.jpg'

    # 读取 topK 标签，并做“主干名”索引，自动兼容 .jpg/.png
    raw_topk = json.loads(Path(topk_json_path).read_text())   # 键多为 '228.png'
    topk_by_stem = {Path(k).stem: v for k, v in raw_topk.items()}

    all_groups = []
    per_image_group_counts = []
    for fn in fnames:
        stem = Path(fn).stem                                  # '228'
        labels = raw_topk.get(fn) or raw_topk.get(f"{stem}.png") or raw_topk.get(f"{stem}.jpg") \
                 or topk_by_stem.get(stem, [])
        prompt_groups = build_prompt_groups_from_labels(
            labels, L=K, template_set=template_set,
            max_templates_per_label=max_templates_per_label
        )
        all_groups.extend(prompt_groups)
        per_image_group_counts.append(len(prompt_groups))

    text_feats_flat = encode_prompts(all_groups)              # (N*K, D)
    D = text_feats_flat.shape[-1]
    N = len(fnames)
    assert sum(per_image_group_counts) == N * K, "unexpected group count; check K"

    text_feats_NKD = text_feats_flat.view(N, K, D).cpu()
    register_prompt_embeds(register_set_name, text_feats_NKD)
    switch_prompt_set(register_set_name)
    unload_clip_model()

    return {"filenames": fnames, "embeds": text_feats_NKD, "K": K, "D": D, "set_name": register_set_name}


def gather_multilabel_text_feats_for_batch(
    all_filenames_in_eval_order: list[str],
    batch_filenames: list[str],
    prompt_set_name: str = "eval-ml",
):
    """
    给定一小批 batch 文件名（与 eval 列表里同名），
    从 PROMPT_CACHE[prompt_set_name] 里取出对应顺序的 (B, K, D) 文本特征。
    - 这样你在训练/验证时就能**一次 forward**把 5 个标签一起送进 backbone 的 SemanticSelfAttention。
    """
    assert prompt_set_name in PROMPT_CACHE, f"Prompt set {prompt_set_name} not registered!"
    text_bank_NKD = PROMPT_CACHE[prompt_set_name]  # (N, K, D)（我们在 prepare_eval_prompts_multilabel 里注册的）:contentReference[oaicite:10]{index=10}
    name_to_idx = {fn: i for i, fn in enumerate(all_filenames_in_eval_order)}
    idxs = [name_to_idx[Path(n).name] for n in batch_filenames]
    # 取子集并保持顺序
    return text_bank_NKD[idxs]  # (B, K, D)

def prepare_classbank_prompts(
    labels_txt_path: str,
    max_templates_per_label: int = 3,
    template_set: str = "clip",
    register_set_name: str = "classbank",
):
    """
    读取全类列表（如 NYU40），为每个类用多模板造句→编码→同类内平均，
    得到一份 (K,D) 的“类原型库”，注册到 PROMPT_CACHE[register_set_name]。
    """
    labels = [ln.strip() for ln in Path(labels_txt_path).read_text().splitlines() if ln.strip()]
    K = len(labels)
    # 利用你已有的“分组编码并在组内平均”的逻辑：
    groups = build_prompt_groups_from_labels(
        labels, L=K, template_set=template_set, max_templates_per_label=max_templates_per_label
    )  # 长度=K；每个元素是该类的若干句子
    text_feats_KD = encode_prompts(groups).cpu()  # (K,D)，每类已对多模板平均
    register_prompt_embeds(register_set_name, text_feats_KD)  # 缓存 (K,D)
    switch_prompt_set(register_set_name)
    unload_clip_model()
    return {"labels": labels, "embeds": text_feats_KD, "K": K, "D": text_feats_KD.shape[-1], "set_name": register_set_name}