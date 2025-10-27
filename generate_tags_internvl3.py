#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, argparse, logging, glob, time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from itertools import islice

import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ===== NYUv2: 40类去掉最后3个模糊类 => 37类，固定ID [0..36] =====
NYU40 = [
    "wall","floor","cabinet","bed","chair","sofa","table","door","window","bookshelf",
    "picture","counter","blinds","desk","shelves","curtain","dresser","pillow","mirror","floor mat",
    "clothes","ceiling","books","refrigerator","television","paper","towel","shower curtain","box","whiteboard",
    "person","night stand","toilet","sink","lamp","bathtub","bag","otherstructure","otherfurniture","otherprop"
]
AMBIG = {"otherstructure","otherfurniture","otherprop"}
NYU37 = [c for c in NYU40 if c not in AMBIG]  # 按上面顺序保留
CLS2ID = {c:i for i,c in enumerate(NYU37)}

# 允许的“同义/拼写归一化”映射（模型常见回答修正）
NORM_MAP = {
    "nightstand": "night stand",
    "night_stand": "night stand",
    "floormat": "floor mat",
    "tv": "television",
    "bookshelf": "bookshelf",
    "book shelf": "bookshelf",
    "white board": "whiteboard",
    "refridgerator": "refrigerator",  # 常见拼写错误
    "book": "books",
    "cloth": "clothes",
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

# ====== 动态切tile（与你原逻辑一致，稍做安全检查） ======
def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=8, image_size=448, use_thumbnail=True):
    orig_w, orig_h = image.size
    aspect_ratio = orig_w / max(orig_h, 1)
    target_ratios = sorted({(i, j) for n in range(min_num, max_num + 1)
                            for i in range(1, n + 1) for j in range(1, n + 1)
                            if 1 <= i*j <= max_num}, key=lambda x: x[0]*x[1])
    tr = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_w, orig_h, image_size)
    target_w = image_size * tr[0]
    target_h = image_size * tr[1]
    blocks = tr[0] * tr[1]
    resized = image.resize((target_w, target_h))
    processed = []
    cols = max(target_w // image_size, 1)
    rows = max(target_h // image_size, 1)
    for r in range(rows):
        for c in range(cols):
            box = (c*image_size, r*image_size, (c+1)*image_size, (r+1)*image_size)
            processed.append(resized.crop(box))
    if use_thumbnail and blocks != 1:
        processed.append(image.resize((image_size, image_size)))
    return processed

def load_image_tiles(image_file: str, input_size=448, max_num=8):
    try:
        img = Image.open(image_file).convert('RGB')
        transform = build_transform(input_size)
        tiles = dynamic_preprocess(img, image_size=input_size, max_num=max_num, use_thumbnail=True)
        pix = [transform(t) for t in tiles]
        return torch.stack(pix)
    except Exception as e:
        logging.error(f"Image error {image_file}: {e}")
        return None

def batched(iterable, n):
    it = iter(iterable)
    while True:
        chunk = tuple(islice(it, n))
        if not chunk:
            return
        yield chunk

# ====== 稳健解析：优先JSON；否则按标签表做“子串+边界”匹配，支持多词 ======
def normalize_label(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[\s_-]+", " ", s)
    return NORM_MAP.get(s, s)

def extract_labels(text: str, allowed: List[str], topk: Optional[int] = None) -> List[str]:
    text = (text or "").strip()
    # 1) 优先找 JSON
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and "labels" in obj and isinstance(obj["labels"], list):
                labs = [normalize_label(x) for x in obj["labels"]]
                # 过滤非允许集
                labs = [x for x in labs if x in allowed]
                # 去重保序
                seen = set(); out=[]
                for x in labs:
                    if x not in seen:
                        out.append(x); seen.add(x)
                return out if topk is None else out[:topk]
        except Exception:
            pass
    # 2) 回退：对每个allowed标签做宽松“词边界/子串”匹配（优先完整词）
    cand = []
    low = " " + re.sub(r"[^a-z0-9\s]", " ", text.lower()) + " "
    for lab in allowed:
        pattern_full = r"\b" + re.escape(lab) + r"\b"
        pattern_space = re.sub(r"\s+", r"\\s+", pattern_full)
        if re.search(pattern_space, low):
            cand.append(lab)
        else:
            # 宽松子串匹配（多词标签也能命中）
            if lab.replace(" ", "") in low.replace(" ", ""):
                cand.append(lab)
    # 保障确定性：按在文本中第一次出现位置排序
    def first_pos(lab):
        idx = low.find(lab)
        if idx == -1:
            idx = low.find(lab.replace(" ", ""))
        return idx if idx >= 0 else 10**9
    cand = sorted(list(dict.fromkeys(cand)), key=first_pos)
    return cand if topk is None else cand[:topk]

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, default="OpenGVLab/InternVL3-38B")
    ap.add_argument("--dataset_dir", type=str, default=str(Path("datasets/NYUDepthv2").resolve()))
    ap.add_argument("--image_folder", type=str, default="RGB")
    ap.add_argument("--output_file", type=str, default="topk_labels2_internvl3.json",
                    help="相对dataset_dir的路径，或绝对路径")
    # ap.add_argument("--num_tags", type=int, default=5, choices=[5,6,8])
    ap.add_argument("--batch_size", type=int, default=1, help="先从1稳起，避免OOM")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--max_tiles", type=int, default=8, help="每图最多tile数，降低OOM风险")
    return ap.parse_args()

def load_model_tokenizer(model_id: str):
    logging.info(f"Loading model: {model_id}")
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto",
    ).eval()
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
    logging.info("Model ready.")
    return model, tok

def main():
    args = parse_args()

    # 规范输出路径：相对 -> 拼到 dataset_dir；绝对 -> 原样
    out_path = Path(args.output_file)
    if not out_path.is_absolute():
        out_path = Path(args.dataset_dir) / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 收集图片
    img_dir = Path(args.dataset_dir) / args.image_folder
    img_paths = sorted(
        glob.glob(str(img_dir / "**" / "*.jpg"), recursive=True) +
        glob.glob(str(img_dir / "**" / "*.png"), recursive=True) +
        glob.glob(str(img_dir / "**" / "*.jpeg"), recursive=True)
    )
    if not img_paths:
        logging.error(f"No images under {img_dir}")
        return

    # 续跑：字典结构 {rel_path: [ids]}
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        if not isinstance(index, dict):
            logging.warning("Output exists but not dict; reinit as dict.")
            index = {}
    else:
        index = {}

    # 只处理未完成的
    rel_paths_all = [str(Path(p).relative_to(args.dataset_dir)) for p in img_paths]
    to_process = [p for p in rel_paths_all if p not in index]
    logging.info(f"Total {len(rel_paths_all)} | To process {len(to_process)} | Output -> {out_path}")

    # 加载模型
    model, tok = load_model_tokenizer(args.model_id)

    # 固定提示词（InternVL3 使用 batch_chat，不一定需要 <image> 标记）
    allowed_list = ", ".join(NYU37)
    prompt_tpl = (
        f"You are a visual recognition expert specializing in indoor scene analysis.\n"
        f"Allowed labels (37): {allowed_list}\n\n"
        f"Follow these steps carefully:\n"
        f"Use ONLY labels from the allowed list; output canonical names exactly as written."
        f"Order matters: the list must be sorted by estimated area dominance."
        f"Prefer large structural surfaces first , then fill remaining slots with clearly visible distinctive items.\n"
        f"If at least 5 labels are clearly visible, you MUST output exactly 5."
        f"If fewer are clearly visible, output as many as are clearly visible (≥1). "
        f"When unsure between synonyms, choose the closest canonical label from the allowed list.\n"
        f"Do NOT include objects that are not visible. Do NOT invent labels not in the allowed list.\n"
    )

    # 主循环
    for batch in tqdm(list(batched(to_process, args.batch_size)), desc="InternVL3 tagging"):
        # 组装batch tiles
        pv_list, n_tiles, rel_list = [], [], []
        for rel in batch:
            full = str(Path(args.dataset_dir) / rel)
            pv = load_image_tiles(full, max_num=args.max_tiles)  # 降低tile数更稳
            if pv is None:
                continue
            pv_list.append(pv)
            n_tiles.append(pv.size(0))
            rel_list.append(rel)
        if not rel_list:
            continue

        # 拼接到同一设备
        pixel_values = torch.cat(pv_list, dim=0).to(model.device, dtype=getattr(model, "dtype", torch.bfloat16))
        questions = [prompt_tpl] * len(rel_list)

        try:
            with torch.no_grad():
                responses = model.batch_chat(
                    tok,
                    pixel_values,
                    num_patches_list=n_tiles,
                    questions=questions,
                    generation_config=dict(max_new_tokens=args.max_new_tokens, do_sample=False),
                )
        except Exception as e:
            logging.error(f"batch_chat failed on {rel_list[0]}: {e}")
            # 失败兜底：全部给空，后续直接记录为空列表
            responses = [""] * len(rel_list)

        # 解析 -> 映射到ID -> 存入字典
        for rel, resp in zip(rel_list, responses):
            labs = extract_labels(resp, [normalize_label(x) for x in NYU37], topk=None)
            ids = [CLS2ID[l] for l in labs]
            index[rel] = ids

        # 每批次持久化，稳一点
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

        # 小憩，避免过热
        time.sleep(0.05)

    logging.info(f"Done. Wrote {len(index)} entries to {out_path}")

if __name__ == "__main__":
    main()
