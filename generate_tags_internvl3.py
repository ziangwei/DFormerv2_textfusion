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

# ===== Fixed vocabularies (hard-coded): NYUv2-37 and SUNRGB-D-37 =====
NYU40 = [
    "wall","floor","cabinet","bed","chair","sofa","table","door","window","bookshelf",
    "picture","counter","blinds","desk","shelves","curtain","dresser","pillow","mirror","floor mat",
    "clothes","ceiling","books","refrigerator","television","paper","towel","shower curtain","box","whiteboard",
    "person","night stand","toilet","sink","lamp","bathtub","bag","otherstructure","otherfurniture","otherprop"
]
AMBIG = {"otherstructure","otherfurniture","otherprop"}
NYU37 = [c for c in NYU40 if c not in AMBIG]  # NYU 37-class (spaces)

# SUNRGB-D official 37-class list (seg37list.mat)
SUN37 = [
    "wall","floor","cabinet","bed","chair","sofa","table","door","window","bookshelf",
    "picture","counter","blinds","desk","shelves","curtain","dresser","pillow","mirror","floor_mat",
    "clothes","ceiling","books","fridge","tv","paper","towel","shower_curtain","box","whiteboard",
    "person","night_stand","toilet","sink","lamp","bathtub","bag"
]

# Dataset-specific normalization maps (variants -> canonical label under that dataset)
NORM_MAPS = {
    "nyu": {
        "nightstand": "night stand",
        "night_stand": "night stand",
        "floormat": "floor mat",
        "floor_mat": "floor mat",
        "tv": "television",
        "tv monitor": "television",
        "book shelf": "bookshelf",
        "bookshelves": "bookshelf",
        "white board": "whiteboard",
        "refridgerator": "refrigerator",
        "shower_curtain": "shower curtain",
        "closet": "cabinet",
        "wardrobe": "cabinet",
        "cloth": "clothes",
        "couch": "sofa",
        "book": "books",
    },
    "sun": {
        "nightstand": "night_stand",
        "night stand": "night_stand",
        "floormat": "floor_mat",
        "floor mat": "floor_mat",
        "television": "tv",
        "tv monitor": "tv",
        "refrigerator": "fridge",
        "fridgerator": "fridge",
        "book shelf": "bookshelf",
        "bookshelves": "bookshelf",
        "white board": "whiteboard",
        "shower curtain": "shower_curtain",
        "closet": "cabinet",
        "wardrobe": "cabinet",
        "cloth": "clothes",
        "couch": "sofa",
        "book": "books",
    }
}

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
                labs = [x for x in labs if x in allowed]  # 过滤非允许集
                seen = set(); out=[]
                for x in labs:
                    if x not in seen:
                        out.append(x); seen.add(x)
                return out if topk is None else out[:topk]
        except Exception:
            pass
    # 2) 回退：对每个 allowed 标签做宽松“词边界/子串”匹配（优先完整词）
    cand = []
    low = " " + re.sub(r"[^a-z0-9\s]", " ", text.lower()) + " "
    for lab in allowed:
        pattern_full = r"\b" + re.escape(lab) + r"\b"
        pattern_space = re.sub(r"\s+", r"\\s+", pattern_full)
        if re.search(pattern_space, low):
            cand.append(lab)
        else:
            if lab.replace(" ", "") in low.replace(" ", ""):
                cand.append(lab)
    # 保障确定性：按在文本中第一次出现位置排序
    def first_pos(lab):
        idx = low.find(lab)
        if idx == -1:
            idx = low.find(lab.replace(" ", ""))
        return 10**9 if idx == -1 else idx
    cand.sort(key=first_pos)
    # 去重保序
    seen = set(); out=[]
    for x in cand:
        if x not in seen:
            out.append(x); seen.add(x)
    return out if topk is None else out[:topk]

# ====== 图像 tiles 预处理（与你原来一致）======
def build_transform(input_size=448):
    MEAN = (0.48145466, 0.4578275, 0.40821073)
    STD  = (0.26862954, 0.26130258, 0.27577711)
    return T.Compose([
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])

def dynamic_preprocess(image, image_size=448, max_num=12, use_thumbnail=True):
    orig_width, orig_height = image.size
    if max_num is None:
        return [image]
    aspect_ratio = orig_width / orig_height
    area = orig_width * orig_height
    s = int((area / max_num) ** 0.5)
    t = image_size
    def _grid(num):
        if num == 1: return [(0, 0, orig_width, orig_height)]
        # 简化：横向/纵向切分为近似 t×t 的块
        xs = list(range(0, orig_width, max(1, int(orig_width / (num**0.5)))))
        ys = list(range(0, orig_height, max(1, int(orig_height / (num**0.5)))))
        boxes=[]
        for i in range(len(xs)-1):
            for j in range(len(ys)-1):
                boxes.append((xs[i], ys[j], xs[i+1], ys[j+1]))
        return boxes[:num]
    blocks = min(max_num, max(1, int(area / (t*t))))
    boxes = _grid(blocks)
    processed=[]
    for box in boxes:
        w = box[2]-box[0]; h = box[3]-box[1]
        if w>=t//3 and h>=t//3:
            processed.append(image.crop(box))
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

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, default="OpenGVLab/InternVL3-38B")
    ap.add_argument("--dataset", type=str, choices=["nyu","sun"], required=True,
                    help="choose which built-in vocabulary to use")
    ap.add_argument("--dataset_dir", type=str, required=True)
    ap.add_argument("--image_folder", type=str, default=None,
                    help="relative subfolder under dataset_dir; if None, scan dataset_dir recursively")
    ap.add_argument("--output_file", type=str, required=True,
                    help="output JSON path; absolute or relative to dataset_dir")
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

    # Dataset-specific labels and normalization
    global NORM_MAP
    if args.dataset == 'nyu':
        LABELS_CANON = NYU37
        NORM_MAP = NORM_MAPS['nyu']
        default_img_folder = 'RGB'
    else:
        LABELS_CANON = SUN37
        NORM_MAP = NORM_MAPS['sun']
        default_img_folder = 'image'

    # we will match on normalized labels but output canonical
    LABELS_MATCH = [normalize_label(x) for x in LABELS_CANON]
    MATCH2CANON = {normalize_label(x): x for x in LABELS_CANON}

    # 规范输出路径：相对 -> 拼到 dataset_dir；绝对 -> 原样
    out_path = Path(args.output_file)
    if not out_path.is_absolute():
        out_path = Path(args.dataset_dir) / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 收集图片
    img_root = Path(args.dataset_dir)
    img_dir = (img_root / args.image_folder) if args.image_folder else img_root
    img_paths = sorted(
        glob.glob(str(img_dir / "**" / "*.jpg"), recursive=True) +
        glob.glob(str(img_dir / "**" / "*.png"), recursive=True) +
        glob.glob(str(img_dir / "**" / "*.jpeg"), recursive=True)
    )
    if not img_paths:
        logging.error(f"No images under {img_dir}")
        return

    def to_rel(p: str) -> str:
        p = Path(p)
        try:
            return str(p.relative_to(img_root)).replace("\\", "/")
        except Exception:
            return p.name

    # 续跑：字典结构 {rel_path: [labels]}
    index: Dict[str, List[str]] = {}
    if out_path.exists():
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            if not isinstance(index, dict):
                logging.warning("Output exists but not dict; reinit empty.")
                index = {}
        except Exception:
            index = {}

    rel_paths_all = [to_rel(p) for p in img_paths]
    to_process = [p for p in rel_paths_all if p not in index]
    logging.info(f"Total {len(rel_paths_all)} | To process {len(to_process)} | Output -> {out_path}")

    # 加载模型
    model, tok = load_model_tokenizer(args.model_id)

    # 固定提示词（InternVL3 使用 batch_chat）
    allowed_list = ", ".join(LABELS_CANON)
    prompt_tpl = (
        f"You are given one image and a FIXED label vocabulary.\n"
        f"Goal: return ONLY a JSON array of STRING labels (no code block, no prose).\n"
        f"Include UP TO {8} labels that correspond to the LARGEST and MOST OBVIOUS regions in the image.\n\n"
        f"Selection rules:\n"
        f"- Include only labels that are obvious and clearly visible.\n"
        f"- Match with high confidence (clearly identifiable and reliable).\n"
        f"- Prioritize labels occupying the largest pixel areas.\n"
        f"- Prefer large structural surfaces first, then distinct objects.\n"
        f"- Use EXACT spelling from the vocabulary; DO NOT invent new labels.\n"
        f"- DO NOT output numeric IDs, indexes, or class numbers.\n\n"
        f"Vocabulary:\n{allowed_list}\n\n"
        f"Output format example:\n[\"wall\", \"floor\", \"table\", \"sofa\", \"window\"]"
    )

    def batch_to_msgs(img_batch: List[str]) -> List[Dict]:
        msgs=[]
        for p in img_batch:
            tiles = load_image_tiles(str(Path(args.dataset_dir) / p), input_size=448, max_num=args.max_tiles)
            if tiles is None:
                msgs.append({"role":"user","content":[{"type":"text","text":prompt_tpl}]})
            else:
                # InternVL3 自定义 multimodal 输入格式（trust_remote_code=True）
                content = [{"type":"text","text":prompt_tpl}]
                for t in tiles:
                    content.append({"type":"image","image":t})
                msgs.append({"role":"user","content":content})
        return msgs

    # 推理
    for rel_list in tqdm(list(batched(to_process, args.batch_size)), total=(len(to_process)+args.batch_size-1)//args.batch_size):
        try:
            msgs = batch_to_msgs(rel_list)
            with torch.no_grad():
                responses = model.batch_chat(
                    tok,
                    msgs,
                    role="user",
                    temperature=0.0,
                    max_new_tokens=args.max_new_tokens,
                    repetition_penalty=1.05,
                    generation_config=dict(max_new_tokens=args.max_new_tokens, do_sample=False),
                )
        except Exception as e:
            logging.error(f"batch_chat failed on {rel_list[0]}: {e}")
            responses = [""] * len(rel_list)

        # 解析 -> 直接写“文本标签”（不再映射 id）
        for rel, resp in zip(rel_list, responses):
            labs_norm = extract_labels(resp, LABELS_MATCH, topk=None)
            labs = [MATCH2CANON.get(x, x) for x in labs_norm]
            index[rel] = labs

        # 每批次持久化，稳一点
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

        time.sleep(0.05)

    logging.info(f"Done. Wrote {len(index)} entries to {out_path}")

if __name__ == "__main__":
    main()
