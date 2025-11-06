#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import glob
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Sequence, Optional

import torch
from PIL import Image
from tqdm import tqdm

from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration


# ----------------------------
# Dataset port & fixed vocabs
# ----------------------------

VOCABS: Dict[str, Sequence[str]] = {
    "NYUDv2_40": [
        "wall","floor","cabinet","bed","chair","sofa","table","door","window","bookshelf",
        "picture","counter","blinds","desk","shelves","curtain","dresser","pillow","mirror","floor mat",
        "clothes","ceiling","books","refrigerator","television","paper","towel","shower curtain","box","whiteboard",
        "person","night stand","toilet","sink","lamp","bathtub","bag"
    ],
    # Placeholder for future dataset port (leave empty for now)
    "SUNRGBD_PORT": [
        "wall","floor","cabinet","bed","chair","sofa","table","door","window","bookshelf",
        "picture","counter","blinds","desk","shelves","curtain","dresser","pillow","mirror","floor_mat",
        "clothes","ceiling","books","fridge","tv","paper","towel","shower_curtain","box","whiteboard",
        "person","night_stand","toilet","sink","lamp","bathtub","bag"
    ]
}

PROMPT_TEMPLATE = (
    "You are given one image and a FIXED label vocabulary.\n"
    "Goal: return ONLY a JSON array of strings (no code block, no prose) "
    "with UP TO {max_labels} labels that correspond to the LARGEST and MOST OBVIOUS regions in the image.\n"
    "Selection rules:\n"
    "- Include only labels that are obvious and "
    "  match with high confidence (clearly visible and reliable identifications).\n"
    "- Prioritize labels that occupy the most pixels.\n"
    "- Use EXACT spelling from the vocabulary; do NOT invent new labels.\n"
    "Vocabulary:\n{vocab_block}\n\n"
    "Output format example:\n[\"wall\", \"floor\", \"table\", \"sofa\", \"window\"]"
)

JSON_ARRAY_RE = re.compile(r"\[[^\[\]]*\]", re.DOTALL)


def build_messages(img: Image.Image, vocab: Sequence[str], max_labels: int) -> List[dict]:
    """Build a single-turn, single-image chat prompt."""
    vocab_block = ", ".join(vocab)
    prompt = PROMPT_TEMPLATE.format(vocab_block=vocab_block, max_labels=max_labels)
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def extract_json_array(text: str, allowed_vocab: Optional[List[str]] = None) -> List[str]:
    """
    Extract a JSON array of strings from raw model text with robust fallback.

    Two-layer fallback strategy (inspired by InternVL):
    1. Try JSON parsing (strict)
    2. If fails, search for vocabulary labels in text using regex (lenient)

    This prevents empty results when model outputs non-JSON text.
    """
    text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()

    # Layer 1: Try JSON parsing
    m = JSON_ARRAY_RE.search(text)
    cand = m.group(0) if m else text
    try:
        arr = json.loads(cand)
        if isinstance(arr, list):
            labels = [str(x).strip() for x in arr if str(x).strip()]
            if labels:  # JSON parse succeeded and got results
                return labels
    except Exception:
        pass

    # Layer 2: Fallback to vocabulary-based extraction (like InternVL)
    if allowed_vocab:
        # Search for vocabulary labels in the text (case-insensitive, word boundaries)
        found = []
        text_lower = " " + re.sub(r"[^a-z0-9\s_]", " ", text.lower()) + " "

        for label in allowed_vocab:
            # Try word boundary match first (more precise)
            pattern = r"\b" + re.escape(label.lower()) + r"\b"
            if re.search(pattern, text_lower):
                found.append(label)
            else:
                # Fallback: substring match (handles "floormat" -> "floor_mat")
                label_no_space = label.replace(" ", "").replace("_", "")
                text_no_space = text_lower.replace(" ", "").replace("_", "")
                if label_no_space in text_no_space:
                    found.append(label)

        # Remove duplicates while preserving order of appearance
        seen = set()
        unique_found = []
        for lab in found:
            if lab not in seen:
                seen.add(lab)
                unique_found.append(lab)

        if unique_found:
            return unique_found

    # Layer 3: Last resort lenient split (original fallback)
    items = re.split(r"[,;\n]+", cand)
    out = []
    for s in items:
        s = s.strip().strip("-*•").strip("\"' ").rstrip(".")
        if s:
            out.append(s)
    return out


def load_images(dataset_dir: str, image_folder: str) -> List[str]:
    """Collect absolute image paths under dataset_dir/image_folder."""
    base = Path(dataset_dir) / image_folder
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if not base.exists():
        raise FileNotFoundError(f"Image folder not found: {base}")
    paths = sorted(str(p) for p in base.rglob("*") if p.suffix.lower() in exts)
    if not paths:
        raise FileNotFoundError(f"No images found under: {base}")
    return paths


def make_batched_inputs(
    processor: AutoProcessor,
    images: List[Image.Image],
    vocab: Sequence[str],
    max_labels: int,
    device: torch.device
):
    """Build batched processor inputs from a list of PIL images."""
    messages_list = [build_messages(img, vocab, max_labels) for img in images]
    inputs = processor.apply_chat_template(
        messages_list,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,         # ★ 关键：对batch做padding
        truncation=False,
    )
    for k, v in list(inputs.items()):
        if hasattr(v, "to"):
            inputs[k] = v.to(device)
    return inputs


def parse_args():
    ap = argparse.ArgumentParser("Generate tags with Qwen3-VL-30B-A3B-Instruct (JSON only)")
    # Keep names & defaults consistent with the previous script:
    ap.add_argument("--model_id", type=str, default="Qwen/Qwen3-VL-30B-A3B-Instruct",
                    help="HF model id or local path")
    ap.add_argument("--dataset_dir", type=str, default=str(Path("datasets/NYUDepthv2").resolve()),
                    help="Dataset root directory")
    ap.add_argument("--image_folder", type=str, default="RGB",
                    help="Subfolder under dataset_dir to scan for images")
    ap.add_argument("--output_file", type=str, default="image_labels_vlm.json",
                    help="Output JSON file path")
    ap.add_argument("--batch_size", type=int, default=4,
                    help="Batch size for inference")
    ap.add_argument("--max_new_tokens", type=int, default=64,
                    help="Max new tokens for generation")
    # NEW but aligned with your request: limit number of labels (also enforced in prompt)
    ap.add_argument("--max_labels", type=int, default=5,
                    help="Maximum number of labels per image (prompt + post-filter)")
    ap.add_argument("--dataset_port", type=str, default="NYUDv2_40",
                    choices=list(VOCABS.keys()),
                    help="Which dataset vocabulary to use")
    return ap.parse_args()


def main():
    args = parse_args()
    ACTIVE_DATASET_PORT = args.dataset_port
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info(f"Active dataset port: {ACTIVE_DATASET_PORT}")

    vocab = VOCABS.get(ACTIVE_DATASET_PORT, [])
    if not vocab:
        logging.warning("Active vocabulary is empty. Please fill VOCABS[ACTIVE_DATASET_PORT] before running.")

    # Discover images
    img_paths = load_images(args.dataset_dir, args.image_folder)
    logging.info(f"Found {len(img_paths)} images under {Path(args.dataset_dir) / args.image_folder}")

    # Load model & processor
    load_kwargs = dict(dtype="auto", device_map="auto")
    logging.info(f"Loading model: {args.model_id}")
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(args.model_id, **load_kwargs)
    processor = AutoProcessor.from_pretrained(args.model_id)
    device = next(model.parameters()).device

    results: Dict[str, List[str]] = {}
    bs = max(1, int(args.batch_size))

    for i in tqdm(range(0, len(img_paths), bs), desc="Generating"):
        batch_paths = img_paths[i : i + bs]
        batch_imgs = [Image.open(p).convert("RGB") for p in batch_paths]

        # Build inputs and generate
        inputs = make_batched_inputs(processor, batch_imgs, vocab, args.max_labels, device)
        with torch.inference_mode():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=0.0,
            )

        attn = inputs["attention_mask"]  # (B, L)
        prompt_lens = attn.sum(dim=1).tolist()  # 每个样本真实长度（不含padding）

        trimmed = [gen_ids[j, int(prompt_lens[j]):] for j in range(len(batch_paths))]
        texts = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        # Parse and sanitize
        for abspath, raw in zip(batch_paths, texts):
            rel_key = str(Path(abspath).resolve().relative_to(Path(args.dataset_dir).resolve()))
            # Pass vocab to enable fallback extraction (Layer 2)
            labels = extract_json_array(raw, allowed_vocab=vocab)
            # Keep only labels from the fixed vocabulary; limit to max_labels
            if vocab:
                labels = [x for x in labels if x in vocab]
            # Deduplicate while preserving order
            seen, uniq = set(), []
            for s in labels:
                if s not in seen:
                    seen.add(s); uniq.append(s)
            results[rel_key] = uniq[: args.max_labels]

        # Close PIL images to release handles
        for img in batch_imgs:
            try:
                img.close()
            except Exception:
                pass

    # Write JSON
    out_path = Path(args.output_file)
    # 自动添加.json后缀（如果没有的话）
    if out_path.suffix.lower() != '.json':
        out_path = out_path.with_suffix('.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info(f"Done. Wrote {len(results)} entries to: {out_path}")


if __name__ == "__main__":
    main()