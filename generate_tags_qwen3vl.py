#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import glob
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Sequence

import torch
from PIL import Image
from tqdm import tqdm

from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration


# ----------------------------
# Dataset port & fixed vocabs
# ----------------------------
ACTIVE_DATASET_PORT = "NYUDv2_40"  # switch here later if needed (e.g., "SUNRGBD_PORT")

VOCABS: Dict[str, Sequence[str]] = {
    "NYUDv2_40": [
        "wall","floor","cabinet","bed","chair","sofa","table","door","window","bookshelf",
        "picture","counter","blinds","desk","shelves","curtain","dresser","pillow","mirror","floor mat",
        "clothes","ceiling","books","refrigerator","television","paper","towel","shower curtain","box","whiteboard",
        "person","night stand","toilet","sink","lamp","bathtub","bag"
    ],
    # Placeholder for future dataset port (leave empty for now)
    "SUNRGBD_PORT": []
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


def extract_json_array(text: str) -> List[str]:
    """Extract a JSON array of strings from raw model text with a robust fallback."""
    text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    m = JSON_ARRAY_RE.search(text)
    cand = m.group(0) if m else text
    try:
        arr = json.loads(cand)
        if isinstance(arr, list):
            return [str(x).strip() for x in arr if str(x).strip()]
    except Exception:
        pass
    # Fallback: lenient split (best-effort if the model adds extra text)
    items = re.split(r"[,;\n]+", cand)
    out = []
    for s in items:
        s = s.strip().strip("-*•").strip("“”\"' ").rstrip(".")
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
    return ap.parse_args()


def main():
    args = parse_args()

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

        # Trim per-sample: remove the prompt part from each sequence
        input_ids = inputs["input_ids"]
        trimmed = []
        for j in range(len(batch_paths)):
            j_len = input_ids[j].shape[0]
            trimmed.append(gen_ids[j, j_len:])

        texts = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        # Parse and sanitize
        for abspath, raw in zip(batch_paths, texts):
            rel_key = str(Path(abspath).resolve().relative_to(Path(args.dataset_dir).resolve()))
            labels = extract_json_array(raw)
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info(f"Done. Wrote {len(results)} entries to: {out_path}")


if __name__ == "__main__":
    main()
