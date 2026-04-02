"""
Generate corrupted versions of GT label JSON for robustness testing.

Usage:
python generate_corrupted_labels.py \
    --input datasets/NYUDepthv2/top6_labels_per_image.json \
    --vocab datasets/NYUDepthv2/nyu40_labels.txt \
    --replace_n 1 2 3 \
    --seeds 123 456 789\
    --output_dir datasets/NYUDepthv2/corrupted_labels/

This will generate:
    corrupted_labels/replace1_seed42.json
    corrupted_labels/replace1_seed123.json
    corrupted_labels/replace1_seed456.json
    corrupted_labels/replace2_seed42.json
    ... (3 replace levels x 3 seeds = 9 files)
"""

import json
import random
import argparse
import os
from copy import deepcopy


def load_vocab(vocab_path):
    """Load the 40-class vocabulary from txt file."""
    with open(vocab_path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]
    return labels


def corrupt_labels(gt_labels_dict, vocab, replace_n, seed):
    """
    For each image, replace `replace_n` GT labels with random incorrect ones.
    
    Rules:
    - Replacement is drawn from vocab MINUS the current image's GT labels
    - If replace_n >= len(labels), replace all but one (keep at least 1 correct)
    - Total number of labels per image stays the same
    """
    rng = random.Random(seed)
    corrupted = {}
    
    stats = {"total_images": 0, "total_replaced": 0, "total_labels": 0}
    
    for img_name, labels in gt_labels_dict.items():
        stats["total_images"] += 1
        stats["total_labels"] += len(labels)
        
        if len(labels) == 0:
            corrupted[img_name] = labels
            continue
        
        # How many to actually replace (cap at len-1 to keep at least 1 correct)
        actual_replace = min(replace_n, len(labels) - 1)
        if actual_replace <= 0:
            corrupted[img_name] = deepcopy(labels)
            continue
        
        # Pool of incorrect labels = vocab - current GT labels
        label_set = set(l.lower().strip() for l in labels)
        pool = [v for v in vocab if v.lower().strip() not in label_set]
        
        if len(pool) < actual_replace:
            actual_replace = len(pool)
        
        # Randomly select which positions to replace
        indices_to_replace = rng.sample(range(len(labels)), actual_replace)
        # Randomly select replacement labels
        replacements = rng.sample(pool, actual_replace)
        
        new_labels = deepcopy(labels)
        for idx, replacement in zip(indices_to_replace, replacements):
            new_labels[idx] = replacement
        
        corrupted[img_name] = new_labels
        stats["total_replaced"] += actual_replace
    
    return corrupted, stats


def main():
    parser = argparse.ArgumentParser(description="Generate corrupted label JSONs for robustness testing")
    parser.add_argument("--input", type=str, required=True, help="Path to GT labels JSON (e.g., top8_labels_per_image.json)")
    parser.add_argument("--vocab", type=str, required=True, help="Path to vocabulary txt (e.g., nyu40_labels.txt)")
    parser.add_argument("--replace_n", type=int, nargs="+", default=[1, 2, 3], help="Number of labels to replace per image")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456], help="Random seeds for reproducibility")
    parser.add_argument("--output_dir", type=str, default="corrupted_labels", help="Output directory")
    args = parser.parse_args()
    
    # Load data
    with open(args.input, "r", encoding="utf-8") as f:
        gt_labels = json.load(f)
    vocab = load_vocab(args.vocab)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loaded {len(gt_labels)} images, {len(vocab)} vocab classes")
    print(f"Will generate {len(args.replace_n)} x {len(args.seeds)} = {len(args.replace_n) * len(args.seeds)} corrupted files")
    print()
    
    for n in args.replace_n:
        for seed in args.seeds:
            corrupted, stats = corrupt_labels(gt_labels, vocab, n, seed)
            
            out_path = os.path.join(args.output_dir, f"replace{n}_seed{seed}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(corrupted, f, indent=2, ensure_ascii=False)
            
            avg_replaced = stats["total_replaced"] / max(stats["total_images"], 1)
            print(f"  replace_n={n}, seed={seed}: avg {avg_replaced:.1f} labels replaced/image -> {out_path}")
    
    print(f"\nDone! Generated files in {args.output_dir}/")
    print(f"\nTo run eval, change image_labels_json_path in your config to point to each corrupted file.")


if __name__ == "__main__":
    main()