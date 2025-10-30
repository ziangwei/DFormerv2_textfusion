#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute per-image intersection (or consensus) of text labels across two or more JSON files.
Each input JSON must be a dict: { "RGB/xxx.jpg": ["label1", "label2", ...], ... }.

Features
- Supports any number of JSON files (>=2).
- Canonicalizes labels by lowercasing and stripping whitespace (configurable).
- Optional synonym map to collapse different surface forms to the same canonical label.
- Choice of image set: only those common to all files, or the union of all images (missing treated as empty).
- Choice of output ordering: follow the first file's order or sort alphabetically.
- Two selection modes:
  * strict intersection (default): label must appear in ALL files for that image.
  * consensus threshold: label must appear in at least --min-count files (<= num_files).

Usage
------
Intersection (default):
    python intersect_labels.py out.json file1.json file2.json [file3.json ...]

Consensus (at least k files agree):
    python intersect_labels.py out.json file1.json file2.json --min-count 2

Alphabetical ordering and union image keys:
    python intersect_labels.py out.json f1.json f2.json --order alpha --image-keys union

With a synonym map (JSON: {"bookshelf": ["shelf","shelves"], "floor mat": ["rug","mat"]}):
    python intersect_labels.py out.json f1.json f2.json --synonyms synonyms.json

Author: ChatGPT (ZIANG WEI project)
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def canonicalize(label: str, do_lower=True, do_strip=True):
    if not isinstance(label, str):
        label = str(label)
    if do_strip:
        label = label.strip()
    if do_lower:
        label = label.lower()
    return label

def build_synonym_lookup(syn_path: Path, do_lower=True, do_strip=True):
    """
    Expects JSON like:
    {
      "bookshelf": ["shelf", "shelves"],
      "floor mat": ["rug", "mat"]
    }
    Returns dict mapping any variant -> canonical key ("bookshelf", "floor mat", ...).
    """
    with open(syn_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    lookup = {}
    for canonical, variants in m.items():
        cano = canonicalize(canonical, do_lower, do_strip)
        lookup[cano] = cano  # canonical maps to itself
        for v in variants:
            lookup[canonicalize(v, do_lower, do_strip)] = cano
    return lookup

def apply_synonym(label: str, syn_lookup: dict | None):
    if syn_lookup is None:
        return label
    return syn_lookup.get(label, label)

def collect_labels_for_image(files_data, image_key, do_lower, do_strip, syn_lookup):
    """
    Returns list of (file_index, ordered_labels_list, normalized_list)
    """
    per_file_lists = []
    for idx, data in enumerate(files_data):
        raw_list = data.get(image_key, [])
        if not isinstance(raw_list, list):
            # try to coerce unexpected format
            if raw_list is None:
                raw_list = []
            else:
                raw_list = [raw_list]
        # Preserve order as provided per file, but normalize for set logic
        norm = [apply_synonym(canonicalize(x, do_lower, do_strip), syn_lookup) for x in raw_list]
        # Deduplicate while preserving order (per file)
        seen = set()
        ordered_norm = []
        for x in norm:
            if x not in seen:
                seen.add(x)
                ordered_norm.append(x)
        per_file_lists.append((idx, raw_list, ordered_norm))
    return per_file_lists

def intersection_or_consensus(per_file_lists, min_count):
    """
    Compute labels that appear in >= min_count files for this image.
    Returns a set of labels (normalized strings).
    """
    cnt = Counter()
    for _, _, ordered_norm in per_file_lists:
        cnt.update(set(ordered_norm))  # presence, not multiplicity
    keep = {lab for lab, c in cnt.items() if c >= min_count}
    return keep

def order_output(labels_set, order_mode, per_file_lists):
    if order_mode == "alpha":
        return sorted(labels_set)
    # "first": follow first file's original order (normalized)
    first_norm_order = per_file_lists[0][2]
    order_map = {lab: i for i, lab in enumerate(first_norm_order)}
    return sorted(labels_set, key=lambda x: order_map.get(x, 10**9))

def main():
    ap = argparse.ArgumentParser(
        description="Compute per-image intersection/consensus of labels across multiple JSONs."
    )
    ap.add_argument("out_json", type=Path, help="Output JSON path")
    ap.add_argument("inputs", nargs="+", type=Path, help="Input JSON files (>=2)")
    ap.add_argument("--min-count", type=int, default=None,
                    help="Minimum number of files that must contain a label for it to be kept. "
                         "Default: all files (strict intersection).")
    ap.add_argument("--image-keys", choices=["common", "union"], default="common",
                    help="Which image keys to evaluate: only those common to all files, or the union.")
    ap.add_argument("--order", choices=["first", "alpha"], default="first",
                    help="Order output labels by the first file's order or alphabetically.")
    ap.add_argument("--no-lower", action="store_true", help="Disable lowercasing during canonicalization.")
    ap.add_argument("--no-strip", action="store_true", help="Disable whitespace stripping during canonicalization.")
    ap.add_argument("--synonyms", type=Path, default=None,
                    help="Optional JSON mapping canonical-> [variants,...] to collapse synonyms.")
    args = ap.parse_args()

    if len(args.inputs) < 2:
        ap.error("Please provide at least two input JSON files.")

    do_lower = not args.no_lower
    do_strip = not args.no_strip
    syn_lookup = build_synonym_lookup(args.synonyms, do_lower, do_strip) if args.synonyms else None

    # Load all files
    files_data = [load_json(p) for p in args.inputs]

    # Determine image keys
    key_sets = [set(d.keys()) for d in files_data]
    if args.image_keys == "common":
        image_keys = set.intersection(*key_sets)
    else:
        image_keys = set.union(*key_sets)

    n_files = len(files_data)
    min_count = args.min_count if args.min_count is not None else n_files

    out = {}
    empty_count = 0
    for k in sorted(image_keys):
        per_file_lists = collect_labels_for_image(files_data, k, do_lower, do_strip, syn_lookup)
        kept = intersection_or_consensus(per_file_lists, min_count=min_count)
        ordered = order_output(kept, args.order, per_file_lists)
        out[k] = ordered
        if not ordered:
            empty_count += 1

    # Write output
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # Print a brief summary
    print(f"Processed {len(image_keys)} images from {n_files} files.")
    print(f"Selection mode: labels appearing in >= {min_count} files.")
    print(f"Image key set: {args.image_keys}. Ordering: {args.order}.")
    print(f"Empty intersections: {empty_count} images.")
    # Preview a few examples
    shown = 0
    for k in sorted(image_keys):
        print(f"{k}: {out[k]}")
        shown += 1
        if shown >= 5:
            break

if __name__ == "__main__":
    main()
