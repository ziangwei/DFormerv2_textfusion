"""
Single-image inference script for DFormerv2 (text-fusion version).

Usage examples:

  # Basic: RGB + Depth, use default class names from config
  python run_infer.py \
    --config local_configs.NYUDepthv2.DFormerv2_B \
    --checkpoint checkpoints/best.pth \
    --rgb path/to/image.jpg \
    --depth path/to/depth.png

  # Provide a custom label list (JSON or TXT)
  python run_infer.py ... --labels my_labels.json

  # Pre-select which classes to highlight in the visualization
  python run_infer.py ... --select "wall,floor,ceiling"

  # Interactive mode: prints all available labels, lets you pick
  python run_infer.py ... --interactive

  # Disable text guidance entirely (pure visual inference)
  python run_infer.py ... --no-text
"""

import argparse
import json
import os
import sys
from importlib import import_module
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def normalize_image(img_hwc: np.ndarray, mean, std) -> np.ndarray:
    """Normalize HWC float32 image in [0,1]."""
    return (img_hwc - mean) / std


def load_rgb(path: str) -> np.ndarray:
    """Load BGR with cv2, return float32 RGB HWC in [0,1]."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read RGB image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb


def load_depth(path: str) -> np.ndarray:
    """Load depth as 3-channel float32 HWC in [0,1]."""
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError(f"Cannot read depth image: {path}")
    if d.dtype == np.uint16:
        d = (d.astype(np.float32) / 65535.0 * 255.0).astype(np.uint8)
    if len(d.shape) == 2:
        d = cv2.merge([d, d, d])
    elif d.shape[2] == 4:
        d = d[:, :, :3]
    return d.astype(np.float32) / 255.0


def preprocess(rgb_01: np.ndarray, depth_01: np.ndarray,
               norm_mean, norm_std) -> tuple:
    """
    Replicate ValPre normalization.
    Returns (rgb_tensor, depth_tensor) both [1, 3, H, W].
    """
    rgb_n = normalize_image(rgb_01, norm_mean, norm_std)
    depth_n = normalize_image(depth_01,
                              np.array([0.48, 0.48, 0.48]),
                              np.array([0.28, 0.28, 0.28]))
    rgb_t = torch.from_numpy(rgb_n.transpose(2, 0, 1)).float().unsqueeze(0)
    dep_t = torch.from_numpy(depth_n.transpose(2, 0, 1)).float().unsqueeze(0)
    return rgb_t, dep_t


def load_labels_from_file(path: str) -> list:
    """
    Load label names from:
      - JSON list:  ["wall", "floor", ...]
      - JSON dict:  {"RGB/0.jpg": ["wall", ...], ...}  → unique labels
      - TXT:        one label per line
    Returns deduplicated ordered list of strings.
    """
    ext = Path(path).suffix.lower()
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            labels = [str(x).strip() for x in data if str(x).strip()]
        elif isinstance(data, dict):
            seen, labels = set(), []
            for v in data.values():
                items = v if isinstance(v, list) else [v]
                for item in items:
                    s = str(item).strip()
                    if s and s not in seen:
                        labels.append(s)
                        seen.add(s)
        else:
            raise ValueError(f"Unsupported JSON structure in {path}")
    else:
        with open(path, "r", encoding="utf-8") as f:
            labels = [ln.strip() for ln in f if ln.strip()]
    return labels


def encode_label_list(labels: list, config) -> torch.Tensor:
    """
    Encode a list of label strings → [N, D] float32 tensor on CPU.
    Uses the same encode_labels_batch pipeline as training.
    """
    from utils.prompt_utils import encode_labels_batch
    embed_dict = encode_labels_batch(
        labels=labels,
        template_set=getattr(config, "text_template_set", "clip"),
        max_templates_per_label=getattr(config, "max_templates_per_label", 3),
        encoder=getattr(config, "text_encoder", "clip"),
        encoder_name=getattr(config, "text_encoder_name", None),
        target_dim=getattr(config, "text_feature_dim", 512),
    )
    ordered = []
    for lb in labels:
        key = lb.strip().lower()
        if key in embed_dict:
            ordered.append(embed_dict[key])
        else:
            dim = getattr(config, "text_feature_dim", 512)
            ordered.append(torch.zeros(dim))
    return torch.stack(ordered, dim=0).float()  # [N, D]


def interactive_select(labels: list) -> list:
    """
    Print all available labels and prompt the user to select a subset.
    Returns the selected label names.
    """
    print("\n" + "=" * 60)
    print("Available labels:")
    for i, lb in enumerate(labels):
        print(f"  [{i:2d}] {lb}")
    print("=" * 60)
    print("Enter class indices (e.g. '0 2 5'), names (e.g. 'wall,floor'),")
    print("or press Enter to use ALL classes.")
    raw = input("> ").strip()
    if not raw:
        print(f"→ Using all {len(labels)} classes.")
        return labels

    selected = []
    # Try parsing as indices first, then fall back to names
    tokens = [t.strip().strip(",") for t in raw.replace(",", " ").split() if t.strip()]
    for tok in tokens:
        if tok.isdigit():
            idx = int(tok)
            if 0 <= idx < len(labels):
                selected.append(labels[idx])
            else:
                print(f"  Warning: index {idx} out of range, skipped.")
        else:
            # Match by name substring (case-insensitive)
            matches = [lb for lb in labels if tok.lower() in lb.lower()]
            if matches:
                selected.extend(matches)
            else:
                print(f"  Warning: '{tok}' not found in label list, skipped.")
    # Deduplicate
    seen, unique = set(), []
    for lb in selected:
        if lb not in seen:
            unique.append(lb)
            seen.add(lb)
    print(f"→ Selected {len(unique)} class(es): {unique}")
    return unique


def colorize_prediction(pred_hw: np.ndarray, num_classes: int,
                        highlight_indices: set = None) -> np.ndarray:
    """
    Returns RGB uint8 image [H, W, 3].
    If highlight_indices is given, non-highlighted classes are shown dark gray.
    """
    cmap_path = Path(__file__).parent / "utils" / "nyucmap.npy"
    if cmap_path.exists():
        palette = np.load(str(cmap_path))         # [N, 3] uint8
        if palette.dtype != np.uint8:
            palette = (palette * 255).clip(0, 255).astype(np.uint8)
    else:
        # Auto-generate a distinguishable palette
        rng = np.random.RandomState(42)
        palette = rng.randint(50, 230, size=(max(num_classes, 256), 3), dtype=np.uint8)

    H, W = pred_hw.shape
    colored = np.zeros((H, W, 3), dtype=np.uint8)
    for cls in range(num_classes):
        mask = pred_hw == cls
        if not mask.any():
            continue
        if highlight_indices is not None and cls not in highlight_indices:
            colored[mask] = [40, 40, 40]          # dark gray for non-selected
        else:
            color = palette[cls % len(palette)]
            colored[mask] = color
    return colored


def print_class_report(pred_hw: np.ndarray, class_names: list,
                       highlight_indices: set = None):
    """Print pixel count and % for each present class."""
    total = pred_hw.size
    print("\n" + "─" * 55)
    print(f"{'ID':>3}  {'Class Name':<22}  {'Pixels':>8}  {'%':>6}")
    print("─" * 55)
    for cls_id, name in enumerate(class_names):
        count = int((pred_hw == cls_id).sum())
        if count == 0:
            continue
        marker = " ◀" if (highlight_indices and cls_id in highlight_indices) else ""
        print(f"{cls_id:>3}  {name:<22}  {count:>8,}  {count/total*100:>5.1f}%{marker}")
    print("─" * 55)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Single-image inference for DFormerv2-textfusion",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--config", required=True,
                        help="Config module path, e.g. local_configs.NYUDepthv2.DFormerv2_B")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--rgb", required=True,
                        help="Path to RGB image")
    parser.add_argument("--depth", required=True,
                        help="Path to depth image (single-channel PNG or 3-channel)")
    parser.add_argument("--labels", default=None,
                        help="Path to label list (JSON list or TXT, one per line).\n"
                             "If omitted, uses class_names from config.")
    parser.add_argument("--select", default=None,
                        help="Comma-separated class names to highlight, e.g. 'wall,floor,ceiling'.\n"
                             "Others shown dark-gray. Overrides --interactive.")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactively choose which classes to highlight.")
    parser.add_argument("--no-text", action="store_true",
                        help="Disable text guidance (pure visual inference).")
    parser.add_argument("--output", default="./infer_output",
                        help="Output directory (default: ./infer_output)")
    parser.add_argument("--device", default="cuda",
                        help="Device: cuda or cpu")
    args = parser.parse_args()

    # ── Config ──────────────────────────────
    config = getattr(import_module(args.config), "C")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.no_text:
        config.enable_text_guidance = False

    # ── Build model ─────────────────────────
    print(f"[1/5] Building model ({config.backbone})...")
    from models.builder import EncoderDecoder as segmodel
    model = segmodel(
        cfg=config,
        criterion=nn.CrossEntropyLoss(ignore_index=config.background),
        norm_layer=nn.BatchNorm2d,
        syncbn=False,
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    weights = ckpt.get("model", ckpt.get("state_dict", ckpt))
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if missing:
        print(f"  Missing keys  ({len(missing)}): {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        print(f"  Unexpected ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
    model.to(device).eval()
    print(f"  Checkpoint loaded: {args.checkpoint}")

    # ── Load images ─────────────────────────
    print(f"[2/5] Loading images...")
    rgb_01 = load_rgb(args.rgb)
    dep_01 = load_depth(args.depth)
    H, W = rgb_01.shape[:2]
    print(f"  RGB:   {args.rgb}  ({W}×{H})")
    print(f"  Depth: {args.depth}")

    rgb_t, dep_t = preprocess(rgb_01, dep_01, config.norm_mean, config.norm_std)
    rgb_t, dep_t = rgb_t.to(device), dep_t.to(device)

    # ── Text features ───────────────────────
    text_feats = None
    used_labels = []

    if config.enable_text_guidance:
        print(f"[3/5] Preparing text features...")

        # 1) Determine the full label list
        if args.labels:
            all_labels = load_labels_from_file(args.labels)
            print(f"  Loaded {len(all_labels)} labels from {args.labels}")
        else:
            all_labels = list(config.class_names)
            print(f"  Using {len(all_labels)} class names from config")

        # 2) Determine which subset to encode / highlight
        if args.select:
            sel_names = [s.strip() for s in args.select.split(",") if s.strip()]
            # Match to available labels (substring, case-insensitive)
            used_labels = []
            for sel in sel_names:
                matches = [lb for lb in all_labels if sel.lower() in lb.lower()]
                used_labels.extend(matches)
            if not used_labels:
                print(f"  Warning: --select matched nothing, falling back to all labels.")
                used_labels = all_labels
            else:
                print(f"  --select resolved to: {used_labels}")
        elif args.interactive:
            used_labels = interactive_select(all_labels)
        else:
            used_labels = all_labels

        # 3) Encode
        print(f"  Encoding {len(used_labels)} label(s) with "
              f"{config.text_encoder}/{config.text_encoder_name or 'default'}...")
        text_feats = encode_label_list(used_labels, config)  # [N, D]
        text_feats = text_feats.unsqueeze(0).to(device)      # [1, N, D]
        print(f"  text_features shape: {list(text_feats.shape)}")
    else:
        print(f"[3/5] Text guidance disabled — skipping.")

    # ── Inference ───────────────────────────
    print(f"[4/5] Running inference...")
    with torch.no_grad():
        if text_feats is not None:
            logits = model(rgb_t, dep_t, text_features=text_feats)  # [1, C, H, W]
        else:
            logits = model(rgb_t, dep_t)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)  # [H, W]

    print(f"  Output shape: {pred.shape}, classes present: "
          f"{sorted(np.unique(pred).tolist())}")

    # ── Highlight indices ────────────────────
    class_names = list(config.class_names)
    highlight_indices = None
    if used_labels and used_labels != all_labels if 'all_labels' in dir() else False:
        highlight_indices = set()
        for lb in used_labels:
            for idx, cn in enumerate(class_names):
                if lb.lower() in cn.lower() or cn.lower() in lb.lower():
                    highlight_indices.add(idx)

    # ── Save outputs ────────────────────────
    print(f"[5/5] Saving outputs to {out_dir}/")

    # Raw index map (grayscale PNG, value = class index 0–39)
    cv2.imwrite(str(out_dir / "pred_indices.png"), pred)

    # Colorized prediction
    colored = colorize_prediction(pred, config.num_classes, highlight_indices)
    colored_bgr = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_dir / "pred_color.png"), colored_bgr)

    # Overlay on original RGB
    orig_u8 = (rgb_01 * 255).clip(0, 255).astype(np.uint8)
    orig_bgr = cv2.cvtColor(orig_u8, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(orig_bgr, 0.5, colored_bgr, 0.5, 0)
    cv2.imwrite(str(out_dir / "pred_overlay.png"), overlay)

    # Also save the original inputs for reference
    cv2.imwrite(str(out_dir / "input_rgb.jpg"), cv2.imread(args.rgb))
    cv2.imwrite(str(out_dir / "input_depth.png"), cv2.imread(args.depth, cv2.IMREAD_UNCHANGED))

    print(f"  pred_indices.png  — raw class index map (0–{config.num_classes-1})")
    print(f"  pred_color.png    — colorized segmentation")
    print(f"  pred_overlay.png  — 50/50 blend with original")

    # Console report
    print_class_report(pred, class_names, highlight_indices)

    # Save JSON report
    report = {}
    total = pred.size
    for cls_id, name in enumerate(class_names):
        count = int((pred == cls_id).sum())
        if count > 0:
            report[name] = {"class_id": cls_id, "pixels": count,
                            "percent": round(count / total * 100, 2),
                            "highlighted": (highlight_indices is None or
                                            cls_id in (highlight_indices or set()))}
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Full report saved to {out_dir}/report.json")
    print("Done.")


if __name__ == "__main__":
    main()
