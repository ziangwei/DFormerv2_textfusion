#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NYUv2 Label Overlay Utility
---------------------------
Given an RGB image and a label map (per-pixel class ids), produce a colorized
label image and an RGB-overlay visualization using the *official* NYUv2 color map
(or any custom colormap you pass in). Useful for pipeline/flowchart figures when
you don't want to run the model but need consistent color blocks.

Supported colormap formats:
- JSON: {"0": [r,g,b], "1":[r,g,b], ...}  (keys can be str or int)
- TXT:  one row per class: "r g b" (space or comma separated); lines starting with '#' are ignored
- NPY:  numpy array of shape (K, 3) with dtype uint8 or int

CLI
---
python nyu_overlay.py \
  --rgb RGB/100.jpg \
  --label LABELS/100.png \
  --colormap nyu40_colors.json \
  --alpha 0.45 \
  --mode overlay \
  --id-shift 0 \
  --void-ids 0 255 \
  --draw-edges \
  --edge-color 255 255 255 \
  --edge-width 1 \
  --out out/100_overlay.png

Notes
-----
- If your label ids start from 1 (NYUv2-40 often uses 1..40), set --id-shift -1 so that id 1 maps to colormap index 0.
- If your "void / unlabeled" id should remain pure RGB, provide it via --void-ids.
- This script does NOT hardcode any NYUv2 colors; pass the official mapping you use to avoid mismatch.
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageFilter

def read_image_rgb(path: str) -> np.ndarray:
    img = Image.open(path).convert('RGB')
    return np.array(img, dtype=np.uint8)

def read_label(path: str) -> np.ndarray:
    # Supports 8-bit or 16-bit PNGs; we convert to int32
    lab = Image.open(path)
    if lab.mode not in ('I;16', 'I', 'L'):
        lab = lab.convert('I')  # generic integer mode
    lab_np = np.array(lab)
    return lab_np.astype(np.int32)

def load_colormap(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        # keys may be strings; convert to list ordered by index
        # Determine max index
        idxs = [int(k) for k in obj.keys()]
        kmax = max(idxs)
        table = np.zeros((kmax + 1, 3), dtype=np.uint8)
        for k, v in obj.items():
            i = int(k)
            if not isinstance(v, (list, tuple)) or len(v) != 3:
                raise ValueError(f'Colormap JSON values must be [r,g,b]; got {v} for key {k}')
            table[i] = np.array(v, dtype=np.uint8)
        return table
    elif ext == '.npy':
        arr = np.load(path)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError('NPY colormap must have shape (K, 3)')
        arr = np.clip(arr.astype(np.int32), 0, 255).astype(np.uint8)
        return arr
    elif ext in ('.txt', '.csv'):
        rows: List[List[int]] = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                # split on comma or whitespace
                parts = [p for p in line.replace(',', ' ').split() if p]
                if len(parts) < 3:
                    continue
                r, g, b = map(int, parts[:3])
                rows.append([r, g, b])
        if not rows:
            raise ValueError('No valid rows found in TXT colormap. Expect lines like: "r g b"')
        table = np.array(rows, dtype=np.uint8)
        return table
    else:
        raise ValueError(f'Unsupported colormap extension: {ext}')

def colorize_labels(labels: np.ndarray, cmap: np.ndarray, id_shift: int = 0, void_ids: Optional[List[int]] = None) -> np.ndarray:
    """
    Map integer labels to RGB via LUT. Unknown ids are modulo-clipped into range (for safety),
    but we try to keep them as close as possible by clipping to [0, K-1].
    """
    lab = labels.copy().astype(np.int32)
    if id_shift != 0:
        lab = lab + int(id_shift)

    K = cmap.shape[0]
    lab_clip = np.clip(lab, 0, K - 1)

    color = cmap[lab_clip]   # (H, W, 3) via advanced indexing if cmap is (K,3) and lab is (H,W)
    if color.ndim != 3:
        # If numpy advanced indexing doesn't broadcast (older versions), do flatten-reshape
        H, W = lab.shape
        color = cmap[lab_clip.reshape(-1)].reshape(H, W, 3)

    if void_ids:
        mask = np.zeros_like(lab, dtype=bool)
        for vid in void_ids:
            vid_shifted = vid + id_shift
            mask |= (lab == vid_shifted)
        # mark void as [0,0,0] so that overlay can keep original RGB for those pixels
        color = color.copy()
        color[mask] = 0
    return color.astype(np.uint8)

def compute_edges(labels: np.ndarray, kernel: int = 1) -> np.ndarray:
    """Return a boolean edge map by 4-neighborhood difference. kernel=1 gives 1px edges."""
    H, W = labels.shape
    e = np.zeros((H, W), dtype=bool)
    e[:, 1:] |= (labels[:, 1:] != labels[:, :-1])
    e[1:, :] |= (labels[1:, :] != labels[:-1, :])
    if kernel > 1:
        # PIL-based dilation (no SciPy): use MaxFilter with size=2k+1
        k = max(1, int(kernel))
        m = Image.fromarray((e * 255).astype('uint8'))
        m = m.filter(ImageFilter.MaxFilter(size=2 * k + 1))
        e = np.array(m) > 0
    return e

def overlay(rgb: np.ndarray,
            colorized: np.ndarray,
            labels: np.ndarray,
            alpha: float = 0.5,
            mode: str = 'overlay',
            void_black_is_transparent: bool = True,
            draw_edges: bool = False,
            edge_color: Tuple[int, int, int] = (255, 255, 255),
            edge_width: int = 1,
            edge_halo_width: int = 0,
            edge_halo_color: Tuple[int, int, int] = (255, 255, 255),
            boost: float = 1.0) -> np.ndarray:
    """
    mode:
      - 'overlay': blend rgb and colorized everywhere (except void=black if enabled)
      - 'blocks' : just show colorized blocks (void stays rgb if enabled)
      - 'rgb'    : return original rgb
    """
    assert rgb.shape[:2] == colorized.shape[:2] == labels.shape[:2], 'Size mismatch between RGB/label/colorized.'
    rgbf = rgb.astype(np.float32)
    colf = colorized.astype(np.float32)

    # Boost color vividness if requested
    if boost and abs(boost - 1.0) > 1e-6:
        colf = np.clip(colf * float(boost), 0, 255)
    # Saturation adjustment (luminance-preserving)
    if 'saturate' in locals():
        s = float(saturate)
        if abs(s - 1.0) > 1e-6:
            # compute luminance Y and interpolate toward/away from it
            Y = (0.299 * colf[...,0] + 0.587 * colf[...,1] + 0.114 * colf[...,2])[...,None]
            colf = Y + (colf - Y) * s
    # Value/brightness scale
    if 'value' in locals():
        v = float(value)
        if abs(v - 1.0) > 1e-6:
            colf = np.clip(colf * v, 0, 255)

    if mode == 'rgb':
        out = rgb.copy()
    elif mode == 'blocks':
        out = colf.copy()
        if void_black_is_transparent:
            # void (black) replaced by rgb at those pixels
            void_mask = (colorized[:, :, 0] == 0) & (colorized[:, :, 1] == 0) & (colorized[:, :, 2] == 0)
            out[void_mask] = rgbf[void_mask]
    else:  # 'overlay'
        out = (1.0 - alpha) * rgbf + alpha * colf
        if void_black_is_transparent:
            void_mask = (colorized[:, :, 0] == 0) & (colorized[:, :, 1] == 0) & (colorized[:, :, 2] == 0)
            out[void_mask] = rgbf[void_mask]

    out = np.clip(out, 0, 255).astype(np.uint8)

    if draw_edges:
        e = compute_edges(labels, kernel=max(1, int(edge_width)))
        out = out.copy()
        # halo first (thicker), then inner edge (thinner) for crisp paper-style look
        if edge_halo_width and edge_halo_width > 0:
            k = int(edge_halo_width)
            m = Image.fromarray((e * 255).astype('uint8')).filter(ImageFilter.MaxFilter(size=2 * k + 1))
            halo = (np.array(m) > 0)
            hr, hg, hb = [np.uint8(c) for c in edge_halo_color]
            out[halo] = np.array([hr, hg, hb], dtype=np.uint8)
        er, eg, eb = [np.uint8(c) for c in edge_color]
        out[e] = np.array([er, eg, eb], dtype=np.uint8)

    return out

def save_image(arr: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr).save(path)

def parse_args():
    p = argparse.ArgumentParser(description='NYUv2 label overlay utility')
    p.add_argument('--rgb', required=True, help='Path to RGB image')
    p.add_argument('--label', required=True, help='Path to label image (uint8/uint16 PNG)')
    p.add_argument('--colormap', required=True, help='Path to colormap file (JSON/TXT/NPY). Official NYUv2 mapping recommended.')
    p.add_argument('--id-shift', type=int, default=0, help='Shift applied to labels BEFORE lookup (e.g., -1 if your ids start at 1)')
    p.add_argument('--alpha', type=float, default=0.5, help='Blend factor for overlay mode')
    p.add_argument('--mode', type=str, default='overlay', choices=['overlay', 'blocks', 'rgb'], help='Visualization mode')
    p.add_argument('--style', type=str, default=None, choices=[None, 'paper', 'paper-soft', 'overlay-soft'], help='Style presets')
    p.add_argument('--boost', type=float, default=1.0, help='Multiply colorized RGB by this factor then clip (e.g., 1.1 更鲜艳; 0.95 更柔和)')
    p.add_argument('--saturate', type=float, default=1.0, help='Saturation multiplier on colorized blocks (e.g., 0.75 更柔和)')
    p.add_argument('--value', type=float, default=1.0, help='Brightness/value multiplier on colorized blocks (e.g., 0.92 稍暗)')
    p.add_argument('--edge-halo-width', type=int, default=0, help='Outer halo width (pixels). 0 to disable')
    p.add_argument('--edge-halo-color', type=int, nargs=3, default=[255,255,255], help='Outer halo color R G B')
    p.add_argument('--void-ids', type=int, nargs='*', default=[], help='Label ids treated as void (kept as RGB in output)')
    p.add_argument('--draw-edges', action='store_true', help='Draw label boundaries')
    p.add_argument('--edge-color', type=int, nargs=3, default=[255, 255, 255], help='Edge color R G B')
    p.add_argument('--edge-width', type=int, default=1, help='Edge width in pixels (approximate)')
    p.add_argument('--out', required=True, help='Output PNG path')
    return p.parse_args()

def main():
    args = parse_args()

    rgb = read_image_rgb(args.rgb)

    # Style presets
    if args.style == 'paper':
        args.mode = 'blocks'
        args.draw_edges = True
        args.edge_width = args.edge_width or 2
        args.edge_halo_width = args.edge_halo_width or 1
        args.edge_color = [0,0,0]
        args.edge_halo_color = [255,255,255]
        if args.boost == 1.0:
            args.boost = 1.1
    elif args.style == 'paper-soft':
        args.mode = 'blocks'
        args.draw_edges = True
        args.edge_width = args.edge_width or 2
        args.edge_halo_width = args.edge_halo_width or 1
        args.edge_color = [30,30,30]
        args.edge_halo_color = [255,255,255]
        args.boost = 1.0 if args.boost == 1.0 else args.boost
        args.saturate = 0.75 if args.saturate == 1.0 else args.saturate
        args.value = 0.95 if args.value == 1.0 else args.value
    elif args.style == 'overlay-soft':
        args.mode = 'overlay'
        args.alpha = min(args.alpha, 0.25)
        args.draw_edges = True
        args.edge_width = args.edge_width or 1
        args.edge_color = [0,0,0]
        args.edge_halo_width = args.edge_halo_width or 0
        args.boost = 1.0 if args.boost == 1.0 else args.boost
        args.saturate = 0.8 if args.saturate == 1.0 else args.saturate
        args.value = 0.95 if args.value == 1.0 else args.value

    labels = read_label(args.label)

    # Resize label to match RGB if needed (nearest)
    if labels.shape[:2] != rgb.shape[:2]:
        labels = np.array(Image.fromarray(labels).resize(rgb.shape[1::-1], resample=Image.NEAREST))

    cmap = load_colormap(args.colormap)  # (K,3)
    colorized = colorize_labels(labels, cmap, id_shift=args.id_shift, void_ids=args.void_ids)

    vis = overlay(
        rgb=rgb,
        colorized=colorized,
        labels=labels,
        alpha=args.alpha,
        mode=args.mode,
        void_black_is_transparent=True,
        draw_edges=args.draw_edges,
        edge_width=args.edge_width,
        edge_halo_width=args.edge_halo_width,
        boost=args.boost,
    )

    save_image(vis, args.out)
    print(f"Saved overlay to: {args.out}")

if __name__ == '__main__':
    main()
