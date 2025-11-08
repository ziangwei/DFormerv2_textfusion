import argparse
import importlib
import os
import json
import re
import sys
import time
from importlib import import_module

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
from scipy.ndimage import gaussian_filter

from models.builder import EncoderDecoder as segmodel
from models.blocks.semantic_alignment import SemanticAlignmentModule
from utils.dataloader.dataloader import ValPre, get_train_loader, get_val_loader
from utils.dataloader.RGBXDataset import RGBXDataset
from utils.engine.engine import Engine
from utils.engine.logger import get_logger
from utils.metric import compute_score, hist_info
from utils.pyt_utils import all_reduce_tensor, ensure_dir, link_file, load_model, parse_devices
from utils.val_mm import evaluate, evaluate_msf
from utils.visualize import print_iou, show_img


class SubsetDataset(torch.utils.data.Dataset):
    """Wrapper to select a subset of dataset by indices"""
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

parser = argparse.ArgumentParser()
parser.add_argument("--config", help="train config file path")
parser.add_argument("--gpus", help="used gpu number")
parser.add_argument("-v", "--verbose", default=False, action="store_true")
parser.add_argument("--epochs", default=0)
parser.add_argument("--show_image", "-s", default=False, action="store_true")
parser.add_argument("--save_path", default=None)
parser.add_argument("--checkpoint_dir")
parser.add_argument("--continue_fpath")


# --- text guidance runtime switches ---
parser.add_argument("--text-source", choices=["labels", "captions", "both", "imglabels"])
parser.add_argument("--text-encoder", choices=["clip", "jinaclip"])
parser.add_argument("--text-encoder-name", type=str)
parser.add_argument("--text-feature-dim", type=int)
parser.add_argument("--label-txt-path", type=str)
parser.add_argument("--caption-json-path", type=str)
parser.add_argument("--text-template-set", choices=["clip", "jinaclip", "none"])
parser.add_argument("--max-templates-per-label", type=int)
parser.add_argument("--max-caption-sentences", type=int)
parser.add_argument("--caption-topk", type=int)
parser.add_argument("--caption-topk-mode", choices=["class_sim", "firstk", "lenk"])
parser.add_argument("--image-labels-json-path", type=str)

# --- SAM per-stage switches ---
parser.add_argument("--sam-enc-stages", type=str, help="Comma separated, e.g., 0,2")
parser.add_argument("--sam-dec-stages", type=str, help="Comma separated, e.g., 1,3")
parser.add_argument("--superpower", default=None, action=argparse.BooleanOptionalAction)

# --- Attention visualization switches (Enhanced) ---
parser.add_argument("--save-attention", action="store_true",
                    help="Save attention maps for visualization")
parser.add_argument("--save-predictions", action="store_true",
                    help="Save segmentation predictions (works without text guidance)")
parser.add_argument("--vis-stage", type=str, default="enc",
                    choices=["enc", "dec"],
                    help="Visualize encoder or decoder attention")
parser.add_argument("--vis-stage-idx", type=str, default="0",
                    help="Which stage index(es) to visualize. Single: '0', Multiple: '0,1,2', All: 'all'")
parser.add_argument("--vis-block-idx", type=int, default=-1,
                    help="Which block in the stage (-1 for last block)")
parser.add_argument("--vis-aggregate", type=str, default="none",
                    choices=["none", "mean", "max", "weighted"],
                    help="How to aggregate attention from multiple stages (none=save separately)")
parser.add_argument("--num-images", type=int, default=None,
                    help="Number of images to process for visualization (None=all)")
parser.add_argument("--random-select", action="store_true",
                    help="Randomly select images instead of sequential selection")
parser.add_argument("--image-indices", type=str, default=None,
                    help="Comma-separated list of image indices to process (e.g., '0,5,10')")
parser.add_argument("--image-paths", type=str, default=None,
                    help="Path to text file containing image paths to process (one per line)")
parser.add_argument("--attention-alpha", type=float, default=0.6,
                    help="Alpha blending factor for overlay visualization (0-1)")
parser.add_argument("--attention-threshold", type=float, default=0.15,
                    help="Zero-out low responses in [0,1] after normalization (0=no threshold)")
parser.add_argument("--attention-smooth", type=float, default=1.5,
                    help="Gaussian smoothing sigma for attention maps (0=no smoothing)")
parser.add_argument("--max-token-vis", type=int, default=64,
                    help="Max tokens to save per image to avoid explosion")
parser.add_argument("--filter-tokens", type=str, default=None,
                    help="Comma-separated list of token names to visualize (e.g., 'floor,wall,ceiling'). "
                         "If set, only these tokens will be visualized.")

# --- Enhanced visualization options ---
parser.add_argument("--vis-competition", type=str, default='softmax',
                    choices=['none', 'softmax', 'margin', 'ratio'],
                    help="Competitive normalization mode to highlight token-specific regions. "
                         "'softmax' (recommended) uses temperature-scaled softmax across all tokens, "
                         "'margin' keeps only regions where the token dominates, "
                         "'ratio' computes ratio vs other tokens, "
                         "'none' disables competitive normalization")
parser.add_argument("--vis-competition-tau", type=float, default=2.0,
                    help="Temperature for softmax competitive normalization (higher = sharper)")
parser.add_argument("--vis-colormap", type=str, default='turbo',
                    choices=['turbo', 'plasma', 'viridis', 'jet', 'magma', 'inferno'],
                    help="Colormap for attention heatmaps. 'turbo' and 'plasma' are recommended for CVPR-style visuals")
parser.add_argument("--vis-gamma", type=float, default=0.75,
                    help="Gamma correction for attention maps (< 1.0 brightens, > 1.0 darkens, 1.0 = no change)")
parser.add_argument("--vis-grid", action='store_true',
                    help="Overlay light grid on attention maps for structure visualization (default: disabled)")
logger = get_logger()


def _parse_stages(stage_string):
    if stage_string is None:
        return None
    stage_string = str(stage_string).strip()
    if not stage_string:
        return []
    result = []
    for chunk in stage_string.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            result.append(int(chunk))
        except ValueError:
            continue
    return result


def _slugify(text: str) -> str:
    if text is None:
        return "token"
    text = str(text).strip()
    text = re.sub(r"[^0-9A-Za-z\u4e00-\u9fa5._-]+", "_", text)  # 保留中英文、数字、._-
    text = re.sub(r"_{2,}", "_", text)
    return (text or "token")[:120]


def _normalize01(arr: np.ndarray) -> np.ndarray:
    mn, mx = float(np.nanmin(arr)), float(np.nanmax(arr))
    if mx - mn < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    out = (arr - mn) / (mx - mn)
    return out.astype(np.float32)


def compute_competitive_map(attn_hwT: np.ndarray, token_idx: int, mode: str = 'softmax', tau: float = 2.0) -> np.ndarray:
    """
    Compute competitive normalization for a specific token to highlight its unique regions.

    Args:
        attn_hwT: (H, W, T) attention maps for all tokens
        token_idx: index of the target token
        mode: 'softmax' | 'margin' | 'ratio'
            - softmax: exp(tau*A_t) / sum_j exp(tau*A_j)  [recommended]
            - margin: A_t - max_{j≠t} A_j
            - ratio: A_t / (sum_{j≠t} A_j + eps)
        tau: temperature for softmax mode (higher = sharper)

    Returns:
        competitive_map: (H, W) in [0, 1]
    """
    H, W, T = attn_hwT.shape
    if token_idx < 0 or token_idx >= T:
        return np.zeros((H, W), dtype=np.float32)

    A_t = attn_hwT[..., token_idx]  # (H, W)

    if mode == 'softmax':
        # Numerically stable softmax with temperature
        attn_scaled = attn_hwT * tau  # (H, W, T)
        max_vals = np.max(attn_scaled, axis=-1, keepdims=True)  # (H, W, 1)
        exp_vals = np.exp(attn_scaled - max_vals)  # (H, W, T)
        sum_exp = np.sum(exp_vals, axis=-1)  # (H, W)
        competitive = exp_vals[..., token_idx] / (sum_exp + 1e-8)  # (H, W)

    elif mode == 'margin':
        # Keep only regions where token_idx is clearly dominant
        others_mask = np.ones(T, dtype=bool)
        others_mask[token_idx] = False
        A_others = attn_hwT[..., others_mask]  # (H, W, T-1)
        max_others = np.max(A_others, axis=-1)  # (H, W)
        competitive = A_t - max_others
        competitive = np.maximum(competitive, 0.0)  # Clip negative

    elif mode == 'ratio':
        # Ratio of target vs all others
        others_mask = np.ones(T, dtype=bool)
        others_mask[token_idx] = False
        A_others = attn_hwT[..., others_mask]  # (H, W, T-1)
        sum_others = np.sum(A_others, axis=-1)  # (H, W)
        competitive = A_t / (sum_others + 1e-8)

    else:
        raise ValueError(f"Unknown competitive mode: {mode}")

    # Normalize to [0, 1]
    return _normalize01(competitive)


def _save_single_token_map(attn: np.ndarray, rgb: np.ndarray, out_prefix: str,
                           alpha: float = 0.5, threshold: float = 0.0, smooth_sigma: float = 0.0,
                           colormap: str = 'turbo', gamma: float = 0.75, enable_grid: bool = False):
    """
    Modern attention visualization with enhanced colormap and processing.

    attn: (H, W) in [0,1]
    rgb : (H, W, 3) uint8 (RGB)
    colormap: 'turbo' | 'plasma' | 'viridis' | 'jet' | 'magma' | 'inferno'
    gamma: gamma correction for brightness (< 1.0 brightens, > 1.0 darkens)
    enable_grid: overlay light grid for structure visualization
    """
    import matplotlib
    matplotlib.use('Agg')  # 无GUI后端
    import matplotlib.pyplot as plt

    H, W = attn.shape

    # Step 1: Gaussian smoothing (reduce pixel-level noise)
    if smooth_sigma and smooth_sigma > 0.0:
        attn = gaussian_filter(attn, sigma=float(smooth_sigma))
        attn = _normalize01(attn)

    # Step 2: Gamma correction (enhance visibility)
    if gamma and gamma > 0.0 and gamma != 1.0:
        attn = np.power(attn, gamma)
        attn = _normalize01(attn)

    # Step 3: Threshold low responses
    if threshold and threshold > 0.0:
        attn = np.where(attn >= threshold, attn, 0.0)

    # Step 4: Apply colormap
    cmap = plt.get_cmap(colormap)
    heat_color = (cmap(attn)[:, :, :3] * 255).astype(np.uint8)  # RGB

    # Step 5: Blend with original image
    overlay = (alpha * heat_color + (1 - alpha) * rgb).astype(np.uint8)

    # Step 6: Slight contrast enhancement (CVPR-style)
    overlay = cv2.convertScaleAbs(overlay, alpha=1.05, beta=5)

    # Step 7: Optional grid overlay
    if enable_grid:
        grid_step = 16
        grid_color = 255
        grid_intensity = 0.05
        # Horizontal lines
        overlay[::grid_step, :] = (1 - grid_intensity) * overlay[::grid_step, :] + grid_intensity * grid_color
        # Vertical lines
        overlay[:, ::grid_step] = (1 - grid_intensity) * overlay[:, ::grid_step] + grid_intensity * grid_color
        overlay = overlay.astype(np.uint8)

    # Save
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_prefix + "_attn.png", overlay_bgr)


def visualize_attention_maps_enhanced(attn_hwT: torch.Tensor,
                                      rgb_np: np.ndarray,
                                      save_prefix: str,
                                      token_names=None,
                                      token_types=None,
                                      alpha: float = 0.5,
                                      threshold: float = 0.0,
                                      smooth_sigma: float = 0.0,
                                      max_tokens: int = 64,
                                      filter_tokens=None,
                                      competition_mode: str = 'softmax',
                                      competition_tau: float = 2.0,
                                      colormap: str = 'turbo',
                                      gamma: float = 0.75,
                                      enable_grid: bool = False):
    """
    attn_hwT: (H, W, T) torch or np
    rgb_np  : (H, W, 3) uint8 RGB
    competition_mode: 'none' | 'softmax' | 'margin' | 'ratio'
    """
    if isinstance(attn_hwT, torch.Tensor):
        attn_np = attn_hwT.detach().cpu().numpy()
    else:
        attn_np = np.asarray(attn_hwT)

    if attn_np.ndim != 3:
        logger.warning(f"Unexpected attention shape {attn_np.shape}, skip.")
        return

    H, W, T = attn_np.shape
    # 构建最终 token 名称表（优先使用 batch meta）
    pad_markers = {"", "<pad>", "[pad]", "pad", "<none>", "none"}
    names = []
    types = []
    for t in range(T):
        nm = (token_names[t] if (token_names and t < len(token_names)) else f"token{t}") if token_names else f"token{t}"
        tp = (token_types[t] if (token_types and t < len(token_types)) else "") if token_types else ""
        if isinstance(nm, bytes):
            nm = nm.decode("utf-8", errors="ignore")
        if isinstance(tp, bytes):
            tp = tp.decode("utf-8", errors="ignore")
        names.append(nm)
        types.append(tp)

    # 如果设置了 filter_tokens，优先过滤指定的 token
    if filter_tokens is not None and len(filter_tokens) > 0:
        # 将过滤列表转为小写，方便匹配
        filter_set = {ft.lower().strip() for ft in filter_tokens}
        selected = []
        for t in range(T):
            nm = names[t].strip() if isinstance(names[t], str) else str(names[t])
            if nm.lower() in filter_set:
                selected.append(t)
        if len(selected) == 0:
            logger.warning(f"No tokens matched filter {filter_tokens}, falling back to energy-based selection")
            # 回退到能量选择
            filter_tokens = None

    # 如果没有设置 filter_tokens 或过滤后为空，使用能量选择
    if filter_tokens is None or len(selected) == 0:
        energy = []
        for t in range(T):
            a = attn_np[..., t]
            a = _normalize01(a)
            score = float(a.mean())
            energy.append((score, t))
        energy.sort(reverse=True)
        selected = [t for _, t in energy[: min(T, max_tokens)]]

    for t in selected:
        nm = names[t].strip() if isinstance(names[t], str) else str(names[t])
        if nm.lower() in pad_markers:
            # 跳过 padding token
            continue
        tp = types[t].strip() if isinstance(types[t], str) else ""
        tag = f"{_slugify(tp)}_{_slugify(nm)}" if tp else _slugify(nm)

        # Apply competitive normalization if enabled
        if competition_mode and competition_mode.lower() != 'none':
            a = compute_competitive_map(attn_np, t, mode=competition_mode, tau=competition_tau)
        else:
            a = _normalize01(attn_np[..., t])

        out_prefix = f"{save_prefix}__{tag}"
        _save_single_token_map(
            a, rgb_np, out_prefix,
            alpha=alpha, threshold=threshold, smooth_sigma=smooth_sigma,
            colormap=colormap, gamma=gamma, enable_grid=enable_grid
        )


def extract_attention_from_model_enhanced(model, vis_stage="enc", stage_indices=None, block_idx=-1):
    """
    返回列表 [(attn(B,N,Hh,T), spatial_shape(h,w), stage_info), ...]
    stage_indices: list of int, or None for all stages
    """
    if hasattr(model, 'module'):
        actual_model = model.module
    else:
        actual_model = model

    attention_data = []

    if vis_stage == "enc":
        backbone = actual_model.backbone
        if hasattr(backbone, 'encoder_sam_blocks'):
            # 如果没有指定 stage，提取所有
            if stage_indices is None:
                stage_indices = list(range(len(backbone.encoder_sam_blocks)))

            for stage_idx in stage_indices:
                if stage_idx >= len(backbone.encoder_sam_blocks):
                    logger.warning(
                        f"Encoder stage {stage_idx} not found! Available: 0-{len(backbone.encoder_sam_blocks) - 1}")
                    continue

                blocks = backbone.encoder_sam_blocks[stage_idx]
                if len(blocks) == 0:
                    logger.warning(f"Encoder stage {stage_idx} has no SAM blocks!")
                    continue

                if block_idx == -1:
                    pick = len(blocks) - 1
                else:
                    pick = min(block_idx, len(blocks) - 1)

                sam_block = blocks[pick]
                if hasattr(sam_block, 'last_attention_map'):
                    attn = sam_block.last_attention_map
                    spatial_shape = getattr(sam_block, 'last_spatial_shape', None)
                    if attn is not None:
                        stage_info = f"enc_stage{stage_idx}_block{pick}"
                        attention_data.append((attn, spatial_shape, stage_info))
                        logger.info(f"✓ Extracted {stage_info}, shape={tuple(attn.shape)}, spatial={spatial_shape}")

    elif vis_stage == "dec":
        decode_head = actual_model.decode_head
        if hasattr(decode_head, 'dec_sam_layers'):
            # 如果没有指定 stage，提取所有
            if stage_indices is None:
                stage_indices = list(range(len(decode_head.dec_sam_layers)))

            for stage_idx in stage_indices:
                if stage_idx >= len(decode_head.dec_sam_layers):
                    logger.warning(
                        f"Decoder stage {stage_idx} not found! Available: 0-{len(decode_head.dec_sam_layers) - 1}")
                    continue

                sam_layer = decode_head.dec_sam_layers[stage_idx]
                if sam_layer is None:
                    logger.warning(f"Decoder stage {stage_idx} has no SAM layer (disabled)!")
                    continue

                if hasattr(sam_layer, 'last_attention_map'):
                    attn = sam_layer.last_attention_map
                    spatial_shape = getattr(sam_layer, 'last_spatial_shape', None)
                    if attn is not None:
                        stage_info = f"dec_stage{stage_idx}"
                        attention_data.append((attn, spatial_shape, stage_info))
                        logger.info(f"✓ Extracted {stage_info}, shape={tuple(attn.shape)}, spatial={spatial_shape}")

    if not attention_data:
        return [(None, None, "No attention found")]

    return attention_data


def aggregate_attention_maps(attn_list, mode="mean"):
    """
    聚合多个 stage 的 attention maps
    attn_list: list of (attn, spatial_shape, stage_info)
    mode: "mean", "max", or "weighted"
    Returns: (aggregated_attn, spatial_shape, "aggregated")
    """
    if not attn_list or len(attn_list) == 0:
        return None, None, "empty"

    # 过滤掉 None
    valid_attn = [(a, s, i) for a, s, i in attn_list if a is not None]
    if len(valid_attn) == 0:
        return None, None, "no_valid"

    if len(valid_attn) == 1:
        return valid_attn[0]

    # 获取目标形状（使用第一个有效 attention 的形状）
    target_attn, target_shape, _ = valid_attn[0]
    B, N, H, T = target_attn.shape

    # 调整所有 attention 到相同形状
    normalized_attns = []
    for attn, spatial_shape, _ in valid_attn:
        # 如果形状不匹配，需要插值
        if attn.shape != (B, N, H, T):
            # 这里简化处理：如果token数不同，跳过
            if attn.shape[0] != B or attn.shape[-1] != T:
                logger.warning(f"Skipping attention with incompatible shape {attn.shape} vs {(B, N, H, T)}")
                continue
            # 如果空间大小不同，插值
            if attn.shape[2] != H:
                # 需要插值到目标大小
                attn_reshaped = attn.reshape(B, N, int(np.sqrt(attn.shape[2])), int(np.sqrt(attn.shape[2])), T)
                attn_reshaped = F.interpolate(
                    attn_reshaped.permute(0, 1, 4, 2, 3),  # (B, N, T, h, w)
                    size=(int(np.sqrt(H)), int(np.sqrt(H))),
                    mode='bilinear',
                    align_corners=False
                ).permute(0, 1, 3, 4, 2).reshape(B, N, H, T)
                normalized_attns.append(attn_reshaped)
            else:
                normalized_attns.append(attn)
        else:
            normalized_attns.append(attn)

    if len(normalized_attns) == 0:
        return valid_attn[0]

    # 聚合
    stacked = torch.stack(normalized_attns, dim=0)  # (num_stages, B, N, H, T)

    if mode == "mean":
        aggregated = stacked.mean(dim=0)
    elif mode == "max":
        aggregated = stacked.max(dim=0)[0]
    elif mode == "weighted":
        # 使用注意力强度作为权重
        weights = stacked.sum(dim=(2, 3, 4), keepdim=True)  # (num_stages, B, 1, 1, 1)
        weights = F.softmax(weights, dim=0)
        aggregated = (stacked * weights).sum(dim=0)
    else:
        aggregated = stacked.mean(dim=0)

    stage_names = "_".join([info.split("_")[-1] for _, _, info in valid_attn[:3]])
    if len(valid_attn) > 3:
        stage_names += f"_and_{len(valid_attn)-3}more"

    return aggregated, target_shape, f"aggregated_{mode}_{stage_names}"


def load_class_names(config):
    """尝试全局类别名称（仅作为元数据缺失时的后备）。"""
    token_names = None
    if hasattr(config, 'label_txt_path') and config.label_txt_path:
        try:
            with open(config.label_txt_path, 'r', encoding="utf-8") as f:
                token_names = [line.strip() for line in f if line.strip()]
            logger.info(f"✓ Loaded {len(token_names)} class names from {config.label_txt_path}")
        except Exception as e:
            logger.warning(f"Failed to load class names: {e}")
    return token_names


@torch.no_grad()
def evaluate_with_attention(model, dataloader, config, device, engine,
                            save_dir=None, vis_stage="enc", stage_indices=None, block_idx=-1,
                            num_images=None, alpha=0.5, threshold=0.0, smooth_sigma=0.0, max_token_vis=64,
                            filter_tokens=None, competition_mode='softmax', competition_tau=2.0,
                            colormap='turbo', gamma=0.75, enable_grid=False, save_predictions=True,
                            aggregate_mode="none"):
    """
    注意力可视化 + 指标计算
    stage_indices: list of int, or None for all stages
    aggregate_mode: "none" (save separately), "mean", "max", "weighted"
    """
    from utils.metrics_new import Metrics

    logger.info("=" * 100)
    logger.info(f"Starting ENHANCED attention visualization")
    logger.info(f"  Stage: {vis_stage.upper()}")
    logger.info(f"  Stage Indices: {stage_indices if stage_indices else 'all'}")
    logger.info(f"  Block Index: {block_idx} (-1=last)")
    logger.info(f"  Aggregate Mode: {aggregate_mode}")
    logger.info(f"  Num Images: {num_images if num_images else 'all'}")
    logger.info(f"  Alpha: {alpha}, Threshold: {threshold}, Smooth Sigma: {smooth_sigma}")
    logger.info(f"  Save Dir: {save_dir}")
    logger.info("=" * 100)

    if not getattr(config, 'enable_text_guidance', False):
        logger.error("=" * 80)
        logger.error("ERROR: enable_text_guidance is False!")
        logger.error("Attention visualization requires text guidance to be enabled.")
        logger.error("=" * 80)
        raise ValueError("Cannot visualize attention without text guidance enabled")

    model.eval()
    n_classes = config.num_classes
    metrics = Metrics(n_classes, config.background, device)

    # 开启 SAM 缓存注意力
    def enable_attention_save(m):
        if isinstance(m, SemanticAlignmentModule):
            m.save_attention = True

    model.apply(enable_attention_save)

    # 全局回退 token 名称
    global_token_names = load_class_names(config)
    processed_images = 0

    # Note: num_images filtering is now handled at DataLoader level, not here
    # This parameter is kept for backward compatibility with attention visualization
    for idx, minibatch in enumerate(dataloader):

        if ((idx + 1) % int(max(len(dataloader) * 0.5, 1)) == 0 or idx == 0) and (
                (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed)
        ):
            print(f"Validation Iter: {idx + 1} / {len(dataloader)}")

        # 数据准备
        images = minibatch["data"]
        labels = minibatch["label"]
        modal_xs = minibatch["modal_x"]
        text_feats = minibatch.get("text_features")
        token_meta_batch = minibatch.get("text_token_meta")

        if len(images.shape) == 3:
            images = images.unsqueeze(0)
        if len(modal_xs.shape) == 3:
            modal_xs = modal_xs.unsqueeze(0)
        if len(labels.shape) == 2:
            labels = labels.unsqueeze(0)

        images_gpu = images.to(device)
        modal_xs_gpu = modal_xs.to(device)
        labels_gpu = labels.to(device)

        if text_feats is not None:
            text_feats = text_feats.to(device).float()
        else:
            logger.warning("text_feats is None! Skipping this batch...")
            continue

        # Forward
        preds = model(images_gpu, modal_xs_gpu, text_features=text_feats)

        # 抽取注意力映射
        attn_data = extract_attention_from_model_enhanced(model, vis_stage, stage_indices, block_idx)

        # 如果需要聚合多个 stage 的 attention
        if aggregate_mode != "none" and len(attn_data) > 1:
            aggregated = aggregate_attention_maps(attn_data, mode=aggregate_mode)
            # 保留聚合结果 + 原始结果
            attn_data_to_save = [aggregated] + attn_data
        else:
            attn_data_to_save = attn_data

        # 更新指标
        metrics.update(preds.softmax(dim=1), labels_gpu)

        # 获取调色板（提前准备，后面保存分割结果时用）
        if config.dataset_name in ["NYUDepthv2", "SUNRGBD"]:
            try:
                palette = np.load("./utils/nyucmap.npy")
            except:
                palette = np.array([[i, i, i] for i in range(256)], dtype=np.uint8)
        else:
            palette = np.array([
                [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
                [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
                [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
                [0, 80, 100], [0, 0, 230], [119, 11, 32],
            ] + [[i, i, i] for i in range(19, 256)], dtype=np.uint8)

        # 为每张图创建独立文件夹并保存所有可视化结果
        if save_dir and attn_data_to_save[0][0] is not None:
            B = images_gpu.shape[0]
            pred_np = preds.argmax(dim=1).cpu().numpy().astype(np.uint8)

            for b in range(B):
                # 获取文件名
                if "fn" in minibatch and len(minibatch["fn"]) > b:
                    fn = minibatch["fn"][b]
                    fn = fn.replace(".jpg", "").replace(".png", "").replace("datasets/", "")
                    fn = re.sub(r"[\\/]+", "_", fn)
                else:
                    fn = f"batch{idx:04d}_img{b}"

                # 为每张图创建独立文件夹
                img_output_dir = os.path.join(save_dir, fn)
                os.makedirs(img_output_dir, exist_ok=True)

                # 1. 保存原始图片（反规范化）
                rgb_tensor = images[b]
                rgb_np = rgb_tensor.permute(1, 2, 0).cpu().numpy()
                mean = np.array(config.norm_mean).reshape(1, 1, 3)
                std = np.array(config.norm_std).reshape(1, 1, 3)
                rgb_np = rgb_np * std + mean
                rgb_np = (rgb_np * 255).clip(0, 255).astype(np.uint8)
                H_img, W_img = rgb_np.shape[:2]

                import matplotlib.pyplot as plt
                original_path = os.path.join(img_output_dir, "00_original.png")
                plt.imsave(original_path, rgb_np)

                # 2. 保存分割结果
                pred_colored = palette[pred_np[b] if pred_np.ndim > 2 else pred_np]
                seg_path = os.path.join(img_output_dir, "01_segmentation.png")
                plt.imsave(seg_path, pred_colored)

                # 解析该图的 token 名称/类型
                meta_token_names, meta_token_types = None, None
                if token_meta_batch is not None:
                    raw_meta = token_meta_batch[b] if isinstance(token_meta_batch, (list, tuple)) else token_meta_batch
                    if isinstance(raw_meta, (bytes, bytearray)):
                        raw_meta = raw_meta.decode("utf-8", errors="ignore")
                    if isinstance(raw_meta, str) and raw_meta:
                        try:
                            meta = json.loads(raw_meta)
                            meta_token_names = meta.get("names") or None
                            meta_token_types = meta.get("types") or None
                        except Exception as exc:
                            logger.warning(f"Failed to parse token metadata for {fn}: {exc}")

                # 3. 保存每个token的attention可视化
                token_counter = 2  # 从02开始编号（00是原图，01是分割）

                for (attn_map, spatial_shape, stage_info) in attn_data_to_save:
                    if attn_map is None:
                        continue

                    # attn_map: (B, N, Hh, T) 或 (B, N, T)
                    attn_single = attn_map[b]
                    if attn_single.dim() == 3:
                        # 对每个token选响应最大的head
                        attn_single = attn_single.max(dim=1)[0]
                    elif attn_single.dim() != 2:
                        continue

                    N, T = attn_single.shape

                    # 还原空间网格
                    if spatial_shape is not None and isinstance(spatial_shape, (tuple, list)) and len(spatial_shape) == 2:
                        h_attn, w_attn = int(spatial_shape[0]), int(spatial_shape[1])
                    else:
                        h_attn = int(np.sqrt(N))
                        w_attn = h_attn
                    if h_attn * w_attn != N:
                        continue

                    attn_2d = attn_single.reshape(h_attn, w_attn, T)  # (H', W', T)

                    # 上采样到图像大小
                    attn_resized = F.interpolate(
                        attn_2d.permute(2, 0, 1).unsqueeze(0),  # (1, T, H', W')
                        size=(H_img, W_img),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0).permute(1, 2, 0)  # (H, W, T)

                    # 获取token名称
                    if meta_token_names is not None and len(meta_token_names) > 0:
                        token_names = list(meta_token_names)
                        token_types = list(meta_token_types) if meta_token_types else None
                    elif global_token_names:
                        token_names = global_token_names[:T]
                        token_types = None
                    else:
                        token_names = [f"tok{t}" for t in range(T)]
                        token_types = None

                    # 为每个token保存attention map
                    pad_markers = {"", "<pad>", "[pad]", "pad", "<none>", "none"}

                    # 如果有filter_tokens，只保存指定的tokens
                    if filter_tokens is not None and len(filter_tokens) > 0:
                        filter_set = {ft.lower().strip() for ft in filter_tokens}
                        tokens_to_save = []
                        for t in range(T):
                            nm = token_names[t].strip() if isinstance(token_names[t], str) else str(token_names[t])
                            if nm.lower() in filter_set and nm.lower() not in pad_markers:
                                tokens_to_save.append(t)
                    else:
                        # 保存所有非padding的tokens
                        tokens_to_save = []
                        for t in range(T):
                            nm = token_names[t].strip() if isinstance(token_names[t], str) else str(token_names[t])
                            if nm.lower() not in pad_markers:
                                tokens_to_save.append(t)

                    # 保存每个token
                    for t in tokens_to_save:
                        nm = token_names[t].strip() if isinstance(token_names[t], str) else str(token_names[t])

                        # 应用竞争性归一化（如果启用）
                        if competition_mode and competition_mode.lower() != 'none':
                            a = compute_competitive_map(attn_resized.cpu().numpy(), t, mode=competition_mode, tau=competition_tau)
                        else:
                            a = _normalize01(attn_resized[..., t].cpu().numpy())

                        # 保存文件，使用编号前缀
                        file_name = f"{token_counter:02d}_attn_{_slugify(nm)}.png"
                        file_path = os.path.join(img_output_dir, file_name)

                        _save_single_token_map(
                            a, rgb_np, file_path.replace("_attn.png", ""),
                            alpha=alpha, threshold=threshold, smooth_sigma=smooth_sigma,
                            colormap=colormap, gamma=gamma, enable_grid=enable_grid
                        )

                        token_counter += 1

                    # 如果是第一张图，打印示例信息
                    if idx == 0 and b == 0:
                        logger.info(f"✓ Saved visualization for image '{fn}':")
                        logger.info(f"  - 00_original.png")
                        logger.info(f"  - 01_segmentation.png")
                        logger.info(f"  - {len(tokens_to_save)} attention maps (02-{token_counter-1:02d})")
                        if token_names:
                            logger.info(f"  - Tokens: {[token_names[t] for t in tokens_to_save[:5]]}{'...' if len(tokens_to_save) > 5 else ''}")

                processed_images += 1

    # 关闭缓存开关，清理引用
    def disable_attention_save(m):
        if isinstance(m, SemanticAlignmentModule):
            m.save_attention = False
            if hasattr(m, "last_attention_map"):
                m.last_attention_map = None
            if hasattr(m, "last_spatial_shape"):
                m.last_spatial_shape = None

    model.apply(disable_attention_save)

    logger.info("=" * 100)
    logger.info(f"✓ Attention visualization completed for {processed_images} images")
    logger.info("=" * 100)

    # 汇总指标
    if engine.distributed:
        all_metrics = [None for _ in range(engine.world_size)]
        torch.distributed.all_gather_object(all_metrics, metrics)
    else:
        all_metrics = metrics

    return all_metrics


with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()
    config = getattr(import_module(args.config), "C")

    # Override text guidance settings
    if args.text_source is not None:
        config.text_source = args.text_source
    if args.text_encoder is not None:
        config.text_encoder = args.text_encoder
    if args.text_encoder_name is not None:
        config.text_encoder_name = args.text_encoder_name
    if args.text_feature_dim is not None:
        config.text_feature_dim = int(args.text_feature_dim)
    if args.label_txt_path is not None:
        config.label_txt_path = args.label_txt_path
    if args.caption_json_path is not None:
        config.caption_json_path = args.caption_json_path
    if args.text_template_set is not None:
        config.text_template_set = args.text_template_set
    if args.max_templates_per_label is not None:
        config.max_templates_per_label = int(args.max_templates_per_label)
    if args.max_caption_sentences is not None:
        config.max_caption_sentences = int(args.max_caption_sentences)
    if args.caption_topk is not None:
        config.caption_topk = int(args.caption_topk)
    if args.caption_topk_mode is not None:
        config.caption_topk_mode = args.caption_topk_mode
    if args.image_labels_json_path is not None:
        config.image_labels_json_path = args.image_labels_json_path

    enc_stages = _parse_stages(args.sam_enc_stages)
    dec_stages = _parse_stages(args.sam_dec_stages)
    if enc_stages is not None:
        config.sam_enc_stages = enc_stages
    if dec_stages is not None:
        config.sam_dec_stages = dec_stages
    if args.superpower is not None:
        config.superpower = bool(args.superpower)

    # ★ 为可视化强制开启 text guidance（不影响评测）
    if args.save_attention:
        logger.info("Forcing enable_text_guidance=True for attention visualization")
        config.enable_text_guidance = True

        if not hasattr(config, 'text_source') or config.text_source is None:
            config.text_source = 'imglabels'  # 默认使用全局类别
        # 确保所有必要的文本配置都有默认值
        if not hasattr(config, 'text_encoder') or config.text_encoder is None:
            config.text_encoder = 'jinaclip'
        if not hasattr(config, 'text_encoder_name') or config.text_encoder_name is None:
            config.text_encoder_name = None
        if not hasattr(config, 'text_feature_dim') or config.text_feature_dim is None:
            config.text_feature_dim = 512
        if not hasattr(config, 'text_template_set') or config.text_template_set is None:
            config.text_template_set = 'clip'
        if not hasattr(config, 'max_templates_per_label') or config.max_templates_per_label is None:
            config.max_templates_per_label = 3

        # 打印实际使用的配置
        logger.info("=" * 80)
        logger.info("TEXT GUIDANCE CONFIGURATION:")
        logger.info(f"  text_source: {config.text_source}")
        logger.info(f"  text_encoder: {config.text_encoder}")
        logger.info(f"  text_feature_dim: {config.text_feature_dim}")
        logger.info(f"  label_txt_path: {getattr(config, 'label_txt_path', 'NOT SET')}")
        logger.info(f"  image_labels_json_path: {getattr(config, 'image_labels_json_path', 'NOT SET')}")

        # 检查文件是否存在
        if config.text_source == 'imglabels':
            img_json = getattr(config, 'image_labels_json_path', None)
            if img_json and os.path.exists(img_json):
                logger.info(f"  ✓ image_labels_json found: {img_json}")
            else:
                logger.error(f"  ✗ image_labels_json NOT FOUND or NOT SET!")
                logger.error(f"  Path checked: {img_json}")
                logger.error(f"  Current working dir: {os.getcwd()}")
                logger.error(f"  This will cause per-image labels to be unavailable!")

        logger.info("=" * 80)

    config.pad = False
    if "x_modal" not in config:
        config["x_modal"] = "d"
    cudnn.benchmark = True

    val_loader, val_sampler = get_val_loader(engine, RGBXDataset, config, int(args.gpus))

    # Apply image selection if specified
    original_dataset = val_loader.dataset
    selected_indices = None

    if args.image_indices is not None:
        # Parse comma-separated indices
        try:
            selected_indices = [int(i.strip()) for i in args.image_indices.split(',')]
            logger.info(f"Using specified image indices: {selected_indices}")
        except ValueError as e:
            logger.error(f"Invalid image indices format: {e}")
            sys.exit(1)

    elif args.image_paths is not None:
        # Load image paths from file
        if not os.path.exists(args.image_paths):
            logger.error(f"Image paths file not found: {args.image_paths}")
            sys.exit(1)

        with open(args.image_paths, 'r') as f:
            target_paths = [line.strip() for line in f if line.strip()]

        # Match paths to dataset indices
        selected_indices = []
        dataset_files = original_dataset._file_names
        for target_path in target_paths:
            target_base = os.path.basename(target_path)
            for idx, file_name in enumerate(dataset_files):
                if target_base in file_name or file_name in target_path:
                    selected_indices.append(idx)
                    break

        logger.info(f"Matched {len(selected_indices)}/{len(target_paths)} images from file")

    elif args.num_images is not None and args.random_select:
        # Random selection
        total_images = len(original_dataset)
        num_select = min(args.num_images, total_images)
        selected_indices = np.random.choice(total_images, size=num_select, replace=False).tolist()
        logger.info(f"Randomly selected {num_select} images from {total_images}")

    elif args.num_images is not None:
        # Sequential selection (first N images)
        total_images = len(original_dataset)
        num_select = min(args.num_images, total_images)
        selected_indices = list(range(num_select))
        logger.info(f"Using first {num_select} images")

    # Apply subset if indices were selected
    if selected_indices is not None and len(selected_indices) > 0:
        subset_dataset = SubsetDataset(original_dataset, selected_indices)
        val_loader = DataLoader(
            subset_dataset,
            batch_size=val_loader.batch_size,
            num_workers=config.num_workers,
            drop_last=False,
            shuffle=False,
            pin_memory=True,
        )
        logger.info(f"Dataset filtered to {len(subset_dataset)} images")

    print(f"Validation dataset size: {len(val_loader.dataset)}")

    if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
        tb_dir = config.tb_dir + "/{}".format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
        generate_tb_dir = config.tb_dir + "/tb"
        tb = SummaryWriter(log_dir=tb_dir)
        engine.link_tb(tb_dir, generate_tb_dir)

    BatchNorm2d = nn.SyncBatchNorm if engine.distributed else nn.BatchNorm2d
    model = segmodel(cfg=config, norm_layer=BatchNorm2d)

    if args.continue_fpath:
        weight = torch.load(args.continue_fpath, map_location="cpu")["model"]
        print("Loading model weights...")
        model.load_state_dict(weight, strict=False)

    if engine.distributed:
        logger.info("Using distributed training mode...")
        if torch.cuda.is_available():
            model.cuda()
            model = DistributedDataParallel(
                model,
                device_ids=[engine.local_rank],
                output_device=engine.local_rank,
                find_unused_parameters=True,
            )
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    engine.register_state(dataloader=val_loader, model=model)

    logger.info("Begin testing...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 如果设置了 --save-predictions 但没有 --save_path，使用默认路径
    if args.save_predictions and not args.save_path:
        args.save_path = "./infer_predictions"
        logger.info(f"--save-predictions enabled but no --save_path specified, using default: {args.save_path}")

    if args.save_attention:
        # 注意力可视化模式
        # 解析 filter_tokens 参数
        filter_tokens_list = None
        if args.filter_tokens:
            filter_tokens_list = [t.strip() for t in args.filter_tokens.split(',') if t.strip()]
            logger.info(f"Filtering tokens: {filter_tokens_list}")

        # 解析 stage indices
        stage_indices = None
        if args.vis_stage_idx.lower() == 'all':
            stage_indices = None  # Extract all stages
            logger.info("Extracting attention from ALL stages")
        else:
            try:
                stage_indices = [int(i.strip()) for i in args.vis_stage_idx.split(',')]
                logger.info(f"Extracting attention from stages: {stage_indices}")
            except ValueError as e:
                logger.error(f"Invalid stage indices format: {e}")
                sys.exit(1)

        with torch.no_grad():
            model.eval()
            # Note: num_images is now handled by dataset filtering above
            all_metrics = evaluate_with_attention(
                model,
                val_loader,
                config,
                device,
                engine,
                save_dir=args.save_path,
                vis_stage=args.vis_stage,
                stage_indices=stage_indices,
                block_idx=args.vis_block_idx,
                num_images=None,  # Dataset already filtered
                alpha=args.attention_alpha,
                threshold=args.attention_threshold,
                smooth_sigma=args.attention_smooth,
                max_token_vis=args.max_token_vis,
                filter_tokens=filter_tokens_list,
                competition_mode=args.vis_competition,
                competition_tau=args.vis_competition_tau,
                colormap=args.vis_colormap,
                gamma=args.vis_gamma,
                enable_grid=args.vis_grid,
                aggregate_mode=args.vis_aggregate,
            )

            if engine.distributed:
                if engine.local_rank == 0:
                    metric = all_metrics[0]
                    for other_metric in all_metrics[1:]:
                        metric.update_hist(other_metric.hist)
                    ious, miou = metric.compute_iou()
                    acc, macc = metric.compute_pixel_acc()
                    f1, mf1 = metric.compute_f1()
                    logger.info(f"mIoU: {miou:.4f}")
            else:
                metric = all_metrics
                ious, miou = metric.compute_iou()
                acc, macc = metric.compute_pixel_acc()
                f1, mf1 = metric.compute_f1()
                logger.info(f"mIoU: {miou:.4f}")

    else:
        # 标准多尺度评估 (使用和eval.py相同的配置以获得一致的mIoU)
        logger.info("Running standard multi-scale+flip evaluation (same as eval.py)...")
        scales = [0.5, 0.75, 1.0, 1.25, 1.5]
        flip = True

        logger.info(f"Evaluation scales: {scales}")
        logger.info(f"Flip augmentation: {flip}")

        # 确定是否保存预测结果
        save_dir_for_eval = args.save_path if args.save_predictions else None

        if save_dir_for_eval:
            logger.info(f"Predictions will be saved to: {save_dir_for_eval}")
            logger.info("Output format: <save_path>/<image_name>_pred.png")

        if engine.distributed:
            print("Multi GPU test")
            with torch.no_grad():
                model.eval()
                all_metrics = evaluate_msf(
                    model,
                    val_loader,
                    config,
                    device,
                    scales,
                    flip,
                    engine,
                    save_dir=save_dir_for_eval,
                )
                if engine.local_rank == 0:
                    metric = all_metrics[0]
                    for other_metric in all_metrics[1:]:
                        metric.update_hist(other_metric.hist)
                    ious, miou = metric.compute_iou()
                    acc, macc = metric.compute_pixel_acc()
                    f1, mf1 = metric.compute_f1()
                    logger.info("=" * 80)
                    logger.info("FINAL RESULTS (Multi-Scale + Flip):")
                    logger.info(f"mIoU: {miou:.4f}")
                    logger.info(f"mAcc: {macc:.4f}")
                    logger.info(f"mF1: {mf1:.4f}")
                    logger.info(f"Per-class IoUs: {[f'{iou:.4f}' for iou in ious]}")
                    logger.info("=" * 80)
                    print(f"mIoU: {miou:.4f}")
        else:
            with torch.no_grad():
                model.eval()
                metric = evaluate_msf(
                    model,
                    val_loader,
                    config,
                    device,
                    scales,
                    flip,
                    engine,
                    save_dir=save_dir_for_eval,
                )
                ious, miou = metric.compute_iou()
                acc, macc = metric.compute_pixel_acc()
                f1, mf1 = metric.compute_f1()
                logger.info("=" * 80)
                logger.info("FINAL RESULTS (Multi-Scale + Flip):")
                logger.info(f"mIoU: {miou:.4f}")
                logger.info(f"mAcc: {macc:.4f}")
                logger.info(f"mF1: {mf1:.4f}")
                logger.info(f"Per-class IoUs: {[f'{iou:.4f}' for iou in ious]}")
                logger.info("=" * 80)
                print(f"mIoU: {miou:.4f}")