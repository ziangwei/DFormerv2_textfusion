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
from utils.modern_visualize import (
    visualize_attention_modern,
    create_segmentation_grid,
    get_segmentation_palette
)

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
parser.add_argument("--vis-stage", type=str, default="enc",
                    choices=["enc", "dec"],
                    help="Visualize encoder or decoder attention")
parser.add_argument("--vis-stage-idx", type=int, default=0,
                    help="Which stage index to visualize (0,1,2,3 for encoder; 0,1,2 for decoder)")
parser.add_argument("--vis-block-idx", type=int, default=-1,
                    help="Which block in the stage (-1 for last block)")
parser.add_argument("--num-images", type=int, default=None,
                    help="Number of images to process for visualization (None=all)")
parser.add_argument("--attention-alpha", type=float, default=0.5,
                    help="Alpha blending factor for overlay visualization (0-1)")
parser.add_argument("--attention-threshold", type=float, default=0.0,
                    help="Zero-out low responses in [0,1] after normalization (0=no threshold)")
parser.add_argument("--attention-smooth", type=float, default=0.0,
                    help="Gaussian smoothing sigma for attention maps (0=no smoothing)")
parser.add_argument("--max-token-vis", type=int, default=64,
                    help="Max tokens to save per image to avoid explosion")
parser.add_argument("--filter-tokens", type=str, default=None,
                    help="Comma-separated list of token names to visualize (e.g., 'floor,wall,ceiling'). "
                         "If set, only these tokens will be visualized.")
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


def _save_single_token_map(attn: np.ndarray, rgb: np.ndarray, out_prefix: str,
                           alpha: float = 0.5, threshold: float = 0.0, smooth_sigma: float = 0.0):
    """
    Modern attention visualization with Viridis colormap (CVPR-style).

    attn: (H, W) in [0,1]
    rgb : (H, W, 3) uint8 (RGB)
    """
    H, W = attn.shape

    # 可选平滑
    if smooth_sigma and smooth_sigma > 0.0:
        attn = gaussian_filter(attn, sigma=float(smooth_sigma))
        attn = _normalize01(attn)

    # 阈值
    if threshold and threshold > 0.0:
        attn = np.where(attn >= threshold, attn, 0.0)

    # 使用现代化 Viridis 配色（而非老旧的 JET）
    overlay = visualize_attention_modern(
        attn, rgb,
        alpha=alpha,
        cmap='viridis',  # CVPR 流行配色
        smooth=(smooth_sigma > 0),
        normalize=True
    )

    # 保存现代化可视化
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_prefix + "_modern.png", overlay_bgr)


def visualize_attention_maps_enhanced(attn_hwT: torch.Tensor,
                                      rgb_np: np.ndarray,
                                      save_prefix: str,
                                      token_names=None,
                                      token_types=None,
                                      alpha: float = 0.5,
                                      threshold: float = 0.0,
                                      smooth_sigma: float = 0.0,
                                      max_tokens: int = 64,
                                      filter_tokens=None):
    """
    attn_hwT: (H, W, T) torch or np
    rgb_np  : (H, W, 3) uint8 RGB
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

        a = _normalize01(attn_np[..., t])
        out_prefix = f"{save_prefix}__{tag}"
        _save_single_token_map(
            a, rgb_np, out_prefix,
            alpha=alpha, threshold=threshold, smooth_sigma=smooth_sigma
        )


def extract_attention_from_model_enhanced(model, vis_stage="enc", stage_idx=0, block_idx=-1):
    """
    返回列表 [(attn(B,N,Hh,T), spatial_shape(h,w), stage_info), ...]
    """
    if hasattr(model, 'module'):
        actual_model = model.module
    else:
        actual_model = model

    attention_data = []

    if vis_stage == "enc":
        backbone = actual_model.backbone
        if hasattr(backbone, 'encoder_sam_blocks'):
            if stage_idx >= len(backbone.encoder_sam_blocks):
                logger.warning(
                    f"Encoder stage {stage_idx} not found! Available: 0-{len(backbone.encoder_sam_blocks) - 1}")
                return [(None, None, "Invalid stage")]

            blocks = backbone.encoder_sam_blocks[stage_idx]
            if len(blocks) == 0:
                logger.warning(f"Encoder stage {stage_idx} has no SAM blocks!")
                return [(None, None, "No SAM blocks")]

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
            if stage_idx >= len(decode_head.dec_sam_layers):
                logger.warning(
                    f"Decoder stage {stage_idx} not found! Available: 0-{len(decode_head.dec_sam_layers) - 1}")
                return [(None, None, "Invalid stage")]

            sam_layer = decode_head.dec_sam_layers[stage_idx]
            if sam_layer is None:
                logger.warning(f"Decoder stage {stage_idx} has no SAM layer (disabled)!")
                return [(None, None, "SAM disabled")]

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
                            save_dir=None, vis_stage="enc", stage_idx=0, block_idx=-1,
                            num_images=None, alpha=0.5, threshold=0.0, smooth_sigma=0.0, max_token_vis=64,
                            filter_tokens=None):
    """
    注意力可视化 + 指标计算
    """
    from utils.metrics_new import Metrics

    logger.info("=" * 100)
    logger.info(f"Starting ENHANCED attention visualization")
    logger.info(f"  Stage: {vis_stage.upper()}")
    logger.info(f"  Stage Index: {stage_idx}")
    logger.info(f"  Block Index: {block_idx} (-1=last)")
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

    for idx, minibatch in enumerate(dataloader):
        if num_images and processed_images >= num_images:
            logger.info(f"✓ Reached target number of images ({num_images}), stopping...")
            break

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
        attn_data = extract_attention_from_model_enhanced(model, vis_stage, stage_idx, block_idx)

        # 更新指标
        metrics.update(preds.softmax(dim=1), labels_gpu)

        # 可视化
        if save_dir and attn_data[0][0] is not None:
            B = images_gpu.shape[0]
            for b in range(B):
                if num_images and processed_images >= num_images:
                    break

                # 反规范化得到 RGB 原图（H,W,3）uint8
                rgb_tensor = images[b]
                rgb_np = rgb_tensor.permute(1, 2, 0).cpu().numpy()
                mean = np.array(config.norm_mean).reshape(1, 1, 3)
                std = np.array(config.norm_std).reshape(1, 1, 3)
                rgb_np = rgb_np * std + mean
                rgb_np = (rgb_np * 255).clip(0, 255).astype(np.uint8)
                H_img, W_img = rgb_np.shape[:2]

                # 文件名
                if "fn" in minibatch and len(minibatch["fn"]) > b:
                    fn = minibatch["fn"][b]
                    fn = fn.replace(".jpg", "").replace(".png", "").replace("datasets/", "")
                    fn = re.sub(r"[\\/]+", "_", fn)
                else:
                    fn = f"batch{idx:04d}_img{b}"

                # 解析该图的 token 名称/类型（来自 dataset 的 text_token_meta）
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

                for (attn_map, spatial_shape, stage_info) in attn_data:
                    if attn_map is None:
                        logger.warning(f"Attention map is None for {fn}, reason: {stage_info}")
                        continue

                    # attn_map: (B, N, Hh, T) 或 (B, N, T)
                    attn_single = attn_map[b]
                    if attn_single.dim() == 3:
                        # 平均掉 head 维： (N, Hh, T) -> (N, T)
                        attn_single = attn_single.mean(dim=1)
                    elif attn_single.dim() != 2:
                        logger.warning(f"Unsupported attention shape: {tuple(attn_single.shape)}")
                        continue

                    N, T = attn_single.shape

                    # 还原空间网格
                    if spatial_shape is not None and isinstance(spatial_shape, (tuple, list)) and len(spatial_shape) == 2:
                        h_attn, w_attn = int(spatial_shape[0]), int(spatial_shape[1])
                    else:
                        h_attn = int(np.sqrt(N))
                        w_attn = h_attn
                    if h_attn * w_attn != N:
                        logger.warning(f"Spatial shape mismatch: {h_attn}x{w_attn}!={N}, skip {fn}.")
                        continue

                    attn_2d = attn_single.reshape(h_attn, w_attn, T)  # (H', W', T)

                    # 上采样到图像大小
                    attn_resized = F.interpolate(
                        attn_2d.permute(2, 0, 1).unsqueeze(0),  # (1, T, H', W')
                        size=(H_img, W_img),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0).permute(1, 2, 0)  # (H, W, T)

                    # 选择名称来源：先用 per-image meta；没有再用全局 label list；最后回退 tok{t}
                    if meta_token_names is not None and len(meta_token_names) > 0:
                        token_names = list(meta_token_names)
                        token_types = list(meta_token_types) if meta_token_types else None
                    elif global_token_names:
                        token_names = global_token_names[:T]
                        token_types = None
                    else:
                        token_names = [f"tok{t}" for t in range(T)]
                        token_types = None

                    save_prefix = os.path.join(save_dir, "attention", stage_info, fn)
                    os.makedirs(os.path.dirname(save_prefix), exist_ok=True)

                    visualize_attention_maps_enhanced(
                        attn_resized,
                        rgb_np,
                        save_prefix,
                        token_names=token_names,
                        token_types=token_types,
                        alpha=alpha,
                        threshold=threshold,
                        smooth_sigma=smooth_sigma,
                        max_tokens=max_token_vis,
                        filter_tokens=filter_tokens
                    )

                    logger.info(f"✓ {fn} ({stage_info}): saved token attentions (T={T}, grid={h_attn}x{w_attn})")

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

    config.pad = False
    if "x_modal" not in config:
        config["x_modal"] = "d"
    cudnn.benchmark = True

    val_loader, val_sampler = get_val_loader(engine, RGBXDataset, config, int(args.gpus))
    print(f"Validation dataset size: {len(val_loader)}")

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

    if args.save_attention:
        # 注意力可视化模式
        # 解析 filter_tokens 参数
        filter_tokens_list = None
        if args.filter_tokens:
            filter_tokens_list = [t.strip() for t in args.filter_tokens.split(',') if t.strip()]
            logger.info(f"Filtering tokens: {filter_tokens_list}")
        with torch.no_grad():
            model.eval()
            all_metrics = evaluate_with_attention(
                model,
                val_loader,
                config,
                device,
                engine,
                save_dir=args.save_path,
                vis_stage=args.vis_stage,
                stage_idx=args.vis_stage_idx,
                block_idx=args.vis_block_idx,
                num_images=args.num_images,
                alpha=args.attention_alpha,
                threshold=args.attention_threshold,
                smooth_sigma=args.attention_smooth,
                max_token_vis=args.max_token_vis,
                filter_tokens=filter_tokens_list,
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
        # 标准多尺度评估
        logger.info("Running standard multi-scale evaluation...")
        if engine.distributed:
            print("Multi GPU test")
            with torch.no_grad():
                model.eval()
                all_metrics = evaluate_msf(
                    model,
                    val_loader,
                    config,
                    device,
                    [0.5, 0.75, 1.0, 1.25, 1.5],
                    True,
                    engine,
                    save_dir=args.save_path,
                )
                if engine.local_rank == 0:
                    metric = all_metrics[0]
                    for other_metric in all_metrics[1:]:
                        metric.update_hist(other_metric.hist)
                    ious, miou = metric.compute_iou()
                    acc, macc = metric.compute_pixel_acc()
                    f1, mf1 = metric.compute_f1()
                    print(f"mIoU: {miou:.4f}")
        else:
            with torch.no_grad():
                model.eval()
                metric = evaluate_msf(
                    model,
                    val_loader,
                    config,
                    device,
                    [0.5, 0.75, 1.0, 1.25, 1.5],
                    True,
                    engine,
                    save_dir=args.save_path,
                )
                ious, miou = metric.compute_iou()
                acc, macc = metric.compute_pixel_acc()
                f1, mf1 = metric.compute_f1()
                print(f"mIoU: {miou:.4f}")
