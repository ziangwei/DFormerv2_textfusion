import argparse
import importlib
import os
import random
import sys
import time
from importlib import import_module
import pathlib

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
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import gaussian_filter

from models.builder import EncoderDecoder as segmodel
from models.blocks.semantic_alignment import SemanticAlignmentModule
from utils.dataloader.dataloader import ValPre, get_train_loader, get_val_loader
from utils.dataloader.RGBXDataset import RGBXDataset
from utils.engine.engine import Engine
from utils.engine.logger import get_logger
from utils.init_func import group_weight, init_weight
from utils.lr_policy import WarmUpPolyLR
from utils.metric import compute_score, hist_info
from utils.pyt_utils import all_reduce_tensor, ensure_dir, link_file, load_model, parse_devices
from utils.val_mm import evaluate, evaluate_msf
from utils.visualize import print_iou, show_img

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
parser.add_argument("--attention-threshold", type=float, default=0.3,
                    help="Threshold for attention visualization (0-1, 0=no threshold)")
parser.add_argument("--attention-smooth", type=float, default=2.0,
                    help="Gaussian smoothing sigma for attention maps (0=no smoothing)")

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


def visualize_attention_maps_enhanced(attn_map, rgb_img, save_path, token_names=None,
                                      alpha=0.5, threshold=0.3, smooth_sigma=2.0):
    """
    改进的attention可视化，包含平滑、阈值化、自适应alpha等特性

    Args:
        attn_map: (H, W, T) - 已上采样到rgb尺寸的attention
        rgb_img: (H, W, 3) - 原始RGB图像 [0,255] uint8
        save_path: 保存路径前缀
        token_names: list[str] - 每个token的名称
        alpha: float - 基础叠加透明度
        threshold: float - attention阈值 (0-1)
        smooth_sigma: float - 高斯平滑sigma
    """
    H, W, T = attn_map.shape
    colormap = cm.get_cmap('jet')

    for t in range(T):
        attn_t = attn_map[:, :, t].cpu().numpy()  # (H, W)

        # 归一化到[0,1]
        attn_min = attn_t.min()
        attn_max = attn_t.max()
        if attn_max - attn_min > 1e-8:
            attn_t_norm = (attn_t - attn_min) / (attn_max - attn_min)
        else:
            attn_t_norm = np.zeros_like(attn_t)

        # ★ 改进1：平滑处理
        if smooth_sigma > 0:
            attn_t_smooth = gaussian_filter(attn_t_norm, sigma=smooth_sigma)
        else:
            attn_t_smooth = attn_t_norm

        # ★ 改进2：阈值化
        if threshold > 0:
            attn_t_thresh = attn_t_smooth.copy()
            attn_t_thresh[attn_t_smooth < threshold] = 0
            # 重新归一化阈值化后的attention
            if attn_t_thresh.max() > 0:
                attn_t_thresh = attn_t_thresh / attn_t_thresh.max()
        else:
            attn_t_thresh = attn_t_smooth

        # 应用colormap
        attn_colored_raw = (colormap(attn_t_norm)[:, :, :3] * 255).astype(np.uint8)
        attn_colored_smooth = (colormap(attn_t_smooth)[:, :, :3] * 255).astype(np.uint8)
        attn_colored_thresh = (colormap(attn_t_thresh)[:, :, :3] * 255).astype(np.uint8)

        # ★ 改进3：自适应alpha（高attention区域更不透明）
        alpha_map = np.clip(attn_t_thresh * 1.5, 0, 1)[:, :, np.newaxis]
        overlay_adaptive = (alpha_map * attn_colored_thresh +
                            (1 - alpha_map) * rgb_img).astype(np.uint8)

        # 固定alpha叠加（用于对比）
        overlay_fixed = (alpha * attn_colored_smooth +
                         (1 - alpha) * rgb_img).astype(np.uint8)

        # ★ 改进4：生成详细的可视化（5列布局）
        token_name = token_names[t] if (token_names and t < len(token_names)) else f"token{t}"
        filename = f"{save_path}_token{t:02d}_{token_name}.png"

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 第一行
        # 原图
        axes[0, 0].imshow(rgb_img)
        axes[0, 0].set_title(f"Original Image", fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')

        # 原始attention热力图
        im1 = axes[0, 1].imshow(attn_t_norm, cmap='jet', vmin=0, vmax=1)
        axes[0, 1].set_title(f"Raw Attention\nmax={attn_t_norm.max():.3f}", fontsize=12)
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

        # 平滑后的attention
        im2 = axes[0, 2].imshow(attn_t_smooth, cmap='jet', vmin=0, vmax=1)
        axes[0, 2].set_title(f"Smoothed (σ={smooth_sigma})", fontsize=12)
        axes[0, 2].axis('off')
        plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

        # 第二行
        # 阈值化后的attention
        im3 = axes[1, 0].imshow(attn_t_thresh, cmap='jet', vmin=0, vmax=1)
        axes[1, 0].set_title(f"Thresholded (>{threshold})", fontsize=12)
        axes[1, 0].axis('off')
        plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

        # 固定alpha叠加
        axes[1, 1].imshow(overlay_fixed)
        axes[1, 1].set_title(f"Fixed Alpha={alpha}", fontsize=12)
        axes[1, 1].axis('off')

        # 自适应alpha叠加
        axes[1, 2].imshow(overlay_adaptive)
        axes[1, 2].set_title(f"Adaptive Alpha\n{token_name}", fontsize=12, fontweight='bold')
        axes[1, 2].axis('off')

        plt.suptitle(f"Attention Visualization: {token_name}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ Saved: {filename}")


def extract_attention_from_model_enhanced(model, vis_stage="enc", stage_idx=0, block_idx=-1):
    """
    增强版attention提取，支持精确指定stage和block

    Args:
        model: 模型实例
        vis_stage: "enc" 或 "dec"
        stage_idx: stage索引 (encoder: 0-3, decoder: 0-2)
        block_idx: block索引 (-1表示该stage的最后一个block)

    Returns:
        list of (attention_map, spatial_shape, stage_info):
            attention_map: (B, N, H, T)
            spatial_shape: (h, w)
            stage_info: str描述信息
    """
    if hasattr(model, 'module'):
        actual_model = model.module
    else:
        actual_model = model

    attention_data = []

    if vis_stage == "enc":
        # Encoder attention
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

            # 选择block
            if block_idx == -1:
                block_idx = len(blocks) - 1
            if block_idx >= len(blocks):
                logger.warning(f"Block {block_idx} not found in stage {stage_idx}! Using last block.")
                block_idx = len(blocks) - 1

            sam_block = blocks[block_idx]
            if hasattr(sam_block, 'last_attention_map'):
                attn = sam_block.last_attention_map
                spatial_shape = getattr(sam_block, 'last_spatial_shape', None)
                if attn is not None:
                    stage_info = f"enc_stage{stage_idx}_block{block_idx}"
                    attention_data.append((attn, spatial_shape, stage_info))
                    logger.info(f"✓ Extracted {stage_info}, shape={attn.shape}, spatial={spatial_shape}")

    elif vis_stage == "dec":
        # Decoder attention
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
                    logger.info(f"✓ Extracted {stage_info}, shape={attn.shape}, spatial={spatial_shape}")

    if not attention_data:
        return [(None, None, "No attention found")]

    return attention_data


def load_class_names(config):
    """加载类别名称"""
    token_names = None

    # 尝试从label_txt_path加载
    if hasattr(config, 'label_txt_path') and config.label_txt_path:
        try:
            with open(config.label_txt_path, 'r') as f:
                token_names = [line.strip() for line in f.readlines()]
            logger.info(f"✓ Loaded {len(token_names)} class names from {config.label_txt_path}")
        except Exception as e:
            logger.warning(f"Failed to load class names: {e}")

    return token_names


@torch.no_grad()
def evaluate_with_attention(model, dataloader, config, device, engine,
                            save_dir=None, vis_stage="enc", stage_idx=0, block_idx=-1,
                            num_images=None, alpha=0.5, threshold=0.3, smooth_sigma=2.0):
    """
    增强版attention可视化评估函数
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

    # 检查text guidance
    if not getattr(config, 'enable_text_guidance', False):
        logger.error("=" * 80)
        logger.error("ERROR: enable_text_guidance is False!")
        logger.error("Attention visualization requires text guidance to be enabled.")
        logger.error("=" * 80)
        raise ValueError("Cannot visualize attention without text guidance enabled")

    model.eval()
    n_classes = config.num_classes
    metrics = Metrics(n_classes, config.background, device)

    # 启用所有SAM的attention保存
    def enable_attention_save(m):
        if isinstance(m, SemanticAlignmentModule):
            m.save_attention = True

    model.apply(enable_attention_save)

    # 加载类别名称
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

        # Forward pass
        preds = model(images_gpu, modal_xs_gpu, text_features=text_feats)

        # 提取attention maps
        attn_data = extract_attention_from_model_enhanced(
            model, vis_stage, stage_idx, block_idx
        )

        # 更新metrics
        metrics.update(preds.softmax(dim=1), labels_gpu)

        # 可视化attention
        if save_dir and attn_data[0][0] is not None:
            B = images_gpu.shape[0]

            for b in range(B):
                if num_images and processed_images >= num_images:
                    break

                # 准备RGB图像
                rgb_tensor = images[b]
                rgb_np = rgb_tensor.permute(1, 2, 0).cpu().numpy()
                mean = np.array(config.norm_mean).reshape(1, 1, 3)
                std = np.array(config.norm_std).reshape(1, 1, 3)
                rgb_np = rgb_np * std + mean
                rgb_np = (rgb_np * 255).clip(0, 255).astype(np.uint8)
                H_img, W_img = rgb_np.shape[:2]

                # 获取文件名
                if "fn" in minibatch and len(minibatch["fn"]) > b:
                    fn = minibatch["fn"][b]
                    fn = fn.replace(".jpg", "").replace(".png", "").replace("datasets/", "")
                else:
                    fn = f"batch{idx:04d}_img{b}"

                # 为每个attention map生成可视化
                for map_idx, (attn_map, spatial_shape, stage_info) in enumerate(attn_data):
                    if attn_map is None:
                        logger.warning(f"Attention map is None for {fn}, reason: {stage_info}")
                        continue

                    # attn_map: (B, N, H_heads, T)
                    if attn_map.dim() == 4:
                        attn_single = attn_map[b].mean(dim=1)  # (N, T) - average over heads
                        N, T = attn_single.shape
                    elif attn_map.dim() == 3:
                        attn_single = attn_map[b]  # (N, T)
                        N, T = attn_single.shape
                    else:
                        logger.warning(f"Unsupported attention shape: {attn_map.shape}")
                        continue

                    # 使用保存的空间形状或尝试推断
                    if spatial_shape is not None:
                        h_attn, w_attn = spatial_shape
                        if h_attn * w_attn != N:
                            logger.warning(f"Spatial shape mismatch: {h_attn}x{w_attn}!={N}, trying sqrt...")
                            h_attn = int(np.sqrt(N))
                            w_attn = h_attn
                    else:
                        h_attn = int(np.sqrt(N))
                        w_attn = h_attn

                    if h_attn * w_attn != N:
                        logger.warning(f"Cannot reshape N={N}, skipping {fn}...")
                        continue

                    attn_2d = attn_single.reshape(h_attn, w_attn, T)  # (H', W', T)

                    # 上采样到原图尺寸
                    attn_resized = F.interpolate(
                        attn_2d.permute(2, 0, 1).unsqueeze(0),  # (1, T, H', W')
                        size=(H_img, W_img),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0).permute(1, 2, 0)  # (H, W, T)

                    # 保存路径
                    save_prefix = os.path.join(save_dir, "attention", stage_info, fn)
                    os.makedirs(os.path.dirname(save_prefix), exist_ok=True)

                    # 生成token names
                    if global_token_names:
                        token_names = global_token_names[:T]
                    else:
                        token_names = [f"tok{t}" for t in range(T)]

                    # 增强可视化
                    visualize_attention_maps_enhanced(
                        attn_resized,
                        rgb_np,
                        save_prefix,
                        token_names=token_names,
                        alpha=alpha,
                        threshold=threshold,
                        smooth_sigma=smooth_sigma
                    )

                    logger.info(f"✓ Processed {fn} ({stage_info}): {T} tokens, resolution {h_attn}x{w_attn}")

                processed_images += 1

    # 关闭attention保存
    def disable_attention_save(m):
        if isinstance(m, SemanticAlignmentModule):
            m.save_attention = False
            m.last_attention_map = None
            m.last_spatial_shape = None

    model.apply(disable_attention_save)

    logger.info("=" * 100)
    logger.info(f"✓ Attention visualization completed for {processed_images} images")
    logger.info("=" * 100)

    # 返回metrics
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

    # ★ 强制启用text guidance（用于可视化）
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

    if engine.distributed:
        BatchNorm2d = nn.SyncBatchNorm
    else:
        BatchNorm2d = nn.BatchNorm2d

    model = segmodel(cfg=config, norm_layer=BatchNorm2d)
    weight = torch.load(args.continue_fpath)["model"]

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

    data_setting = {
        "rgb_root": config.rgb_root_folder,
        "rgb_format": config.rgb_format,
        "gt_root": config.gt_root_folder,
        "gt_format": config.gt_format,
        "transform_gt": config.gt_transform,
        "x_root": config.x_root_folder,
        "x_format": config.x_format,
        "x_single_channel": config.x_is_single_channel,
        "dataset_name": getattr(config, "dataset_name", None),
        "backbone": getattr(config, "backbone", None),
        "enable_text_guidance": getattr(config, "enable_text_guidance", False),
        "label_txt_path": getattr(config, "label_txt_path", None),
        "caption_json_path": getattr(config, "caption_json_path", None),
        "image_labels_json_path": getattr(config, "image_labels_json_path", None),
        "text_template_set": getattr(config, "text_template_set", "clip"),
        "max_templates_per_label": getattr(config, "max_templates_per_label", 3),
        "text_source": getattr(config, "text_source", "both"),
        "text_encoder": getattr(config, "text_encoder", "jinaclip"),
        "text_encoder_name": getattr(config, "text_encoder_name", None),
        "text_feature_dim": getattr(config, "text_feature_dim", 512),
        "max_caption_sentences": getattr(config, "max_caption_sentences", 8),
        "caption_topk": getattr(config, "caption_topk", 0),
        "caption_topk_mode": getattr(config, "caption_topk_mode", "class_sim"),
        "max_image_labels": getattr(config, "max_image_labels", 0),
    }

    all_dev = [0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.save_attention:
        # ★ 增强的attention可视化模式
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
                smooth_sigma=args.attention_smooth
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
        # 标准评估模式
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