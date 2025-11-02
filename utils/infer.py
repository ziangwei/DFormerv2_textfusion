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

# --- Attention visualization switches ---
parser.add_argument("--save-attention", action="store_true", help="Save attention maps for visualization")
parser.add_argument("--vis-stage", type=str, default="dec_last",
                    choices=["dec_last", "enc_last", "dec_all", "enc_all"],
                    help="Which stage SAM to visualize: dec_last, enc_last, dec_all, enc_all")
parser.add_argument("--num-images", type=int, default=None,
                    help="Number of images to process for visualization (None=all)")
parser.add_argument("--attention-alpha", type=float, default=0.5,
                    help="Alpha blending factor for overlay visualization (0-1)")

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


def visualize_attention_maps(attn_map, rgb_img, save_path, token_names=None, alpha=0.5):
    """
    可视化attention maps，生成每个token的注意力热力图

    Args:
        attn_map: (H, W, T) - 已上采样到rgb尺寸的attention
        rgb_img: (H, W, 3) - 原始RGB图像 [0,255] uint8
        save_path: 保存路径前缀
        token_names: list[str] - 每个token的名称（可选）
        alpha: float - 叠加透明度
    """
    H, W, T = attn_map.shape

    # 创建colormap
    colormap = cm.get_cmap('jet')

    # 为每个token生成可视化
    for t in range(T):
        attn_t = attn_map[:, :, t]  # (H, W)

        # 归一化到[0,1]
        attn_min = attn_t.min()
        attn_max = attn_t.max()
        if attn_max - attn_min > 1e-8:
            attn_t = (attn_t - attn_min) / (attn_max - attn_min)
        else:
            attn_t = torch.zeros_like(attn_t)

        # 应用colormap
        attn_colored = (colormap(attn_t.cpu().numpy())[:, :, :3] * 255).astype(np.uint8)

        # 与原图叠加 (alpha blending)
        overlay = (alpha * attn_colored + (1 - alpha) * rgb_img).astype(np.uint8)

        # 保存
        token_name = token_names[t] if token_names else f"token{t}"
        filename = f"{save_path}_token{t:02d}_{token_name}.png"

        # 创建对比图：原图|attention热力图|叠加图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(rgb_img)
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        axes[1].imshow(attn_t.cpu().numpy(), cmap='jet')
        axes[1].set_title(f"Attention: {token_name}")
        axes[1].axis('off')

        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved attention visualization: {filename}")


# utils/infer.py

def extract_attention_from_model(model, stage="dec_last"):
    """
    从模型中提取指定stage的SAM attention

    Returns:
        list of (attention_map, spatial_shape):
            attention_map: (B, N, H, T) 或 None
            spatial_shape: (h, w) tuple 或 None
    """
    if hasattr(model, 'module'):
        actual_model = model.module
    else:
        actual_model = model

    attention_data = []

    if stage.startswith("dec"):
        decode_head = actual_model.decode_head
        if hasattr(decode_head, 'dec_sam_layers'):
            if stage == "dec_last":
                for i in range(len(decode_head.dec_sam_layers) - 1, -1, -1):
                    sam_layer = decode_head.dec_sam_layers[i]
                    if sam_layer is not None and hasattr(sam_layer, 'last_attention_map'):
                        attn = sam_layer.last_attention_map
                        spatial_shape = getattr(sam_layer, 'last_spatial_shape', None)
                        if attn is not None:
                            attention_data.append((attn, spatial_shape))
                            break
            else:  # dec_all
                for i, sam_layer in enumerate(decode_head.dec_sam_layers):
                    if sam_layer is not None and hasattr(sam_layer, 'last_attention_map'):
                        attn = sam_layer.last_attention_map
                        spatial_shape = getattr(sam_layer, 'last_spatial_shape', None)
                        if attn is not None:
                            attention_data.append((attn, spatial_shape))

    elif stage.startswith("enc"):
        backbone = actual_model.backbone
        if hasattr(backbone, 'encoder_sam_blocks'):
            if stage == "enc_last":
                for i in range(len(backbone.encoder_sam_blocks) - 1, -1, -1):
                    blocks = backbone.encoder_sam_blocks[i]
                    if len(blocks) > 0:
                        last_block = blocks[-1]
                        if hasattr(last_block, 'last_attention_map'):
                            attn = last_block.last_attention_map
                            spatial_shape = getattr(last_block, 'last_spatial_shape', None)
                            if attn is not None:
                                attention_data.append((attn, spatial_shape))
                                break
            else:  # enc_all
                for i, blocks in enumerate(backbone.encoder_sam_blocks):
                    for block in blocks:
                        if hasattr(block, 'last_attention_map'):
                            attn = block.last_attention_map
                            spatial_shape = getattr(block, 'last_spatial_shape', None)
                            if attn is not None:
                                attention_data.append((attn, spatial_shape))

    return attention_data if attention_data else [(None, None)]


@torch.no_grad()
def evaluate_with_attention(model, dataloader, config, device, engine,
                            save_dir=None, vis_stage="dec_last",
                            num_images=None, alpha=0.5):
    """
    带attention可视化的评估函数
    """
    from utils.metrics_new import Metrics

    logger.info(f"Starting evaluation with attention visualization...")
    logger.info(f"Visualization stage: {vis_stage}, num_images: {num_images}, alpha: {alpha}")

    attn_data = extract_attention_from_model(model, vis_stage)

    model.eval()
    n_classes = config.num_classes
    metrics = Metrics(n_classes, config.background, device)

    # 启用所有SAM的attention保存
    def enable_attention_save(m):
        if isinstance(m, SemanticAlignmentModule):
            m.save_attention = True

    model.apply(enable_attention_save)

    processed_images = 0

    for idx, minibatch in enumerate(dataloader):
        if num_images and processed_images >= num_images:
            logger.info(f"Reached target number of images ({num_images}), stopping...")
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

        # Forward pass
        preds = model(images_gpu, modal_xs_gpu, text_features=text_feats)

        # 提取attention maps
        attn_maps = extract_attention_from_model(model, vis_stage)

        # 更新metrics
        metrics.update(preds.softmax(dim=1), labels_gpu)

        # 可视化attention
        if save_dir and attn_data[0][0] is not None:
            B = images_gpu.shape[0]

            for b in range(B):
                if num_images and processed_images >= num_images:
                    break

                # 准备RGB图像（denormalize）
                rgb_tensor = images[b]  # (3, H, W)
                rgb_np = rgb_tensor.permute(1, 2, 0).cpu().numpy()

                # Denormalize
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
                for map_idx, (attn_map, spatial_shape) in enumerate(attn_data):
                    if attn_map is None:
                        continue

                    # attn_map: (B, N, H_heads, T)
                    if attn_map.dim() == 4:
                        attn_single = attn_map[b].mean(dim=1)  # (N, T)
                        N, T = attn_single.shape

                        # ★ 使用保存的空间形状
                        if spatial_shape is not None:
                            h_attn, w_attn = spatial_shape
                            if h_attn * w_attn == N:
                                attn_2d = attn_single.reshape(h_attn, w_attn, T)
                            else:
                                logger.warning(
                                    f"Spatial shape mismatch: {h_attn}x{w_attn}={h_attn * w_attn} != N={N}, skipping...")
                                continue
                        else:
                            # 降级方案：尝试正方形
                            h_attn = int(np.sqrt(N))
                            w_attn = h_attn
                            if h_attn * w_attn == N:
                                attn_2d = attn_single.reshape(h_attn, w_attn, T)
                            else:
                                logger.warning(
                                    f"Cannot reshape N={N} to square and no spatial_shape provided, skipping...")
                                continue

                    elif attn_map.dim() == 3:
                        # (B, N, T)
                        attn_single = attn_map[b]  # (N, T)
                        N, T = attn_single.shape

                        h_attn = int(np.sqrt(N))
                        w_attn = h_attn

                        if h_attn * w_attn == N:
                            attn_2d = attn_single.reshape(h_attn, w_attn, T)
                        else:
                            logger.warning(f"Cannot reshape N={N} to square, skipping...")
                            continue
                    else:
                        logger.warning(f"Unsupported attention shape: {attn_map.shape}")
                        continue

                    # 上采样到原图尺寸
                    attn_resized = F.interpolate(
                        attn_2d.permute(2, 0, 1).unsqueeze(0),  # (1, T, H', W')
                        size=(H_img, W_img),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0).permute(1, 2, 0)  # (H, W, T)

                    # 保存路径
                    stage_name = vis_stage if len(attn_maps) == 1 else f"{vis_stage}_map{map_idx}"
                    save_prefix = os.path.join(save_dir, "attention", stage_name, fn)
                    os.makedirs(os.path.dirname(save_prefix), exist_ok=True)

                    # 生成token names（如果有的话）
                    token_names = [f"tok{t}" for t in range(T)]

                    # 可视化
                    visualize_attention_maps(
                        attn_resized,
                        rgb_np,
                        save_prefix,
                        token_names=token_names,
                        alpha=alpha
                    )

                    logger.info(f"Processed attention for {fn} (map {map_idx}/{len(attn_maps)})")

                processed_images += 1

    # 关闭attention保存
    def disable_attention_save(m):
        if isinstance(m, SemanticAlignmentModule):
            m.save_attention = False
            m.last_attention_map = None

    model.apply(disable_attention_save)

    logger.info(f"Attention visualization completed for {processed_images} images")

    # 返回metrics（兼容原有逻辑）
    if engine.distributed:
        all_metrics = [None for _ in range(engine.world_size)]
        torch.distributed.all_gather_object(all_metrics, metrics)
    else:
        all_metrics = metrics

    return all_metrics


with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()
    config = getattr(import_module(args.config), "C")

    # Override text guidance settings if supplied by CLI.
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

    config.pad = False  # Do not pad when inference
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

    # 根据是否需要attention可视化选择评估函数
    if args.save_attention:
        logger.info("=" * 80)
        logger.info("RUNNING WITH ATTENTION VISUALIZATION")
        logger.info(f"Stage: {args.vis_stage}")
        logger.info(f"Number of images: {args.num_images if args.num_images else 'all'}")
        logger.info(f"Alpha: {args.attention_alpha}")
        logger.info(f"Save path: {args.save_path}")
        logger.info("=" * 80)

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
                num_images=args.num_images,
                alpha=args.attention_alpha
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
        # 原有的评估逻辑（多尺度测试）
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