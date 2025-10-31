import argparse
import importlib
import os
import random
import sys
import time
from importlib import import_module

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.builder import EncoderDecoder as segmodel
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

# from eval import evaluate_mid

# SEED=1
# # np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed_all(SEED)
# torch.backends.cudnn.deterministic=False
# torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser()
parser.add_argument("--config", help="train config file path")
parser.add_argument("--gpus", help="used gpu number")
# parser.add_argument('-d', '--devices', default='0,1', type=str)
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


# parser.add_argument('--save_path', '-p', default=None)
logger = get_logger()

# os.environ['MASTER_PORT'] = '169710'

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
    print(len(val_loader))

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

    print("load model")
    model.load_state_dict(weight, strict=False)

    if engine.distributed:
        logger.info(".............distributed training.............")
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

    logger.info("begin testing:")
    best_miou = 0.0
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
    # val_pre = ValPre()
    # val_dataset = RGBXDataset(data_setting, 'val', val_pre)
    # test_loader, test_sampler = get_test_loader(engine, RGBXDataset,config)
    all_dev = [0]

    # segmentor = SegEvaluator(val_dataset, config.num_classes, config.norm_mean,
    #                                 config.norm_std, None,
    #                                 config.eval_scale_array, config.eval_flip,
    #                                 all_dev, config,args.verbose, args.save_path,args.show_image)

    if engine.distributed:
        print("multi GPU test")
        with torch.no_grad():
            model.eval()
            device = torch.device("cuda")
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
            # all_metrics = evaluate_msf(
            #     model,
            #     val_loader,
            #     config,
            #     device,
            #     [0.5, 0.75, 1.0, 1.25, 1.5],
            #     True,
            #     engine,
            # )
            if engine.local_rank == 0:
                metric = all_metrics[0]
                for other_metric in all_metrics[1:]:
                    metric.update_hist(other_metric.hist)
                ious, miou = metric.compute_iou()
                acc, macc = metric.compute_pixel_acc()
                f1, mf1 = metric.compute_f1()
                print(miou, "---------")
    else:
        with torch.no_grad():
            model.eval()
            device = torch.device("cuda")
            # metric=evaluate(model, val_loader,config, device, engine)
            # print('acc, macc, f1, mf1, ious, miou',acc, macc, f1, mf1, ious, miou)
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
            print("miou", miou)
