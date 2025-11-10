import argparse
import pprint
import time
from importlib import import_module
import json
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from models.builder import EncoderDecoder as segmodel
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
from utils.val_mm import evaluate, evaluate_msf
from utils.dataloader.dataloader import get_val_loader
from utils.dataloader.RGBXDataset import RGBXDataset
from utils.engine.engine import Engine
from utils.engine.logger import get_logger


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument("--config", help="train config file path")
parser.add_argument("--gpus", help="used gpu number")
parser.add_argument("-v", "--verbose", default=False, action="store_true")
parser.add_argument("--epochs", default=0)
parser.add_argument("--show_image", "-s", default=False, action="store_true")
parser.add_argument("--save_path", default=None)
parser.add_argument("--checkpoint_dir")
parser.add_argument("--continue_fpath")
parser.add_argument("--sliding", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--compile", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--compile_mode", default="default")
parser.add_argument("--syncbn", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--mst", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--amp", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--val_amp", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--pad_SUNRGBD", default=False, action=argparse.BooleanOptionalAction)

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
parser.add_argument("--sam-enc-stages", type=str, default="1,2,3", help="Comma separated, e.g., 0,2")
parser.add_argument("--sam-dec-stages", type=str, default="1,2,3", help="Comma separated, e.g., 1,3")
parser.add_argument("--superpower", default=False, action=argparse.BooleanOptionalAction)

# --- SAM temperature & attention styles ---
parser.add_argument(
    "--sam-decoder-use-cosine",
    dest="sam_decoder_use_cosine",
    default=None,
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--sam-decoder-learnable-temp",
    dest="sam_decoder_learnable_temp",
    default=None,
    action=argparse.BooleanOptionalAction,
)
parser.add_argument("--sam-decoder-logit-init", type=float, default=None)
parser.add_argument(
    "--sam-encoder-use-cosine",
    dest="sam_encoder_use_cosine",
    default=None,
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--sam-encoder-learnable-temp",
    dest="sam_encoder_learnable_temp",
    default=None,
    action=argparse.BooleanOptionalAction,
)
parser.add_argument("--sam-encoder-logit-init", type=float, default=None)

torch.set_float32_matmul_precision("high")
import torch._dynamo
torch._dynamo.config.suppress_errors = True


def _parse_stages(s: str):
    if s is None:
        return [0,1,2,3]
    s = str(s).strip()
    if not s:
        return []
    out = []
    for x in s.split(","):
        x = x.strip()
        if x == "":
            continue
        try:
            out.append(int(x))
        except:
            pass
    return out


with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()
    config = getattr(import_module(args.config), "C")

    # === override text guidance config by CLI (if provided) ===
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

    # === SAM per-stage ===
    config.sam_enc_stages = _parse_stages(args.sam_enc_stages)
    config.sam_dec_stages = _parse_stages(args.sam_dec_stages)
    config.superpower = bool(getattr(args, "superpower", False))

    # === SAM temperature & attention styles ===
    if args.sam_decoder_use_cosine is not None:
        config.sam_decoder_use_cosine = bool(args.sam_decoder_use_cosine)
    if args.sam_decoder_learnable_temp is not None:
        config.sam_decoder_learnable_temp = bool(args.sam_decoder_learnable_temp)
    if args.sam_decoder_logit_init is not None:
        config.sam_decoder_logit_init = float(args.sam_decoder_logit_init)
    if args.sam_encoder_use_cosine is not None:
        config.sam_encoder_use_cosine = bool(args.sam_encoder_use_cosine)
    if args.sam_encoder_learnable_temp is not None:
        config.sam_encoder_learnable_temp = bool(args.sam_encoder_learnable_temp)
    if args.sam_encoder_logit_init is not None:
        config.sam_encoder_logit_init = float(args.sam_encoder_logit_init)

    logger = get_logger(config.log_dir, config.log_file, rank=engine.local_rank)

    # check if pad_SUNRGBD is used correctly
    if args.pad_SUNRGBD and config.dataset_name != "SUNRGBD":
        args.pad_SUNRGBD = False
        logger.warning("pad_SUNRGBD is only used for SUNRGBD dataset")
    if (args.pad_SUNRGBD) and (not config.backbone.startswith("DFormerv2")):
        raise ValueError("DFormerv1 is not recommended with pad_SUNRGBD")
    if (not args.pad_SUNRGBD) and config.backbone.startswith("DFormerv2") and config.dataset_name == "SUNRGBD":
        raise ValueError("DFormerv2 is not recommended without pad_SUNRGBD")
    config.pad = args.pad_SUNRGBD

    cudnn.benchmark = True
    if config.dataset_name != "SUNRGBD":
        val_batch_size = int(config.batch_size)
    elif not args.pad_SUNRGBD:
        val_batch_size = int(args.gpus)
    else:
        val_batch_size = 8 * int(args.gpus)

    if args.mst:
        val_loader, val_sampler = get_val_loader(
            engine,
            RGBXDataset,
            config,
            val_batch_size=val_batch_size,
        )
    else:
        val_loader, val_sampler = get_val_loader(
            engine,
            RGBXDataset,
            config,
            val_batch_size=val_batch_size,
        )
    logger.info(f"val dataset len:{len(val_loader) * int(args.gpus)}")

    if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
        tb_dir = config.tb_dir + "/{}".format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
        generate_tb_dir = config.tb_dir + "/tb"
        tb = SummaryWriter(log_dir=tb_dir)
        engine.link_tb(tb_dir, generate_tb_dir)
        pp = pprint.PrettyPrinter(indent=4)
        logger.info("config: \n" + pp.pformat(config))

    logger.info("args parsed:")
    for k in args.__dict__:
        logger.info(k + ": " + str(args.__dict__[k]))

    criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=config.background)

    if args.syncbn:
        BatchNorm2d = nn.SyncBatchNorm
        logger.info("using syncbn")
    else:
        BatchNorm2d = nn.BatchNorm2d
        logger.info("using regular bn")

    model = segmodel(
        cfg=config,
        criterion=criterion,
        norm_layer=BatchNorm2d,
        syncbn=args.syncbn,
    )

    weight = torch.load(args.continue_fpath, map_location=torch.device("cpu"))
    if "model" in weight:
        weight = weight["model"]
    elif "state_dict" in weight:
        weight = weight["state_dict"]

    logger.info(f"load model from {args.continue_fpath}")
    print(model.load_state_dict(weight, strict=False))

    if engine.distributed:
        logger.info(".............distributed training.............")
        if torch.cuda.is_available():
            device = torch.device("cuda")
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

    if args.compile:
        model = torch.compile(model, backend="inductor", mode=args.compile_mode)

    torch.cuda.empty_cache()
    if args.amp:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            if engine.distributed:
                with torch.no_grad():
                    model.eval()
                    device = torch.device("cuda")
                    if args.mst:
                        all_metrics = evaluate_msf(
                            model,
                            val_loader,
                            config,
                            device,
                            [0.5, 0.75, 1.0, 1.25, 1.5],
                            True,
                            engine,
                            sliding=args.sliding,
                        )
                    else:
                        all_metrics = evaluate(
                            model,
                            val_loader,
                            config,
                            device,
                            engine,
                            sliding=args.sliding,
                        )
                    if engine.local_rank == 0:
                        metric = all_metrics[0]
                        for other_metric in all_metrics[1:]:
                            metric.update_hist(other_metric.hist)
                        ious, miou = metric.compute_iou()
                        acc, macc = metric.compute_pixel_acc()
                        f1, mf1 = metric.compute_f1()
                        print("miou", miou)
                        logger.info(f"miou:{miou}, macc:{macc}, mf1:{mf1}")
                        logger.info(f"ious:{ious}")

            elif not engine.distributed:
                with torch.no_grad():
                    model.eval()
                    device = torch.device("cuda")
                    if args.mst:
                        metric = evaluate_msf(
                            model,
                            val_loader,
                            config,
                            device,
                            [0.5, 0.75, 1.0, 1.25, 1.5],
                            True,
                            engine,
                            sliding=args.sliding,
                        )
                    else:
                        metric = evaluate(
                            model,
                            val_loader,
                            config,
                            device,
                            engine,
                            sliding=args.sliding,
                        )
                    ious, miou = metric.compute_iou()
                    acc, macc = metric.compute_pixel_acc()
                    f1, mf1 = metric.compute_f1()
                    logger.info(f"miou:{miou}, macc:{macc}, mf1:{mf1}")
                    logger.info(f"ious:{ious}")
    else:
        if engine.distributed:
            with torch.no_grad():
                model.eval()
                device = torch.device("cuda")
                if args.mst:
                    all_metrics = evaluate_msf(
                        model,
                        val_loader,
                        config,
                        device,
                        [0.5, 0.75, 1.0, 1.25, 1.5],
                        True,
                        engine,
                        sliding=args.sliding,
                    )
                else:
                    all_metrics = evaluate(
                        model,
                        val_loader,
                        config,
                        device,
                        engine,
                        sliding=args.sliding,
                    )
                if engine.local_rank == 0:
                    metric = all_metrics[0]
                    for other_metric in all_metrics[1:]:
                        metric.update_hist(other_metric.hist)
                    ious, miou = metric.compute_iou()
                    acc, macc = metric.compute_pixel_acc()
                    f1, mf1 = metric.compute_f1()
                    print("miou", miou)
                    logger.info(f"miou:{miou}, macc:{macc}, mf1:{mf1}")
                    logger.info(f"ious:{ious}")

        elif not engine.distributed:
            with torch.no_grad():
                model.eval()
                device = torch.device("cuda")
                if args.mst:
                    metric = evaluate_msf(
                        model,
                        val_loader,
                        config,
                        device,
                        [0.5, 0.75, 1.0, 1.25, 1.5],
                        True,
                        engine,
                        sliding=args.sliding,
                    )
                else:
                    metric = evaluate(
                        model,
                        val_loader,
                        config,
                        device,
                        engine,
                        sliding=args.sliding,
                    )
                ious, miou = metric.compute_iou()
                acc, macc = metric.compute_pixel_acc()
                f1, mf1 = metric.compute_f1()
                logger.info(f"miou:{miou}, macc:{macc}, mf1:{mf1}")
                logger.info(f"ious:{ious}")

    logger.info("end testing")