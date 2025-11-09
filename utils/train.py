import argparse
import datetime
import os
import pprint
import random
import time
from importlib import import_module
import tempfile
import json
import numpy as np
import torch.distributed as dist
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
from val_mm import evaluate, evaluate_msf
from pathlib import Path
from models.builder import EncoderDecoder as segmodel
from utils.dataloader.dataloader import get_train_loader, get_val_loader
from utils.dataloader.RGBXDataset import RGBXDataset
from utils.engine.engine import Engine
from utils.engine.logger import get_logger
from utils.init_func import configure_optimizers, group_weight
from utils.lr_policy import WarmUpPolyLR
import torch.distributed as dist
from utils.pyt_utils import all_reduce_tensor

# Ensure HuggingFace tokenizers run in single-thread mode before dataloaders fork
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_TOKENIZERS_PARALLELISM", "false")

parser = argparse.ArgumentParser()
parser.add_argument("--config", help="train config file path")
parser.add_argument("--gpus", default=2, type=int, help="used gpu number")
# parser.add_argument('-d', '--devices', default='0,1', type=str)
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
parser.add_argument("--use_seed", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--seed", type=int, default=None, help="Random seed (overrides config.seed if provided)")
parser.add_argument("--local-rank", default=0)

# --- text guidance runtime switches ---
parser.add_argument("--enable-text-guidance", dest="enable_text_guidance", default=None, action=argparse.BooleanOptionalAction, help="Enable/disable text guidance (overrides config)")
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
parser.add_argument("--sam-enc-stages", type=str, default="0,1,2,3", help="Comma separated, e.g., 0,2")
parser.add_argument("--sam-dec-stages", type=str, default="0,1,2,3", help="Comma separated, e.g., 1,3")
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

# os.environ['MASTER_PORT'] = '169710'
torch.set_float32_matmul_precision("high")
import torch._dynamo
torch._dynamo.config.suppress_errors = True
# torch._dynamo.config.automatic_dynamic_shapes = False

def _parse_stages(s: str):
    if s is None:
        return [0,1,2,3]
    s = str(s).strip()
    if not s:
        # 空串代表“不插入任何层”
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

def is_eval(epoch, config):
    return epoch > int(config.checkpoint_start_epoch) or epoch == 1 or epoch % 10 == 0

class gpu_timer:
    def __init__(self, beta=0.6) -> None:
        self.start_time = None
        self.stop_time = None
        self.mean_time = None
        self.beta = beta
        self.first_call = True
    def start(self):
        torch.cuda.synchronize()
        self.start_time = time.perf_counter()
    def stop(self):
        if self.start_time is None:
            print("Use start() before stop(). ")
        torch.cuda.synchronize()
        self.stop_time = time.perf_counter()
        elapsed = self.stop_time - self.start_time
        self.start_time = None
        if self.first_call:
            self.mean_time = elapsed
            self.first_call = False
        else:
            self.mean_time = self.beta * self.mean_time + (1 - self.beta) * elapsed

def set_seed(seed):
    # seed init.
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # torch seed init.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True  # train speed is slower after enabling this opts.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True, warn_only=True)

with Engine(custom_parser=parser) as engine:
    try:
        args = parser.parse_args()

        config = getattr(import_module(args.config), "C")
        # === override text guidance config by CLI (if provided) ===

        if args.enable_text_guidance is not None:
            config.enable_text_guidance = args.enable_text_guidance
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

        # === Seed Configuration ===
        if args.use_seed:
            # Deterministic mode: use CLI seed if provided, otherwise use config seed
            actual_seed = args.seed if args.seed is not None else config.seed
            set_seed(actual_seed)
            seed_source = "CLI argument" if args.seed is not None else "config file"
            logger.info(f"✓ Deterministic training enabled with seed: {actual_seed} (from {seed_source})")
        else:
            # Non-deterministic mode: generate 5-digit random seed for logging/reproducibility
            # IMPORTANT: In distributed training, rank 0 generates seed and broadcasts to others
            if engine.distributed:
                if engine.local_rank == 0:
                    actual_seed = random.randint(10000, 99999)
                    seed_tensor = torch.tensor(actual_seed, dtype=torch.int64, device='cuda')
                else:
                    seed_tensor = torch.tensor(0, dtype=torch.int64, device='cuda')
                # Broadcast seed from rank 0 to all ranks
                dist.broadcast(seed_tensor, src=0)
                actual_seed = int(seed_tensor.item())
            else:
                actual_seed = random.randint(10000, 99999)

            # Set the seed for basic reproducibility (all ranks use same seed)
            random.seed(actual_seed)
            np.random.seed(actual_seed)
            torch.manual_seed(actual_seed)
            torch.cuda.manual_seed_all(actual_seed)
            # Enable benchmark for speed (non-deterministic)
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            logger.info(f"✓ Non-deterministic training (faster) with seed: {actual_seed}")
            logger.info(f"  To reproduce: modify config.seed={actual_seed} and use --use_seed")

        # Store actual seed for logging/checkpointing
        config.actual_seed = actual_seed

        if not args.compile and args.compile_mode != "default":
            logger.warning("compile_mode is only valid when compile is enabled, ignoring compile_mode")

        # ==== [Fix] 指定节点本地缓存目录 + 预热，避免并发下载竞态/NFS冲突 ====

        try:
            import socket, tempfile
            node_cache = os.path.join(tempfile.gettempdir(), f"hf_cache_{socket.gethostname()}")
            os.environ.setdefault("HF_HOME", node_cache)
            os.environ.setdefault("TRANSFORMERS_CACHE", node_cache)
            os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
            logger.info(f"[HF cache] Using cache dir: {node_cache}")
        except Exception as _e:
            logger.warning(f"[HF cache] set cache dir skipped: {_e}")

        def _warmup_text_cache(cfg):
            try:
                from utils.prompt_utils import encode_prompts
                _ = encode_prompts(
                    ["cache warmup"],  # 任意一句
                    encoder=getattr(cfg, "text_encoder", "jinaclip"),
                    encoder_name=getattr(cfg, "text_encoder_name", None),
                    target_dim=getattr(cfg, "text_feature_dim", 512),
                )
                logger.info("[HF warmup] text encoder cache ready.")
            except Exception as _e:
                # 如果节点不联网，可以提前把模型下到缓存，再设置 TRANSFORMERS_OFFLINE=1
                logger.warning(f"[HF warmup] skipped: {_e}")


        def _prebuild_text_caches(cfg):
            """
            由 rank0 在构建 DataLoader 之前预先实例化一次 Dataset（split='train'），
            触发 RGBXDataset 的 classbank/captions/imglabels 缓存构建。
            其他 rank 只需 barrier 后直接读缓存。
            """
            if not getattr(cfg, "enable_text_guidance", False):
                logger.info("[Text prebuild] skipped: text guidance disabled.")
                return

            # 没有可用的文本资产就没有预构建的必要
            if not any(
                getattr(cfg, _key, None)
                for _key in ("label_txt_path", "caption_json_path", "image_labels_json_path")
            ):
                logger.info("[Text prebuild] skipped: no text assets configured.")
                return

            try:
                from utils.dataloader.RGBXDataset import RGBXDataset
                data_setting = {
                    "rgb_root": cfg.rgb_root_folder,
                    "rgb_format": cfg.rgb_format,
                    "gt_root": cfg.gt_root_folder,
                    "gt_format": cfg.gt_format,
                    "transform_gt": cfg.gt_transform,
                    "x_root": cfg.x_root_folder,
                    "x_format": cfg.x_format,
                    "x_single_channel": cfg.x_is_single_channel,
                    "class_names": cfg.class_names,
                    "train_source": cfg.train_source,
                    "eval_source": cfg.eval_source,
                    "dataset_name": cfg.dataset_name,
                    "backbone": cfg.backbone,
                    "enable_text_guidance": getattr(cfg, "enable_text_guidance", False),
                    "label_txt_path": getattr(cfg, "label_txt_path", None),
                    "caption_json_path": getattr(cfg, "caption_json_path", None),
                    "image_labels_json_path": getattr(cfg, "image_labels_json_path", None),
                    "text_template_set": getattr(cfg, "text_template_set", "clip"),
                    "max_templates_per_label": getattr(cfg, "max_templates_per_label", 3),
                    "text_source": getattr(cfg, "text_source", "both"),
                    "text_encoder": getattr(cfg, "text_encoder", "jinaclip"),
                    "text_encoder_name": getattr(cfg, "text_encoder_name", None),
                    "text_feature_dim": getattr(cfg, "text_feature_dim", 512),
                    "max_caption_sentences": getattr(cfg, "max_caption_sentences", 8),
                    "caption_topk": getattr(cfg, "caption_topk", 0),
                    "caption_topk_mode": getattr(cfg, "caption_topk_mode", "class_sim"),
                    "max_image_labels": getattr(cfg, "max_image_labels", 0),
                }
                # 只实例化，不用 loader；触发 __init__ -> _prepare_text_guidance_assets() -> *_encode_*()
                dataset = RGBXDataset(data_setting, "train", preprocess=None)
                built_parts = []
                if getattr(dataset, "enable_text_guidance", False):
                    if getattr(dataset, "_use_label_text", False) and dataset.class_text_features is not None and dataset.class_text_features.numel() > 0:
                        built_parts.append("labels")
                    if getattr(dataset, "_use_caption_text", False) and isinstance(dataset.caption_text_features, dict) and len(dataset.caption_text_features) > 0:
                        built_parts.append("captions")
                    if getattr(dataset, "_use_imglabel_text", False) and isinstance(dataset.imglabel_text_features, dict) and len(dataset.imglabel_text_features) > 0:
                        built_parts.append("imglabels")
                summary = ", ".join(built_parts) if built_parts else "none"
                logger.info(f"[Text prebuild] text caches are ready for: {summary}.")
            except Exception as _e:
                logger.warning(f"[Text prebuild] skipped: {_e}")


        if engine.distributed and dist.is_available() and dist.is_initialized():
            if engine.local_rank == 0:
                logger.info(f"[warmup] rank{engine.local_rank} start")
                _warmup_text_cache(config)
                logger.info(f"[warmup] rank{engine.local_rank} done")
            logger.info(f"[warmup] rank{engine.local_rank} waiting barrier#1")
            dist.barrier()
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            logger.info(f"[warmup] rank{engine.local_rank} waiting barrier#2")
            dist.barrier()
        else:
            _warmup_text_cache(config)

        # === 预构建文本缓存（仅 rank0 实际执行；其余 rank 等待后直接读缓存） ===
        if engine.distributed and dist.is_available() and dist.is_initialized():
            try:
                if engine.local_rank == 0:
                    logger.info(f"[prebuild] rank{engine.local_rank} start")
                    _prebuild_text_caches(config)
                    logger.info(f"[prebuild] rank{engine.local_rank} done")
            except Exception as e:
                # 打印异常，不要让 rank0 提前 return，确保能走到 barrier
                logger.exception(f"[prebuild] rank{engine.local_rank} failed: {e}")
            finally:
                logger.info(f"[prebuild] rank{engine.local_rank} waiting barrier")
                dist.barrier()
        else:
            _prebuild_text_caches(config)

        train_loader, train_sampler = get_train_loader(engine, RGBXDataset, config)

        val_dl_factor = 4

        # SUNRGBD: increase from gpus*1 to gpus*16 for ~4x speedup
        # Other datasets: use 4x training batch
        if config.dataset_name == "SUNRGBD":
            val_batch_size = 32  # More aggressive for SUNRGBD
            logger.info(f"Using larger validation batch size for SUNRGBD: {val_batch_size}")
        else:
            val_batch_size = int(config.batch_size * val_dl_factor)
            logger.info(f"Using {val_dl_factor}x training batch for validation: {val_batch_size}")

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
            try:
                tb = SummaryWriter(log_dir=tb_dir)
                engine.link_tb(tb_dir, generate_tb_dir)
            except OSError as e:
                logger.warning(
                    f"Could not create TensorBoard directory {tb_dir}: {e}. "
                    "Using a temporary location instead."
                )
                tmp_dir = tempfile.mkdtemp(prefix="tb-")
                tb = SummaryWriter(log_dir=tmp_dir)
            pp = pprint.PrettyPrinter(indent=4)
            logger.info("config: \n" + pp.pformat(config))

        logger.info("args parsed:")
        for k in args.__dict__:
            logger.info(k + ": " + str(args.__dict__[k]))

        criterion = nn.CrossEntropyLoss(reduction="none", ignore_index=config.background)

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
        base_lr = config.lr
        if engine.distributed:
            base_lr = config.lr

        params_list = []
        params_list = group_weight(params_list, model, BatchNorm2d, base_lr)

        if config.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(
                params_list,
                lr=base_lr,
                betas=(0.9, 0.999),
                weight_decay=config.weight_decay,
            )
        elif config.optimizer == "SGDM":
            optimizer = torch.optim.SGD(
                params_list,
                lr=base_lr,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
            )
        else:
            raise NotImplementedError

        total_iteration = config.nepochs * config.niters_per_epoch
        lr_policy = WarmUpPolyLR(
            base_lr,
            config.lr_power,
            total_iteration,
            config.niters_per_epoch * config.warm_up_epoch,
        )

        device = torch.device(
            f"cuda:{engine.local_rank}" if torch.cuda.is_available() else "cpu"
        )

        if engine.distributed:
            logger.info(".............distributed training.............")
            if torch.cuda.is_available():
                model.cuda()
                model = DistributedDataParallel(
                    model,
                    device_ids=[engine.local_rank],
                    output_device=engine.local_rank,
                    find_unused_parameters=True,
                    gradient_as_bucket_view=False,
                )
        else:
            model.to(device)

        engine.register_state(dataloader=train_loader, model=model, optimizer=optimizer)
        if engine.continue_state_object:
            engine.restore_checkpoint()

        optimizer.zero_grad()

        logger.info("begin trainning:")
        data_setting = {
            "rgb_root": config.rgb_root_folder,
            "rgb_format": config.rgb_format,
            "gt_root": config.gt_root_folder,
            "gt_format": config.gt_format,
            "transform_gt": config.gt_transform,
            "x_root": config.x_root_folder,
            "x_format": config.x_format,
            "x_single_channel": config.x_is_single_channel,
            "class_names": config.class_names,
            "train_source": config.train_source,
            "eval_source": config.eval_source,
        }

        uncompiled_model = model
        compiled_model = torch.compile(model, backend="inductor", mode=args.compile_mode) if args.compile else model
        miou, best_miou = 0.0, 0.0
        train_timer = gpu_timer()
        eval_timer = gpu_timer()

        if args.amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(engine.state.epoch, config.nepochs + 1):
            model = compiled_model
            model.train()
            if engine.distributed:
                train_sampler.set_epoch(epoch)

            dataloader = iter(train_loader)
            sum_loss = 0
            train_timer.start()

            for idx in range(config.niters_per_epoch):
                optimizer.zero_grad(set_to_none=True)
                engine.update_iteration(epoch, idx)

                minibatch = next(dataloader)
                imgs = minibatch["data"]
                gts = minibatch["label"]
                modal_xs = minibatch["modal_x"]

                text_feats = minibatch.get("text_features")

                imgs = imgs.cuda(non_blocking=True)
                gts = gts.cuda(non_blocking=True)
                modal_xs = modal_xs.cuda(non_blocking=True)

                if text_feats is not None:
                    text_feats = text_feats.cuda(non_blocking=True).float()

                if args.amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        loss = model(imgs, modal_xs, label=gts, text_features=text_feats)
                else:
                    loss = model(imgs, modal_xs, label=gts, text_features=text_feats)

                # reduce the whole loss over multi-gpu
                if engine.distributed:
                    reduce_loss = all_reduce_tensor(loss, world_size=engine.world_size)

                if args.amp:
                    scaler.scale(loss).backward()
                    # Gradient clipping for stability
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                # if not args.amp:
                #     if epoch == 1:
                #         for name, param in model.named_parameters():
                #             if param.grad is None:
                #                 logger.warning(f"{name} has no grad, please check")

                current_idx = (epoch - 1) * config.niters_per_epoch + idx
                lr = lr_policy.get_lr(current_idx)
                for i in range(len(optimizer.param_groups)):
                    optimizer.param_groups[i]["lr"] = lr

                if engine.distributed:
                    sum_loss += reduce_loss.item()
                    print_str = (
                        "Epoch {}/{}".format(epoch, config.nepochs)
                        + " Iter {}/{}:".format(idx + 1, config.niters_per_epoch)
                        + " lr=%.4e" % lr
                        + " loss=%.4f total_loss=%.4f" % (reduce_loss.item(), (sum_loss / (idx + 1)))
                    )
                else:
                    # 维持原有打印逻辑
                    sum_loss += loss.item()
                    print_str = (
                        f"Epoch {epoch}/{config.nepochs} "
                        + f"Iter {idx + 1}/{config.niters_per_epoch}: "
                        + f"lr={lr:.4e} loss={loss:.4f} total_loss={(sum_loss / (idx + 1)):.4f}"
                    )

                if ((idx + 1) % int((config.niters_per_epoch) * 0.1) == 0 or idx == 0) and (
                    (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed)
                ):
                    print(print_str)

                del loss
            logger.info(print_str)
            train_timer.stop()

            logger.info(f"[Train] epoch {epoch} - before is_eval()")
            for h in logger.handlers: h.flush()

            if is_eval(epoch, config):
                logger.info("[Eval] Entering evaluation—val_loader length = %d", len(val_loader))
                for h in logger.handlers: h.flush()
                eval_timer.start()
                torch.cuda.empty_cache()

                try:
                    if engine.distributed:
                        with torch.no_grad():
                            model.eval()
                            device = torch.device("cuda")
                            if args.val_amp:
                                with torch.autocast(device_type="cuda", dtype=torch.float16):
                                    if args.mst:
                                        all_metrics = evaluate_msf(
                                            model, val_loader, config, device,
                                            [0.5, 0.75, 1.0, 1.25, 1.5], True, engine,
                                            sliding=args.sliding,
                                        )
                                    else:
                                        all_metrics = evaluate(
                                            model, val_loader, config, device, engine,
                                            sliding=args.sliding,
                                        )
                            else:
                                if args.mst:
                                    all_metrics = evaluate_msf(
                                        model, val_loader, config, device,
                                        [0.5, 0.75, 1.0, 1.25, 1.5], True, engine,
                                        sliding=args.sliding,
                                    )
                                else:
                                    all_metrics = evaluate(
                                        model, val_loader, config, device, engine,
                                        sliding=args.sliding,
                                    )
                            if engine.local_rank == 0:
                                metric = all_metrics[0]
                                for other_metric in all_metrics[1:]:
                                    metric.update_hist(other_metric.hist)
                                ious, miou = metric.compute_iou()
                                acc, macc = metric.compute_pixel_acc()
                                f1, mf1 = metric.compute_f1()
                                if miou > best_miou:
                                    best_miou = miou
                                    engine.save_and_link_checkpoint(
                                        config.log_dir,
                                        config.log_dir,
                                        config.log_dir_link,
                                        infor="_miou_" + str(miou),
                                        metric=miou,
                                    )
                                print("miou", miou, "best", best_miou)
                    else:
                        with torch.no_grad():
                            model.eval()
                            device = torch.device("cuda")
                            if args.val_amp:
                                with torch.autocast(device_type="cuda", dtype=torch.float16):
                                    if args.mst:
                                        metric = evaluate_msf(
                                            model, val_loader, config, device,
                                            [0.5, 0.75, 1.0, 1.25, 1.5], True, engine,
                                            sliding=args.sliding,
                                        )
                                    else:
                                        metric = evaluate(
                                            model, val_loader, config, device, engine,
                                            sliding=args.sliding,
                                        )
                            else:
                                if args.mst:
                                    metric = evaluate_msf(
                                        model, val_loader, config, device,
                                        [0.5, 0.75, 1.0, 1.25, 1.5], True, engine,
                                        sliding=args.sliding,
                                    )
                                else:
                                    metric = evaluate(
                                        model, val_loader, config, device, engine,
                                        sliding=args.sliding,
                                    )
                            ious, miou = metric.compute_iou()
                            acc, macc = metric.compute_pixel_acc()
                            f1, mf1 = metric.compute_f1()
                        if miou > best_miou:
                            best_miou = miou
                            engine.save_and_link_checkpoint(
                                config.log_dir,
                                config.log_dir,
                                config.log_dir_link,
                                infor="_miou_" + str(miou),
                                metric=miou,
                            )
                        print("miou", miou, "best", best_miou)
                except Exception as e:
                    logger.exception("Exception during evaluation!")
                for h in logger.handlers:
                    h.flush()

                logger.info(
                    f"Epoch {epoch} validation result: mIoU: {miou:.4f},Best mIoU: {best_miou:.4f}, "
                )
                eval_timer.stop()

            eval_count = 0
            for i in range(engine.state.epoch + 1, config.nepochs + 1):
                if is_eval(i, config):
                    eval_count += 1

            left_time = train_timer.mean_time * (config.nepochs - engine.state.epoch) + eval_timer.mean_time * eval_count
            eta = (datetime.datetime.now() + datetime.timedelta(seconds=left_time)).strftime("%Y-%m-%d %H:%M:%S")
            logger.info(
                f"Avg train time: {train_timer.mean_time:.2f}s, avg eval time: {eval_timer.mean_time:.2f}s, left eval count: {eval_count}, ETA: {eta}"
            )

    except Exception:
        logger.exception("Engine initialization or training loop failed with exception")
        raise