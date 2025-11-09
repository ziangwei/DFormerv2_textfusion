from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import pathlib
import torch
import argparse
import yaml
import math
import os
import time
from pathlib import Path
from tabulate import tabulate
from torch.utils.data import DataLoader
from torch.nn import functional as F
from utils.engine.logger import get_logger
from utils.metrics_new import Metrics
import numpy as np

@torch.no_grad()
def evaluate(model, dataloader, config, device, engine, save_dir=None, sliding=False, classbank_KD: torch.Tensor | None = None, prompt_mode: str = "single"):
    logger = get_logger(config.log_dir, config.log_file, rank=engine.local_rank)
    logger.info(f"[evaluate] Entered evaluate(), dataloader size={len(dataloader)}")
    for h in logger.handlers:
        h.flush()

    print("Evaluating...")
    model.eval()
    n_classes = config.num_classes
    metrics = Metrics(n_classes, config.background, device)


    for idx, minibatch in enumerate(dataloader):

        logger.info(f"[evaluate] Batch {idx}/{len(dataloader)}, keys={list(minibatch.keys())}")
        for h in logger.handlers: h.flush()

        print_interval = max(1, int(len(dataloader) * 0.5))
        if ((idx + 1) % print_interval == 0 or idx == 0) and (
            (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed)
        ):
            print(f"Validation Iter: {idx + 1} / {len(dataloader)}")

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
        # print(images.shape,labels.shape)

        images = [images.to(device), modal_xs.to(device)]
        labels = labels.to(device)
        if text_feats is not None:
            text_feats = text_feats.to(device).float()
        if sliding:
            preds = slide_inference(
                model,
                images[0],
                images[1],
                text_feats,
                config,
            ).softmax(dim=1)
        else:
            preds = model(images[0], images[1], text_features=text_feats).softmax(dim=1)
        # print(preds.shape,labels.shape)
        B, H, W = labels.shape
        metrics.update(preds, labels)
        # for i in range(B):
        #     metrics.update(preds[i].unsqueeze(0), labels[i].unsqueeze(0))
        # metrics.update(preds, labels)

        # 安全释放eval缓存：在no_grad上下文中清理，不影响训练
        # 每处理完一个batch就清理，防止内存累积导致OOM
        torch.cuda.empty_cache()

        if save_dir is not None:
            palette = [
                [128, 64, 128],
                [244, 35, 232],
                [70, 70, 70],
                [102, 102, 156],
                [190, 153, 153],
                [153, 153, 153],
                [250, 170, 30],
                [220, 220, 0],
                [107, 142, 35],
                [152, 251, 152],
                [70, 130, 180],
                [220, 20, 60],
                [255, 0, 0],
                [0, 0, 142],
                [0, 0, 70],
                [0, 60, 100],
                [0, 80, 100],
                [0, 0, 230],
                [119, 11, 32],
            ]
            palette = np.array(palette, dtype=np.uint8)
            cmap = ListedColormap(palette)
            names = minibatch["fn"][0].replace(".jpg", "").replace(".png", "").replace("datasets/", "")
            save_name = save_dir + "/" + names + "_pred.png"
            pathlib.Path(save_name).parent.mkdir(parents=True, exist_ok=True)
            preds = preds.argmax(dim=1).cpu().squeeze().numpy().astype(np.uint8)
            if config.dataset_name in ["KITTI-360", "EventScape"]:
                preds = palette[preds]
                plt.imsave(save_name, preds)
            elif config.dataset_name in ["NYUDepthv2", "SUNRGBD"]:
                palette = np.load("./utils/nyucmap.npy")
                preds = palette[preds]
                plt.imsave(save_name, preds)
            elif config.dataset_name in ["MFNet"]:
                palette = np.array(
                    [
                        [0, 0, 0],
                        [64, 0, 128],
                        [64, 64, 0],
                        [0, 128, 192],
                        [0, 0, 192],
                        [128, 128, 0],
                        [64, 64, 128],
                        [192, 128, 128],
                        [192, 64, 0],
                    ],
                    dtype=np.uint8,
                )
                preds = palette[preds]
                plt.imsave(save_name, preds)
            else:
                assert 1 == 2

    # ious, miou = metrics.compute_iou()
    # acc, macc = metrics.compute_pixel_acc()
    # f1, mf1 = metrics.compute_f1()
    if engine.distributed:
        all_metrics = [None for _ in range(engine.world_size)]
        # all_predictions = Metrics(n_classes, config.background, device)
        torch.distributed.all_gather_object(all_metrics, metrics)  # list of lists
    else:
        all_metrics = metrics
    return all_metrics


def slide_inference(model, imgs, modal_xs,text_features, config=None):
    """Inference by sliding-window with overlap.

    If h_crop > h_img or w_crop > w_img, the small patch will be used to
    decode without padding.

    Args:
        inputs (tensor): the tensor should have a shape NxCxHxW,
            which contains all images in the batch.
        batch_img_metas (List[dict]): List of image metainfo where each may
            also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
            'ori_shape', and 'pad_shape'.
            For details on the values of these keys see
            `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

    Returns:
        Tensor: The segmentation results, seg_logits from model of each
            input image.
    """


    h_crop, w_crop = config.eval_crop_size

    logger = get_logger(config.log_dir, config.log_file, rank=None)
    logger.info(f"[slide_inference] Entered slide_inference(), h_grids={h_crop}, w_grids={w_crop}")
    for h in logger.handlers: h.flush()

    # new add:
    if h_crop > imgs.shape[-2] or w_crop > imgs.shape[-1]:
        imgs = F.interpolate(imgs, size=(h_crop, w_crop), mode="bilinear", align_corners=True)
        modal_xs = F.interpolate(modal_xs, size=(h_crop, w_crop), mode="bilinear", align_corners=True)

    h_stride, w_stride = [
        int(config.eval_stride_rate * config.eval_crop_size[0]),
        int(config.eval_stride_rate * config.eval_crop_size[1]),
    ]
    batch_size, _, h_img, w_img = imgs.shape
    assert imgs.shape[-2:] == modal_xs.shape[-2:]
    out_channels = config.num_classes
    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
    preds = imgs.new_zeros((batch_size, out_channels, h_img, w_img))
    count_mat = imgs.new_zeros((batch_size, 1, h_img, w_img))
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)
            crop_img = imgs[:, :, y1:y2, x1:x2]
            crop_modal_xs = modal_xs[:, :, y1:y2, x1:x2]
            # the output of encode_decode is seg logits tensor map
            # with shape [N, C, H, W]
            crop_seg_logit = model(crop_img, crop_modal_xs, text_features=text_features)
            preds += F.pad(
                crop_seg_logit,
                (int(x1), int(preds.shape[3] - x2), int(y1), int(preds.shape[2] - y2)),
            )

            count_mat[:, :, y1:y2, x1:x2] += 1
    assert (count_mat == 0).sum() == 0
    seg_logits = preds / count_mat

    return seg_logits


@torch.no_grad()
def evaluate_msf(
    model,
    dataloader,
    config,
    device,
    scales,
    flip,
    engine,
    save_dir=None,
    sliding=False,
):

    model.eval()

    n_classes = config.num_classes
    metrics = Metrics(n_classes, config.background, device)

    # 防止 dataloader 为空导致除零错误
    dataloader_len = len(dataloader)
    if dataloader_len == 0:
        if (engine.distributed and engine.local_rank == 0) or not engine.distributed:
            print("⚠️  WARNING: Dataloader is empty! No data to evaluate.")
        return {
            'mIoU': 0.0,
            'Acc': 0.0,
            'per_class_iou': [0.0] * n_classes,
            'per_class_acc': [0.0] * n_classes,
        }

    # 设置进度打印间隔（避免除零）
    print_interval = max(1, int(dataloader_len * 0.5))

    for idx, minibatch in enumerate(dataloader):
        if ((idx + 1) % print_interval == 0 or idx == 0) and (
            (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed)
        ):
            print(f"Validation Iter: {idx + 1} / {dataloader_len}")

        images = minibatch["data"]
        labels = minibatch["label"]
        modal_xs = minibatch["modal_x"]
        text_feats = minibatch.get("text_features")
        # print(images.shape,labels.shape)
        images = [images.to(device), modal_xs.to(device)]
        labels = labels.to(device)
        if text_feats is not None:
            text_feats = text_feats.to(device).float()
        B, H, W = labels.shape
        scaled_logits = torch.zeros(B, n_classes, H, W).to(device)

        for scale in scales:
            new_H, new_W = int(scale * H), int(scale * W)
            new_H, new_W = (
                int(math.ceil(new_H / 32)) * 32,
                int(math.ceil(new_W / 32)) * 32,
            )
            scaled_images = [
                F.interpolate(img, size=(new_H, new_W), mode="bilinear", align_corners=True) for img in images
            ]
            scaled_images = [scaled_img.to(device) for scaled_img in scaled_images]

            if sliding:
                logits = slide_inference(
                    model, scaled_images[0], scaled_images[1], text_feats, config=config
                )
            else:
                logits = model(scaled_images[0], scaled_images[1], text_features=text_feats)

            logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=True)
            scaled_logits += logits.softmax(dim=1)

            if flip:
                scaled_images = [torch.flip(scaled_img, dims=(3,)) for scaled_img in scaled_images]
                if sliding:
                    logits = slide_inference(
                        model,
                        scaled_images[0],
                        scaled_images[1],
                        text_feats,
                        config,
                    )
                else:
                    logits = model(scaled_images[0], scaled_images[1], text_features=text_feats)
                logits = torch.flip(logits, dims=(3,))
                logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=True)
                scaled_logits += logits.softmax(dim=1)

        if save_dir is not None:
            palette = [
                [128, 64, 128],
                [244, 35, 232],
                [70, 70, 70],
                [102, 102, 156],
                [190, 153, 153],
                [153, 153, 153],
                [250, 170, 30],
                [220, 220, 0],
                [107, 142, 35],
                [152, 251, 152],
                [70, 130, 180],
                [220, 20, 60],
                [255, 0, 0],
                [0, 0, 142],
                [0, 0, 70],
                [0, 60, 100],
                [0, 80, 100],
                [0, 0, 230],
                [119, 11, 32],
            ]
            palette = np.array(palette, dtype=np.uint8)
            cmap = ListedColormap(palette)
            names = minibatch["fn"][0].replace(".jpg", "").replace(".png", "").replace("datasets/", "")
            save_name = save_dir + "/" + names + "_pred.png"
            pathlib.Path(save_name).parent.mkdir(parents=True, exist_ok=True)
            preds = scaled_logits.argmax(dim=1).cpu().squeeze().numpy().astype(np.uint8)
            if config.dataset_name in ["KITTI-360", "EventScape"]:
                preds = palette[preds]
                plt.imsave(save_name, preds)
            elif config.dataset_name in ["NYUDepthv2", "SUNRGBD"]:
                palette = np.load("./utils/nyucmap.npy")
                preds = palette[preds]
                plt.imsave(save_name, preds)
            elif config.dataset_name in ["MFNet"]:
                palette = np.array(
                    [
                        [0, 0, 0],
                        [64, 0, 128],
                        [64, 64, 0],
                        [0, 128, 192],
                        [0, 0, 192],
                        [128, 128, 0],
                        [64, 64, 128],
                        [192, 128, 128],
                        [192, 64, 0],
                    ],
                    dtype=np.uint8,
                )
                preds = palette[preds]
                plt.imsave(save_name, preds)
            else:
                assert 1 == 2

        metrics.update(scaled_logits, labels)

        # 安全释放eval缓存：在no_grad上下文中清理，不影响训练
        # MSF模式下每个图像内存消耗更大（多尺度+flip），必须及时清理
        torch.cuda.empty_cache()

    # ious, miou = metrics.compute_iou()
    # acc, macc = metrics.compute_pixel_acc()
    # f1, mf1 = metrics.compute_f1()

    if engine.distributed:
        all_metrics = [None for _ in range(engine.world_size)]
        # 每个进程把自己的 metrics 发送/接收到 all_metrics 列表中
        torch.distributed.all_gather_object(all_metrics, metrics)
    else:
        all_metrics = metrics
    return all_metrics

def main(cfg):
    device = torch.device(cfg["DEVICE"])

    eval_cfg = cfg["EVAL"]
    transform = get_val_augmentation(eval_cfg["IMAGE_SIZE"])
    # cases = ['cloud', 'fog', 'night', 'rain', 'sun']
    # cases = ['motionblur', 'overexposure', 'underexposure', 'lidarjitter', 'eventlowres']
    cases = [None]  # all

    model_path = Path(eval_cfg["MODEL_PATH"])
    if not model_path.exists():
        raise FileNotFoundError
    # print(f"Evaluating {model_path}...")

    exp_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    eval_path = os.path.join(os.path.dirname(eval_cfg["MODEL_PATH"]), "eval_{}.txt".format(exp_time))

    for case in cases:
        dataset = eval(cfg["DATASET"]["NAME"])(
            cfg["DATASET"]["ROOT"], "val", transform, cfg["DATASET"]["MODALS"], case
        )
        # --- test set
        # dataset = eval(cfg['DATASET']['NAME'])(cfg['DATASET']['ROOT'], 'test', transform, cfg['DATASET']['MODALS'], case)

        model = eval(cfg["MODEL"]["NAME"])(cfg["MODEL"]["BACKBONE"], dataset.n_classes, cfg["DATASET"]["MODALS"])
        msg = model.load_state_dict(torch.load(str(model_path), map_location="cpu"))
        # print(msg)
        model = model.to(device)
        sampler_val = None
        dataloader = DataLoader(
            dataset,
            batch_size=eval_cfg["BATCH_SIZE"],
            num_workers=eval_cfg["BATCH_SIZE"],
            pin_memory=False,
            sampler=sampler_val,
        )
        if True:
            if eval_cfg["MSF"]["ENABLE"]:
                acc, macc, f1, mf1, ious, miou = evaluate_msf(
                    model,
                    dataloader,
                    device,
                    eval_cfg["MSF"]["SCALES"],
                    eval_cfg["MSF"]["FLIP"],
                )
            else:
                acc, macc, f1, mf1, ious, miou = evaluate(model, dataloader, device)

            table = {
                "Class": list(dataset.CLASSES) + ["Mean"],
                "IoU": ious + [miou],
                "F1": f1 + [mf1],
                "Acc": acc + [macc],
            }
            print("mIoU : {}".format(miou))
            print("Results saved in {}".format(eval_cfg["MODEL_PATH"]))

        with open(eval_path, "a+") as f:
            f.writelines(eval_cfg["MODEL_PATH"])
            f.write("\n============== Eval on {} {} images =================\n".format(case, len(dataset)))
            f.write("\n")
            print(tabulate(table, headers="keys"), file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="configs/DELIVER.yaml")
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    setup_cudnn()
    main(cfg)