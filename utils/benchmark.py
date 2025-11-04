# benchmark.py  —— 需要已安装 thop
# 用法示例：
#   python benchmark.py --config configs.nyudv2.my_cfg --height 480 --width 640 --device cuda:0
import os
import sys
import argparse
import importlib
import torch
import torch.nn as nn
from thop import profile

THIS_DIR = os.path.dirname(os.path.abspath(__file__))           # .../<repo>/utils
REPO_ROOT = os.path.dirname(THIS_DIR)                           # .../<repo>
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from models.builder import EncoderDecoder as segmodel



def build_model_from_config(cfg_module: str, device: torch.device):
    """
    按你工程习惯构建模型，并返回一个 make_inputs(h,w) 函数。
    不改任何训练/推理逻辑，只用于 FLOPs/Params 统计。
    """
    C = getattr(importlib.import_module(cfg_module), "C")

    criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=C.background)
    BatchNorm2d = nn.BatchNorm2d

    model = segmodel(cfg=C, criterion=criterion, norm_layer=BatchNorm2d)
    model.eval().to(device)

    def make_inputs(h, w):
        rgb = torch.ones(1, 3, h, w, device=device)
        dep = torch.ones(1, 1, h, w, device=device)

        # 文本引导开关与 token 数设置，保持与你配置一致（不改逻辑）
        src = getattr(C, "text_source", "both")
        cap_k = getattr(C, "caption_topk", 0)
        cap_max = getattr(C, "max_caption_sentences", 0)
        cap_tokens = cap_k if (isinstance(cap_k, int) and cap_k > 0) else cap_max

        if src == "labels":
            text_tokens = C.num_classes
        elif src == "captions":
            text_tokens = cap_tokens
        else:
            text_tokens = C.num_classes + cap_tokens

        text_dim = getattr(C, "text_feature_dim", 512)
        enable_text = getattr(C, "enable_text_guidance", False)

        if enable_text:
            dummy_text = torch.zeros(1, text_tokens, text_dim, device=device)
            # 你的 forward 支持 (rgb, depth, None, text_features)
            return (rgb, dep, None, dummy_text)
        else:
            # 你的 forward 支持 (rgb, depth)
            return (rgb, dep)

    return C, model, make_inputs


def humanize(num, unit=""):
    if num < 1e3:
        return f"{num:.0f}{unit}"
    if num < 1e6:
        return f"{num/1e3:.2f} K{unit}"
    if num < 1e9:
        return f"{num/1e6:.2f} M{unit}"
    if num < 1e12:
        return f"{num/1e9:.2f} G{unit}"
    return f"{num/1e12:.2f} T{unit}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="如: configs.nyudv2.my_cfg（模块路径）")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    C, model, make_inputs = build_model_from_config(args.config, device)
    inputs = make_inputs(args.height, args.width)

    def _clear_thop_buffers(module: torch.nn.Module):
        """移除 thop 上次统计时留下的 total_ops/total_params 缓冲。"""
        for name in ("total_ops", "total_params"):
            if hasattr(module, name):
                try:
                    delattr(module, name)
                except AttributeError:
                    pass
            if name in getattr(module, "_buffers", {}):
                module._buffers.pop(name, None)

    # 避免重复 profile 同一模型时报 "attribute 'total_ops' already exists"。
    model.apply(_clear_thop_buffers)

    # 使用 thop 统计（thop 返回的是 MACs 与 Params；业内常按 MACs 记为 FLOPs）
    with torch.no_grad():
        macs, params = profile(model, inputs=inputs)

    print("=" * 60)
    print(f"[Config]   {args.config}")
    print(f"[Input]    3x{args.height}x{args.width} + 1x{args.height}x{args.width} (RGB+Depth)")
    print(f"[Backend]  thop")
    print("-" * 60)
    print(f"Params: {humanize(params, ' Params')}  ({params/1e6:.2f} M)")
    print(f"FLOPs : {humanize(macs,   ' FLOPs')}   ({macs/1e9:.2f} G)")
    print("=" * 60)
    print("Note: thop 统计的是乘加次数（MACs）；论文中通常按 FLOPs 报告，口径一致即可。")


if __name__ == "__main__":
    main()
