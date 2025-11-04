# benchmark.py  —— 需要已安装 thop
# 用法示例：
#   python benchmark.py --config configs.nyudv2.my_cfg --height 480 --width 640 --device cuda:0
import os
import sys
import argparse
import importlib
import torch
import torch.nn as nn
from thop import profile, clever_format
from thop.vision.basic_hooks import zero_ops

THIS_DIR = os.path.dirname(os.path.abspath(__file__))           # .../<repo>/utils
REPO_ROOT = os.path.dirname(THIS_DIR)                           # .../<repo>
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from models.builder import EncoderDecoder as segmodel


# ========== Custom Ops for thop ==========
# Register handlers for layers that thop doesn't support by default

def layernorm_flops_counter_hook(module, input, output):
    """LayerNorm: 2 * normalized_shape for mean/var, 2 * normalized_shape for scale/shift"""
    input = input[0]
    batch_flops = input.numel()  # Mean and variance
    batch_flops += input.numel()  # Normalize
    batch_flops += input.numel()  # Scale and shift
    module.__flops__ += int(batch_flops)

def gelu_flops_counter_hook(module, input, output):
    """GELU: approximation requires ~8 ops per element"""
    input = input[0]
    module.__flops__ += int(input.numel() * 8)

def dropout_flops_counter_hook(module, input, output):
    """Dropout: no computation during eval, but count mask generation during training"""
    if module.training:
        input = input[0]
        module.__flops__ += int(input.numel())
    else:
        module.__flops__ += 0

def syncbn_flops_counter_hook(module, input, output):
    """SyncBatchNorm: similar to BatchNorm"""
    input = input[0]
    batch_flops = input.numel() * 2  # Mean and variance
    batch_flops += input.numel() * 2  # Normalize + affine
    module.__flops__ += int(batch_flops)

def identity_flops_counter_hook(module, input, output):
    """Identity/Pass-through layers: zero ops"""
    module.__flops__ += 0

# Build custom ops dict
CUSTOM_OPS = {
    nn.LayerNorm: layernorm_flops_counter_hook,
    nn.GELU: gelu_flops_counter_hook,
    nn.Dropout: dropout_flops_counter_hook,
    nn.Dropout2d: dropout_flops_counter_hook,
    nn.SyncBatchNorm: syncbn_flops_counter_hook,
    nn.Identity: identity_flops_counter_hook,
}

# Try to add custom layers from your codebase
try:
    from models.encoders.DFormerv2 import LayerNorm2d
    CUSTOM_OPS[LayerNorm2d] = layernorm_flops_counter_hook
except ImportError:
    pass



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

    print("=" * 60)
    print(f"[Config]   {args.config}")
    print(f"[Backbone] {C.backbone}")
    print(f"[Input]    RGB: 3x{args.height}x{args.width}, Depth: 1x{args.height}x{args.width}")
    if getattr(C, "enable_text_guidance", False):
        src = getattr(C, "text_source", "both")
        print(f"[Text]     Enabled (source={src})")
    print(f"[Backend]  thop with custom ops")
    print("-" * 60)

    # 使用 thop 统计，传入 custom_ops 避免 AttributeError
    try:
        with torch.no_grad():
            macs, params = profile(model, inputs=inputs, custom_ops=CUSTOM_OPS, verbose=False)

        # Use clever_format for better readability
        macs_str, params_str = clever_format([macs, params], "%.3f")

        print(f"Parameters: {params_str}  ({params/1e6:.2f} M)")
        print(f"FLOPs (MACs): {macs_str}  ({macs/1e9:.2f} G)")
        print("=" * 60)
        print("Note: FLOPs reported as MACs (Multiply-Accumulate operations)")
        print("      For paper reporting, typically FLOPs ≈ 2 × MACs")

    except Exception as e:
        print(f"ERROR: Failed to profile model: {e}")
        print(f"Try running with --device cpu if CUDA errors occur")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
