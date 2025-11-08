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

def sam_flops_counter_hook(module, input, output):
    """
    SemanticAlignmentModule (SAM) FLOPs 计算

    SAM 的主要计算：
    1. Linear 投影 (q_proj, k_proj, v_proj, out_proj) - 由 thop 自动计算
    2. Multi-head Attention:
       - Q·K^T: 2 * B * N * T * Dh * num_heads
       - Softmax: 5 * B * N * num_heads * T (exp + sum + div)
       - Attention·V: 2 * B * N * T * Dh * num_heads
    3. FFN (decoder only):
       - fc1: 2 * B * N * query_dim * (4 * query_dim)
       - fc2: 2 * B * N * (4 * query_dim) * query_dim
    4. LayerNorm (norm1, norm2) - 由自定义钩子计算
    5. Residual/gating - 可忽略（加法）

    注意：这是一个近似计算，实际 FLOPs 会因 Top-K、padding mask 等而变化
    """
    visual_features, text_features = input

    # 获取形状
    if visual_features.dim() == 4:  # (B, H, W, Cv)
        B, H, W, Cv = visual_features.shape
        N = H * W
    else:  # (B, N, Cv)
        B, N, Cv = visual_features.shape

    # 文本 token 数
    if text_features is not None:
        if text_features.dim() == 3:  # (B, T, Ct)
            T = text_features.size(1)
        elif text_features.dim() == 2:  # (T, Ct)
            T = text_features.size(0)
        else:
            T = 1
    else:
        # 如果没有文本特征，FLOPs 几乎为 0（只有残差）
        module.__flops__ += 0
        return

    # 多头参数
    num_heads = getattr(module, 'num_heads', 1)
    Dh = Cv // num_heads

    flops = 0

    # Multi-head Attention:
    # Q·K^T: B * num_heads * N * T * Dh (matmul: 2*Dh FLOPs per output element)
    flops += 2 * B * num_heads * N * T * Dh

    # Softmax over T dimension: ~5 ops per element (exp, sum, div, etc.)
    flops += 5 * B * num_heads * N * T

    # Attention·V: B * num_heads * N * T * Dh
    flops += 2 * B * num_heads * N * T * Dh

    # Top-K selection (if enabled): ignore for simplicity (small overhead)

    # FFN (decoder mode - check if module has FFN)
    # Most SAM instances use FFN in decoder, so count it
    # fc1: B * N * Cv * (4*Cv)
    flops += 2 * B * N * Cv * (4 * Cv)
    # fc2: B * N * (4*Cv) * Cv
    flops += 2 * B * N * (4 * Cv) * Cv

    # Normalize (cosine similarity if enabled): 2*N*Cv for Q, 2*T*Cv for K
    if getattr(module, 'decoder_use_cosine', False) or getattr(module, 'encoder_use_cosine', False):
        flops += 2 * N * Cv  # normalize Q
        flops += 2 * T * Cv  # normalize K

    module.__flops__ += int(flops)

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

try:
    from models.blocks.semantic_alignment import SemanticAlignmentModule
    CUSTOM_OPS[SemanticAlignmentModule] = sam_flops_counter_hook
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
        max_img_labels = getattr(C, "max_image_labels", 0)

        if src == "labels":
            text_tokens = C.num_classes
        elif src == "captions":
            text_tokens = cap_tokens
        elif src == "imglabels":
            # Per-image labels: use max_image_labels if set, otherwise default to 6
            text_tokens = max_img_labels if max_img_labels > 0 else 6
        elif src == "both":
            text_tokens = C.num_classes + cap_tokens
        else:
            # Fallback for unknown text_source
            text_tokens = C.num_classes

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