# benchmark.py  —— 需要已安装 thop
# 用法示例：
#   python utils/benchmark.py --config local_configs.NYUDepthv2.DFormerv2_S --height 480 --width 640 --device cpu
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

def safe_layernorm_flops_counter_hook(module, input, output):
    """LayerNorm: 2 * normalized_shape for mean/var, 2 * normalized_shape for scale/shift"""
    input = input[0]
    batch_flops = input.numel()  # Mean and variance
    batch_flops += input.numel()  # Normalize
    batch_flops += input.numel()  # Scale and shift
    # 兼容thop的两种计数方式
    if not hasattr(module, '__flops__'):
        module.__flops__ = 0
    module.__flops__ += int(batch_flops)
    # total_ops在register_buffer中已初始化
    if hasattr(module, 'total_ops'):
        module.total_ops += int(batch_flops)

def safe_gelu_flops_counter_hook(module, input, output):
    """GELU: approximation requires ~8 ops per element"""
    input = input[0]
    batch_flops = int(input.numel() * 8)
    if not hasattr(module, '__flops__'):
        module.__flops__ = 0
    module.__flops__ += batch_flops
    if hasattr(module, 'total_ops'):
        module.total_ops += batch_flops

def safe_dropout_flops_counter_hook(module, input, output):
    """Dropout: no computation during eval, but count mask generation during training"""
    batch_flops = 0
    if module.training:
        input = input[0]
        batch_flops = int(input.numel())
    if not hasattr(module, '__flops__'):
        module.__flops__ = 0
    module.__flops__ += batch_flops
    if hasattr(module, 'total_ops'):
        module.total_ops += batch_flops

def safe_syncbn_flops_counter_hook(module, input, output):
    """SyncBatchNorm: similar to BatchNorm"""
    input = input[0]
    batch_flops = input.numel() * 2  # Mean and variance
    batch_flops += input.numel() * 2  # Normalize + affine
    batch_flops = int(batch_flops)
    if not hasattr(module, '__flops__'):
        module.__flops__ = 0
    module.__flops__ += batch_flops
    if hasattr(module, 'total_ops'):
        module.total_ops += batch_flops

def safe_batchnorm_flops_counter_hook(module, input, output):
    """BatchNorm2d: similar to SyncBatchNorm"""
    input = input[0]
    batch_flops = input.numel() * 2  # Mean and variance
    batch_flops += input.numel() * 2  # Normalize + affine
    batch_flops = int(batch_flops)
    if not hasattr(module, '__flops__'):
        module.__flops__ = 0
    module.__flops__ += batch_flops
    if hasattr(module, 'total_ops'):
        module.total_ops += batch_flops

def safe_identity_flops_counter_hook(module, input, output):
    """Identity/Pass-through layers: zero ops"""
    if not hasattr(module, '__flops__'):
        module.__flops__ = 0
    # total_ops already initialized to 0 in register_buffer

# Build custom ops dict with safe hooks
CUSTOM_OPS = {
    nn.LayerNorm: safe_layernorm_flops_counter_hook,
    nn.GELU: safe_gelu_flops_counter_hook,
    nn.Dropout: safe_dropout_flops_counter_hook,
    nn.Dropout2d: safe_dropout_flops_counter_hook,
    nn.SyncBatchNorm: safe_syncbn_flops_counter_hook,
    nn.BatchNorm2d: safe_batchnorm_flops_counter_hook,
    nn.Identity: safe_identity_flops_counter_hook,
}

# Try to add custom layers from your codebase
try:
    from models.encoders.DFormerv2 import LayerNorm2d
    CUSTOM_OPS[LayerNorm2d] = safe_layernorm_flops_counter_hook
except ImportError:
    pass


def build_model_from_config(cfg_module: str, device: torch.device, load_pretrained: bool = False,
                           override_text_tokens: int = None, use_avg_tokens: bool = False):
    """
    按你工程习惯构建模型，并返回一个 make_inputs(h,w) 函数。
    不改任何训练/推理逻辑，只用于 FLOPs/Params 统计。

    Args:
        cfg_module: 配置模块路径
        device: 设备
        load_pretrained: 是否加载预训练模型（默认False，仅统计参数时不需要）
        override_text_tokens: 覆盖文本token数（用于测试不同配置）
        use_avg_tokens: 使用平均token数而非最大值（假设75%有效）
    """
    C = getattr(importlib.import_module(cfg_module), "C")

    # 暂时保存原始预训练路径
    original_pretrained = getattr(C, 'pretrained_model', None)

    # 如果不加载预训练，临时设置为None
    if not load_pretrained:
        C.pretrained_model = None

    criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=C.background)
    BatchNorm2d = nn.BatchNorm2d

    model = segmodel(cfg=C, criterion=criterion, norm_layer=BatchNorm2d)

    # 恢复原始配置
    if not load_pretrained:
        C.pretrained_model = original_pretrained

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

        # 支持覆盖token数或使用平均值
        if override_text_tokens is not None:
            text_tokens = override_text_tokens
        elif use_avg_tokens and src == "imglabels":
            # 假设平均有效token是max的75%（例如max=8，平均≈6）
            text_tokens = int(text_tokens * 0.75)
            text_tokens = max(1, text_tokens)  # 至少1个

        text_dim = getattr(C, "text_feature_dim", 512)
        enable_text = getattr(C, "enable_text_guidance", False)

        if enable_text:
            dummy_text = torch.zeros(1, text_tokens, text_dim, device=device)
            # 你的 forward 支持 (rgb, depth, None, text_features)
            return (rgb, dep, None, dummy_text), text_tokens  # 同时返回实际使用的token数
        else:
            # 你的 forward 支持 (rgb, depth)
            return (rgb, dep), 0

    return C, model, make_inputs


def humanize(num, unit=""):
    """将数字转换为人类可读格式"""
    if num < 1e3:
        return f"{num:.0f}{unit}"
    if num < 1e6:
        return f"{num/1e3:.2f} K{unit}"
    if num < 1e9:
        return f"{num/1e6:.2f} M{unit}"
    if num < 1e12:
        return f"{num/1e9:.2f} G{unit}"
    return f"{num/1e12:.2f} T{unit}"


def count_module_params(module, name="Module"):
    """统计单个模块的参数量"""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def analyze_sam_components(model):
    """分析模型中所有SAM组件的参数量（修复重复计数bug）

    Bug修复：
    - 之前使用named_modules()遍历所有模块（包括父子），并对每个调用parameters()
    - 这导致参数被重复计数（父模块包含子模块参数，子模块又被单独统计）
    - 修复：只统计SemanticAlignmentModule类型的叶子模块
    """
    sam_stats = {
        'encoder_sam': 0,
        'decoder_sam': 0,
        'total_sam': 0,
        'encoder_sam_modules': [],
        'decoder_sam_modules': []
    }

    # 修复：只统计SemanticAlignmentModule类型，避免重复计数
    try:
        from models.blocks.semantic_alignment import SemanticAlignmentModule

        for name, module in model.named_modules():
            # 只统计SemanticAlignmentModule实例（叶子模块）
            if isinstance(module, SemanticAlignmentModule):
                params = sum(p.numel() for p in module.parameters())
                if params > 0:
                    # 区分encoder和decoder侧的SAM
                    if 'backbone' in name or 'encoder' in name:
                        sam_stats['encoder_sam'] += params
                        sam_stats['encoder_sam_modules'].append((name, params))
                    elif 'decode_head' in name or 'decoder' in name:
                        sam_stats['decoder_sam'] += params
                        sam_stats['decoder_sam_modules'].append((name, params))
    except ImportError:
        # 降级方案：使用named_parameters避免重复
        # 统计包含'sam'的参数名
        counted_params = set()
        for name, param in model.named_parameters():
            if 'sam' in name.lower() and name not in counted_params:
                counted_params.add(name)
                # 区分encoder和decoder
                if 'backbone' in name or 'encoder' in name:
                    sam_stats['encoder_sam'] += param.numel()
                elif 'decode_head' in name or 'decoder' in name:
                    sam_stats['decoder_sam'] += param.numel()

    sam_stats['total_sam'] = sam_stats['encoder_sam'] + sam_stats['decoder_sam']
    return sam_stats


def analyze_hierarchical_params(model):
    """分层统计模型各组件参数"""
    stats = {}

    # Backbone
    if hasattr(model, 'backbone'):
        stats['backbone'], _ = count_module_params(model.backbone)

    # Decoder
    if hasattr(model, 'decode_head'):
        stats['decode_head'], _ = count_module_params(model.decode_head)

    # Auxiliary head
    if hasattr(model, 'aux_head') and model.aux_head is not None:
        stats['aux_head'], _ = count_module_params(model.aux_head)
    else:
        stats['aux_head'] = 0

    stats['total'] = sum(p.numel() for p in model.parameters())

    return stats


def print_detailed_stats(model, config_name, sam_stats, hierarchical_stats, total_params, total_macs=None):
    """打印详细的模型统计信息"""
    print("\n" + "=" * 80)
    print("模型参数详细统计")
    print("=" * 80)

    # 1. 基本信息
    print(f"\n配置: {config_name}")
    print(f"总参数: {humanize(total_params)} ({total_params/1e6:.2f}M)")
    if total_macs is not None:
        print(f"总FLOPs: {humanize(total_macs * 2)} ({total_macs * 2 / 1e9:.2f}G)")
        print(f"总MACs:  {humanize(total_macs)} ({total_macs / 1e9:.2f}G)")

    # 2. 分层统计
    print("\n" + "-" * 80)
    print("分层参数统计:")
    print("-" * 80)
    for component, params in hierarchical_stats.items():
        if component != 'total' and params > 0:
            percentage = (params / total_params) * 100
            print(f"  {component:20s}: {humanize(params):>10s} ({params/1e6:6.2f}M)  {percentage:5.1f}%")

    # 3. SAM组件统计
    if sam_stats['total_sam'] > 0:
        print("\n" + "-" * 80)
        print("SAM组件统计:")
        print("-" * 80)
        print(f"  Encoder侧SAM: {humanize(sam_stats['encoder_sam']):>10s} ({sam_stats['encoder_sam']/1e6:6.2f}M)")
        if sam_stats['encoder_sam_modules']:
            for name, params in sam_stats['encoder_sam_modules'][:5]:  # 只显示前5个
                print(f"    - {name[:50]:50s}: {params/1e6:6.2f}M")
            if len(sam_stats['encoder_sam_modules']) > 5:
                print(f"    ... 及其他 {len(sam_stats['encoder_sam_modules']) - 5} 个模块")

        print(f"\n  Decoder侧SAM: {humanize(sam_stats['decoder_sam']):>10s} ({sam_stats['decoder_sam']/1e6:6.2f}M)")
        if sam_stats['decoder_sam_modules']:
            for name, params in sam_stats['decoder_sam_modules'][:5]:
                print(f"    - {name[:50]:50s}: {params/1e6:6.2f}M")
            if len(sam_stats['decoder_sam_modules']) > 5:
                print(f"    ... 及其他 {len(sam_stats['decoder_sam_modules']) - 5} 个模块")

        sam_percentage = (sam_stats['total_sam'] / total_params) * 100
        print(f"\n  SAM总参数: {humanize(sam_stats['total_sam']):>10s} ({sam_stats['total_sam']/1e6:6.2f}M)  {sam_percentage:5.1f}%")

    print("\n" + "=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark模型参数和FLOPs统计工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基础统计
  python utils/benchmark.py --config local_configs.NYUDepthv2.DFormerv2_S

  # 详细统计（包含SAM分析）
  python utils/benchmark.py --config local_configs.NYUDepthv2.DFormerv2_B --detailed

  # 跳过FLOPs统计（更快）
  python utils/benchmark.py --config local_configs.NYUDepthv2.DFormerv2_L --params-only

  # 加载预训练模型
  python utils/benchmark.py --config local_configs.NYUDepthv2.DFormerv2_S --load-pretrained
        """
    )
    parser.add_argument("--config", required=True, help="配置模块路径，如: local_configs.NYUDepthv2.DFormerv2_S")
    parser.add_argument("--height", type=int, default=480, help="输入图像高度")
    parser.add_argument("--width", type=int, default=640, help="输入图像宽度")
    parser.add_argument("--device", type=str, default="cpu", help="设备: cpu, cuda:0, etc.")
    parser.add_argument("--load-pretrained", action="store_true", help="是否加载预训练模型权重")
    parser.add_argument("--detailed", action="store_true", help="显示详细的参数分析（包含SAM组件统计）")
    parser.add_argument("--params-only", action="store_true", help="仅统计参数，跳过FLOPs计算（更快）")
    parser.add_argument("--text-tokens", type=int, default=None, help="覆盖文本token数量（用于测试不同token数的FLOPs，默认使用配置中的max_image_labels）")
    parser.add_argument("--avg-tokens", action="store_true", help="使用实际平均token数而非最大值（imglabels场景，假设平均75%%有效）")
    args = parser.parse_args()

    device = torch.device(args.device if (torch.cuda.is_available() and 'cuda' in args.device) else "cpu")

    print("=" * 80)
    print("模型Benchmark工具")
    print("=" * 80)

    # 构建模型
    C, model, make_inputs = build_model_from_config(
        args.config, device, args.load_pretrained,
        override_text_tokens=args.text_tokens,
        use_avg_tokens=args.avg_tokens
    )
    inputs_tuple = make_inputs(args.height, args.width)
    inputs, actual_text_tokens = inputs_tuple if isinstance(inputs_tuple, tuple) else (inputs_tuple, 0)

    # 基本信息
    print(f"\n配置文件: {args.config}")
    print(f"Backbone:  {C.backbone}")
    print(f"Decoder:   {getattr(C, 'decoder', 'N/A')}")
    print(f"输入尺寸: RGB: 3x{args.height}x{args.width}, Depth: 1x{args.height}x{args.width}")

    if getattr(C, "enable_text_guidance", False):
        src = getattr(C, "text_source", "both")
        max_img_labels = getattr(C, "max_image_labels", 0) if src == "imglabels" else 0

        # 显示token信息
        token_info = f"source={src}, tokens={actual_text_tokens}"
        if args.text_tokens is not None:
            token_info += f" (override={args.text_tokens})"
        elif args.avg_tokens and src == "imglabels":
            token_info += f" (avg, max={max_img_labels})"
        elif src == "imglabels":
            token_info += f" (max={max_img_labels})"

        print(f"文本引导: Enabled ({token_info})")
    else:
        print(f"文本引导: Disabled")

    # SAM配置
    if hasattr(C, 'sam_enc_stages'):
        print(f"Encoder SAM stages: {getattr(C, 'sam_enc_stages', [])} (superpower={getattr(C, 'superpower', False)})")
    if hasattr(C, 'sam_dec_stages'):
        print(f"Decoder SAM stages: {getattr(C, 'sam_dec_stages', [])}")

    print(f"设备: {device}")
    print("-" * 80)

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n参数统计:")
    print(f"  总参数:      {humanize(total_params)} ({total_params/1e6:.2f}M)")
    print(f"  可训练参数:  {humanize(trainable_params)} ({trainable_params/1e6:.2f}M)")

    # FLOPs统计
    macs = None
    if not args.params_only:
        def _clear_and_init_thop_buffers(module: torch.nn.Module):
            """清除并初始化thop需要的buffers，确保所有层都有total_ops和total_params"""
            # 先清除旧的
            for name in ("total_ops", "total_params"):
                if hasattr(module, name):
                    try:
                        delattr(module, name)
                    except AttributeError:
                        pass
                if name in getattr(module, "_buffers", {}):
                    module._buffers.pop(name, None)

            # 初始化为0，防止thop访问时AttributeError
            # 注意：必须是tensor类型，因为thop会调用.item()方法
            if not isinstance(module, nn.ModuleList) and not isinstance(module, nn.ModuleDict):
                module.register_buffer("total_ops", torch.zeros(1))
                module.register_buffer("total_params", torch.zeros(1))

        model.apply(_clear_and_init_thop_buffers)

        try:
            print("\n正在计算FLOPs...")
            with torch.no_grad():
                macs, params_thop = profile(model, inputs=inputs, custom_ops=CUSTOM_OPS, verbose=False)

            macs_str, _ = clever_format([macs, params_thop], "%.3f")
            print(f"  MACs:  {macs_str} ({macs/1e9:.2f}G)")
            print(f"  FLOPs: {humanize(macs * 2)} ({macs * 2 / 1e9:.2f}G)")
            print(f"  注: FLOPs ≈ 2 × MACs")

        except Exception as e:
            print(f"\nWARNING: FLOPs计算失败: {e}")
            print("继续进行参数统计...")
            if args.device != 'cpu':
                print("提示: 尝试使用 --device cpu")

    # 详细分析
    if args.detailed:
        print("\n正在进行详细分析...")
        sam_stats = analyze_sam_components(model)
        hierarchical_stats = analyze_hierarchical_params(model)
        print_detailed_stats(model, args.config, sam_stats, hierarchical_stats, total_params, macs)
    else:
        print("\n提示: 使用 --detailed 参数查看详细的组件和SAM统计信息")

    print("=" * 80)


if __name__ == "__main__":
    main()