#!/usr/bin/env python3
"""
Compact Benchmark: Visual vs Text Component Analysis
精简的参数和FLOPs统计工具，区分视觉部分和文本部分
"""
# 设置无头模式，避免 OpenGL 依赖
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '0'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import sys
import argparse
import importlib
import torch
import torch.nn as nn
from thop import profile

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.builder import EncoderDecoder as segmodel


def humanize(num):
    """Convert number to human-readable format"""
    if num < 1e6:
        return f"{num/1e3:.2f}K"
    if num < 1e9:
        return f"{num/1e6:.2f}M"
    return f"{num/1e9:.2f}G"


def is_text_related(name):
    """Check if a parameter/module is text-related"""
    text_keywords = [
        'text_encoder', 'text_proj', 'text_embed',
        'sam', 'semantic_alignment',  # SAM modules are text-related
        'caption', 'label_embed'
    ]
    name_lower = name.lower()
    return any(keyword in name_lower for keyword in text_keywords)


def analyze_params(model):
    """Separate visual and text parameters"""
    visual_params = 0
    text_params = 0

    for name, param in model.named_parameters():
        if is_text_related(name):
            text_params += param.numel()
        else:
            visual_params += param.numel()

    total = visual_params + text_params
    return {
        'visual': visual_params,
        'text': text_params,
        'total': total,
        'visual_pct': 100 * visual_params / total if total > 0 else 0,
        'text_pct': 100 * text_params / total if total > 0 else 0,
    }


def count_sam_params(model):
    """Count SAM module parameters separately"""
    encoder_sam = 0
    decoder_sam = 0

    try:
        from models.blocks.semantic_alignment import SemanticAlignmentModule
        for name, module in model.named_modules():
            if isinstance(module, SemanticAlignmentModule):
                params = sum(p.numel() for p in module.parameters())
                if 'backbone' in name or 'encoder' in name:
                    encoder_sam += params
                elif 'decode_head' in name or 'decoder' in name:
                    decoder_sam += params
    except ImportError:
        pass

    return encoder_sam, decoder_sam


def analyze_flops_simple(model, inputs, device, param_stats):
    """
    Simplified FLOPs analysis - just compute total and estimate breakdown
    """
    results = {}

    # Custom ops for common layers
    custom_ops = {
        nn.LayerNorm: lambda m, x, y: x[0].numel() * 4,
        nn.GELU: lambda m, x, y: x[0].numel() * 8,
        nn.BatchNorm2d: lambda m, x, y: x[0].numel() * 4,
        nn.SyncBatchNorm: lambda m, x, y: x[0].numel() * 4,
        nn.Dropout: lambda m, x, y: 0,
        nn.Dropout2d: lambda m, x, y: 0,
        nn.Identity: lambda m, x, y: 0,
    }

    try:
        from models.encoders.DFormerv2 import LayerNorm2d
        custom_ops[LayerNorm2d] = lambda m, x, y: x[0].numel() * 4
    except ImportError:
        pass

    model.eval()

    # Compute total FLOPs
    try:
        with torch.no_grad():
            macs, _ = profile(model, inputs=inputs, custom_ops=custom_ops, verbose=False)
        results['total'] = macs * 2  # MACs to FLOPs

        # Estimate visual vs text FLOPs based on parameter ratio
        # This is approximate but reasonable
        text_ratio = param_stats['text'] / param_stats['total'] if param_stats['total'] > 0 else 0
        results['text_estimated'] = results['total'] * text_ratio
        results['visual_estimated'] = results['total'] * (1 - text_ratio)

    except Exception as e:
        results['total'] = 0
        results['text_estimated'] = 0
        results['visual_estimated'] = 0
        print(f"Warning: Failed to profile model: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Compact benchmark: Visual vs Text analysis")
    parser.add_argument("--config", required=True, help="Config module path (e.g., local_configs.NYUDepthv2.DFormerv2_S)")
    parser.add_argument("--height", type=int, default=480, help="Input height")
    parser.add_argument("--width", type=int, default=640, help="Input width")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu, cuda:0, etc.")
    parser.add_argument("--skip-flops", action="store_true", help="Skip FLOPs calculation (faster)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else "cpu")

    # Load config and build model
    C = getattr(importlib.import_module(args.config), "C")
    C.pretrained_model = None  # Don't load pretrained for benchmark

    criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=C.background)
    model = segmodel(cfg=C, criterion=criterion, norm_layer=nn.BatchNorm2d)
    model.eval().to(device)

    # Prepare inputs
    rgb = torch.randn(1, 3, args.height, args.width, device=device)
    depth = torch.randn(1, 1, args.height, args.width, device=device)

    enable_text = getattr(C, "enable_text_guidance", False)
    text_tokens = 0

    if enable_text:
        # Determine text token count
        src = getattr(C, "text_source", "both")
        if src == "labels":
            text_tokens = C.num_classes
        elif src == "captions":
            text_tokens = getattr(C, "max_caption_sentences", 0)
        elif src == "imglabels":
            text_tokens = getattr(C, "max_image_labels", 8)
        elif src == "both":
            text_tokens = C.num_classes + getattr(C, "max_caption_sentences", 0)

        text_dim = getattr(C, "text_feature_dim", 512)
        text_features = torch.randn(1, text_tokens, text_dim, device=device)
        inputs = (rgb, depth, None, text_features)
    else:
        inputs = (rgb, depth)

    # ========== Print Results ==========
    print("\n" + "=" * 70)
    print("COMPACT BENCHMARK: Visual vs Text Component Analysis")
    print("=" * 70)
    print(f"Config:       {args.config}")
    print(f"Input size:   {args.height}x{args.width}")
    print(f"Text guidance: {'Enabled' if enable_text else 'Disabled'}")
    if enable_text:
        print(f"Text tokens:  {text_tokens} (source: {getattr(C, 'text_source', 'N/A')})")
    print(f"Device:       {device}")
    print("-" * 70)

    # Analyze parameters
    param_stats = analyze_params(model)
    encoder_sam, decoder_sam = count_sam_params(model)
    total_sam = encoder_sam + decoder_sam

    print("\n📊 PARAMETER ANALYSIS:")
    print("-" * 70)
    print(f"{'Component':<25} {'Parameters':>15} {'Percentage':>12}")
    print("-" * 70)
    print(f"{'Visual (Pure)':<25} {humanize(param_stats['visual']):>15} {param_stats['visual_pct']:>11.1f}%")
    print(f"{'Text-related (incl SAM)':<25} {humanize(param_stats['text']):>15} {param_stats['text_pct']:>11.1f}%")
    print("-" * 70)
    print(f"{'Total':<25} {humanize(param_stats['total']):>15} {'100.0%':>12}")
    print("-" * 70)

    # SAM breakdown
    if total_sam > 0:
        sam_pct = 100 * total_sam / param_stats['total']
        print(f"\n  SAM Module Breakdown:")
        print(f"  {'- Encoder SAM:':<23} {humanize(encoder_sam):>15} ({100*encoder_sam/param_stats['total']:>5.1f}%)")
        print(f"  {'- Decoder SAM:':<23} {humanize(decoder_sam):>15} ({100*decoder_sam/param_stats['total']:>5.1f}%)")
        print(f"  {'- SAM Total:':<23} {humanize(total_sam):>15} ({sam_pct:>5.1f}%)")

    # FLOPs analysis
    if not args.skip_flops:
        print("\n⚡ FLOPS ANALYSIS:")
        print("-" * 70)
        try:
            flops_stats = analyze_flops_simple(model, inputs, device, param_stats)

            if flops_stats['total'] > 0:
                print(f"{'Component':<25} {'FLOPs':>15} {'Percentage':>12}")
                print("-" * 70)

                # Show visual vs text breakdown
                visual_pct = 100 * flops_stats['visual_estimated'] / flops_stats['total']
                text_pct = 100 * flops_stats['text_estimated'] / flops_stats['total']

                print(f"{'Visual (estimated)':<25} {humanize(flops_stats['visual_estimated']):>15} {visual_pct:>11.1f}%")
                if enable_text:
                    print(f"{'Text (estimated)':<25} {humanize(flops_stats['text_estimated']):>15} {text_pct:>11.1f}%")
                print("-" * 70)
                print(f"{'Total FLOPs':<25} {humanize(flops_stats['total']):>15} {'100.0%':>12}")
                print("-" * 70)
                print("  Note: Visual/Text FLOPs estimated from parameter ratio")
                print("        Text tokens participate in cross-attention with visual features")
            else:
                print("ERROR: FLOPs = 0, calculation may have failed")
        except Exception as e:
            print(f"ERROR: FLOPs calculation failed: {e}")
            print("Try running with --skip-flops or --device cpu")
    else:
        print("\nℹ️  FLOPs calculation skipped (use without --skip-flops to enable)")

    print("\n" + "=" * 70)

    # Quick summary
    print("\n📝 SUMMARY:")
    print(f"  Total Params: {humanize(param_stats['total'])} = {humanize(param_stats['visual'])} (visual) + {humanize(param_stats['text'])} (text)")
    if not args.skip_flops:
        try:
            if flops_stats.get('total', 0) > 0:
                print(f"  Total FLOPs:  {humanize(flops_stats['total'])}")
        except NameError:
            pass
    if total_sam > 0:
        print(f"  SAM Params:   {humanize(total_sam)} ({100*total_sam/param_stats['total']:.1f}% of total)")
    print()


if __name__ == "__main__":
    main()
