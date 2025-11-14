#!/usr/bin/env python3
"""
调试模型加载流程，理解 missing keys 的来源
"""

import torch
import sys

def analyze_checkpoint(path, name="checkpoint"):
    """分析 checkpoint 包含的键"""
    print(f"\n{'='*80}")
    print(f"分析 {name}: {path}")
    print(f"{'='*80}")

    try:
        ckpt = torch.load(path, map_location='cpu')

        # 提取 state_dict
        if isinstance(ckpt, dict):
            if 'model' in ckpt:
                state_dict = ckpt['model']
                print("✓ 使用 checkpoint['model']")
            elif 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
                print("✓ 使用 checkpoint['state_dict']")
            else:
                state_dict = ckpt
                print("✓ 使用整个 checkpoint")
        else:
            state_dict = ckpt
            print("✓ checkpoint 本身就是 state_dict")

        # 统计键
        all_keys = list(state_dict.keys())
        encoder_sam_keys = [k for k in all_keys if 'encoder_sam_blocks' in k]
        decoder_sam_keys = [k for k in all_keys if 'decoder_sam_blocks' in k]
        backbone_keys = [k for k in all_keys if not ('sam' in k or 'decode_head' in k or 'decoder' in k)]

        print(f"\n总键数: {len(all_keys)}")
        print(f"  - Backbone 相关: {len(backbone_keys)}")
        print(f"  - encoder_sam_blocks: {len(encoder_sam_keys)}")
        print(f"  - decoder_sam_blocks: {len(decoder_sam_keys)}")

        if encoder_sam_keys:
            print(f"\n✓ 包含 encoder_sam_blocks 参数")
            # 分析 stage 分布
            stages = set()
            for k in encoder_sam_keys:
                if 'encoder_sam_blocks.' in k:
                    parts = k.split('.')
                    if len(parts) >= 2:
                        stage = parts[1]  # encoder_sam_blocks.{stage}.{block}.{param}
                        stages.add(stage)
            print(f"  Stages: {sorted(stages)}")
            print(f"\n  前 10 个键:")
            for k in encoder_sam_keys[:10]:
                print(f"    - {k}")
        else:
            print(f"\n✗ 不包含 encoder_sam_blocks 参数")
            print(f"  这说明这是一个 **pretrained backbone**，不包含 SAM 模块")

        return {
            'has_encoder_sam': len(encoder_sam_keys) > 0,
            'has_decoder_sam': len(decoder_sam_keys) > 0,
            'encoder_sam_count': len(encoder_sam_keys),
            'decoder_sam_count': len(decoder_sam_keys),
            'total_keys': len(all_keys),
        }

    except Exception as e:
        print(f"✗ 加载失败: {e}")
        return None

def main():
    print("\n" + "="*80)
    print("模型加载流程分析")
    print("="*80)

    print("\n理论分析:")
    print("-" * 80)
    print("评估时的加载顺序:")
    print("  1. 创建模型（models/builder.py）")
    print("  2. 加载 pretrained_model（DFormerv2.py init_weights）")
    print("     └─> 报 missing keys: encoder_sam_blocks.*")
    print("  3. 加载训练好的 checkpoint（eval.py line 231）")
    print("     └─> 覆盖所有参数，包括 encoder_sam_blocks")
    print("\n关键发现:")
    print("  - missing keys 警告来自步骤 2（加载 pretrained backbone）")
    print("  - pretrained backbone 只包含基础视觉特征提取器，不包含 SAM 模块")
    print("  - 这是**正常现象**，因为 SAM 是针对 segmentation 的特定模块")
    print("  - 步骤 3 会用训练好的完整权重覆盖，所以**不影响最终结果**")
    print("-" * 80)

    # 如果提供了 checkpoint 路径，分析它
    if len(sys.argv) > 1:
        print("\n实际 checkpoint 分析:")
        print("-" * 80)

        # 分析 pretrained model
        pretrained_path = "checkpoints/pretrained/DFormerv2_Large_pretrained.pth"
        print("\n[1] Pretrained Backbone:")
        analyze_checkpoint(pretrained_path, "Pretrained Model")

        # 分析训练好的 checkpoint
        if len(sys.argv) > 1:
            trained_path = sys.argv[1]
            print("\n[2] 训练好的 Checkpoint:")
            result = analyze_checkpoint(trained_path, "Trained Checkpoint")

            if result:
                print("\n" + "="*80)
                print("结论:")
                print("="*80)
                if result['has_encoder_sam']:
                    print("✓ 训练好的 checkpoint 包含 encoder_sam_blocks")
                    print("✓ 评估时加载这个 checkpoint 后，所有参数都会被正确覆盖")
                    print("✓ missing keys 警告可以安全忽略")
                else:
                    print("✗ 训练好的 checkpoint 也不包含 encoder_sam_blocks")
                    print("✗ 这说明训练时可能没有启用 superpower 或 sam_enc_stages")
                    print("✗ 评估时如果启用了这些配置，会导致参数不匹配")

if __name__ == "__main__":
    main()
