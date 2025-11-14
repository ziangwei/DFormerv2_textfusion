#!/usr/bin/env python3
"""检查 checkpoint 中是否包含 encoder_sam_blocks 相关的键"""

import torch
import sys

def check_checkpoint(checkpoint_path):
    """检查 checkpoint 包含哪些键"""
    print(f"正在加载 checkpoint: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # 如果 checkpoint 是一个字典，尝试找到 state_dict
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
                print("✓ 找到 checkpoint['model']")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print("✓ 找到 checkpoint['state_dict']")
            else:
                state_dict = checkpoint
                print("✓ 使用整个 checkpoint 作为 state_dict")
        else:
            state_dict = checkpoint
            print("✓ checkpoint 本身就是 state_dict")

        # 查找所有 encoder_sam_blocks 相关的键
        encoder_sam_keys = [k for k in state_dict.keys() if 'encoder_sam_blocks' in k]

        print(f"\n=== Checkpoint 信息 ===")
        print(f"总共键数: {len(state_dict.keys())}")
        print(f"encoder_sam_blocks 相关的键数: {len(encoder_sam_keys)}")

        if len(encoder_sam_keys) > 0:
            print(f"\n✓ Checkpoint 包含 encoder_sam_blocks 参数")
            print(f"\n前10个 encoder_sam_blocks 键:")
            for key in encoder_sam_keys[:10]:
                print(f"  - {key}")
            if len(encoder_sam_keys) > 10:
                print(f"  ... 还有 {len(encoder_sam_keys) - 10} 个键")
        else:
            print(f"\n✗ Checkpoint 不包含任何 encoder_sam_blocks 参数")
            print(f"\n这意味着训练时 superpower=False 或者 sam_enc_stages 为空")

        # 检查是否有 decoder_sam_blocks
        decoder_sam_keys = [k for k in state_dict.keys() if 'decoder_sam_blocks' in k]
        print(f"\ndecoder_sam_blocks 相关的键数: {len(decoder_sam_keys)}")

        # 显示一些样本键来了解结构
        print(f"\n=== Checkpoint 结构样本 (前20个键) ===")
        for i, key in enumerate(list(state_dict.keys())[:20]):
            print(f"  {i+1}. {key}")

    except Exception as e:
        print(f"✗ 加载 checkpoint 失败: {e}")
        return

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python check_checkpoint.py <checkpoint_path>")
        print("\n示例:")
        print("  python check_checkpoint.py checkpoints/SUNRGBD_DFormerv2_L_20251113-171849/epoch-130_miou_49.95.pth")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    check_checkpoint(checkpoint_path)
