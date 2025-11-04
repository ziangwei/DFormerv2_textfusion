#!/usr/bin/env python3
"""
规范化 image_labels.json 的 keys

这个工具会：
1. 读取原始的 image_labels.json
2. 为每个条目创建多个 key 变体（完整路径、basename、无扩展名等）
3. 生成新的 JSON，确保能匹配到各种路径格式

用法：
    python utils/normalize_imglabels_keys.py \
        --input datasets/sunrgbd/image_labels.json \
        --output datasets/sunrgbd/image_labels.normalized.json
"""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Normalize image_labels.json keys")
    parser.add_argument("--input", required=True, help="Input image_labels.json")
    parser.add_argument("--output", required=True, help="Output normalized JSON")
    parser.add_argument("--force", action="store_true", help="Overwrite output if exists")
    args = parser.parse_args()

    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        return

    # 检查输出文件
    if os.path.exists(args.output) and not args.force:
        print(f"❌ Output file already exists: {args.output}")
        print("   Use --force to overwrite")
        return

    # 加载原始 JSON
    print(f"📖 Loading: {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        original = json.load(f)

    print(f"   Original entries: {len(original)}")

    # 规范化 keys
    normalized = {}
    key_variants_count = 0

    for orig_key, labels in original.items():
        # 生成多个 key 变体
        variants = set()

        # 1. 原始 key
        variants.add(orig_key)

        # 2. basename
        variants.add(os.path.basename(orig_key))

        # 3. 去掉扩展名（basename）
        name_without_ext = os.path.splitext(os.path.basename(orig_key))[0]
        variants.add(name_without_ext)

        # 4. basename 加各种扩展名
        for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG']:
            variants.add(name_without_ext + ext)

        # 5. 如果原始 key 有路径，保留路径但去掉扩展名
        if '/' in orig_key or '\\' in orig_key:
            dir_path = os.path.dirname(orig_key)
            variants.add(os.path.join(dir_path, name_without_ext))

        # 将所有变体都加到输出中
        for v in variants:
            if v:  # 非空
                normalized[v] = labels
                key_variants_count += 1

    print(f"   Normalized entries: {len(normalized)} (from {len(original)} original)")
    print(f"   Total key variants: {key_variants_count}")
    print(f"   Average variants per entry: {key_variants_count / max(len(original), 1):.1f}")

    # 显示示例
    print("\n📋 Sample key mappings:")
    sample_orig_keys = list(original.keys())[:3]
    for orig_key in sample_orig_keys:
        labels = original[orig_key]
        if isinstance(labels, list):
            label_str = ', '.join(str(l) for l in labels[:3])
            if len(labels) > 3:
                label_str += f", ... ({len(labels)} total)"
        else:
            label_str = str(labels)

        print(f"\n   Original key: '{orig_key}'")
        print(f"   Labels: [{label_str}]")
        print(f"   Generated variants:")

        # 找出这个 key 生成的所有变体
        variants = []
        basename = os.path.basename(orig_key)
        name_no_ext = os.path.splitext(basename)[0]
        for v in normalized.keys():
            if (v == orig_key or
                v == basename or
                v.startswith(name_no_ext) or
                basename.startswith(v)):
                variants.append(v)
                if len(variants) >= 5:  # 只显示前5个
                    break

        for v in variants:
            print(f"      - '{v}'")

    # 保存
    print(f"\n💾 Saving to: {args.output}")
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)

    print(f"✅ Done!")
    print(f"\n💡 Next steps:")
    print(f"   1. Update your config to use the normalized JSON:")
    print(f"      C.image_labels_json_path = '{args.output}'")
    print(f"\n   2. Or replace the original file:")
    print(f"      mv {args.output} {args.input}")


if __name__ == "__main__":
    main()
