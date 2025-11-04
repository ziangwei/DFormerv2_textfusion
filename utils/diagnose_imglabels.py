#!/usr/bin/env python3
"""
诊断工具：检查 imglabels JSON 的 key 匹配问题

用法：
    python utils/diagnose_imglabels.py \
        --image-labels-json datasets/sunrgbd/image_labels.json \
        --eval-source datasets/sunrgbd/test.txt \
        --rgb-root datasets/sunrgbd/SUNRGBD \
        --rgb-format .jpg
"""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Diagnose imglabels key matching")
    parser.add_argument("--image-labels-json", required=True, help="Path to image_labels.json")
    parser.add_argument("--eval-source", required=True, help="Path to eval list (e.g., test.txt)")
    parser.add_argument("--rgb-root", required=True, help="RGB image root directory")
    parser.add_argument("--rgb-format", default=".jpg", help="RGB image format")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples to check")
    args = parser.parse_args()

    # 1. 加载 image_labels.json
    if not os.path.exists(args.image_labels_json):
        print(f"❌ Image labels JSON not found: {args.image_labels_json}")
        return

    with open(args.image_labels_json, 'r', encoding='utf-8') as f:
        image_labels_dict = json.load(f)

    print("=" * 80)
    print("🔍 ImgLabels Key Matching Diagnosis")
    print("=" * 80)
    print(f"\n1. Loaded image_labels.json: {args.image_labels_json}")
    print(f"   Total entries: {len(image_labels_dict)}")

    # 显示 JSON 中的 key 格式示例
    sample_keys = list(image_labels_dict.keys())[:5]
    print(f"\n   Sample keys in JSON:")
    for k in sample_keys:
        labels = image_labels_dict[k]
        if isinstance(labels, list):
            label_str = ', '.join(str(l) for l in labels[:3])
            if len(labels) > 3:
                label_str += f", ... ({len(labels)} total)"
        else:
            label_str = str(labels)
        print(f"      '{k}' -> [{label_str}]")

    # 2. 加载评估列表
    if not os.path.exists(args.eval_source):
        print(f"\n❌ Eval source not found: {args.eval_source}")
        return

    with open(args.eval_source, 'r') as f:
        eval_items = [line.strip() for line in f if line.strip()]

    print(f"\n2. Loaded eval source: {args.eval_source}")
    print(f"   Total items: {len(eval_items)}")

    # 3. 测试匹配
    print(f"\n3. Testing key matching for first {args.num_samples} images:")
    print("=" * 80)

    matched = 0
    unmatched = 0

    for i, item_name in enumerate(eval_items[:args.num_samples]):
        # 构造 rgb_path（模拟 get_path 函数的行为）
        base_name = os.path.basename(item_name).split('.')[0]
        rgb_path = os.path.join(args.rgb_root, base_name + args.rgb_format)

        # 尝试各种 key 候选
        key_candidates = [
            item_name,                      # 原始 item_name
            os.path.basename(item_name),    # basename of item_name
            rgb_path,                       # 完整 rgb_path
            os.path.basename(rgb_path),     # basename of rgb_path
            base_name + args.rgb_format,    # base_name + format
            base_name,                      # 纯 base_name
        ]

        # 尝试匹配
        found_key = None
        found_labels = None
        for k in key_candidates:
            if k in image_labels_dict:
                found_key = k
                found_labels = image_labels_dict[k]
                break

        # 显示结果
        print(f"\n[{i+1}] item_name: {item_name}")
        print(f"    rgb_path: {rgb_path}")

        if found_key:
            matched += 1
            labels = found_labels
            if isinstance(labels, list):
                label_str = ', '.join(str(l) for l in labels[:5])
                if len(labels) > 5:
                    label_str += f", ... ({len(labels)} total)"
            else:
                label_str = str(labels)
            print(f"    ✅ MATCHED with key: '{found_key}'")
            print(f"    Labels: [{label_str}]")
        else:
            unmatched += 1
            print(f"    ❌ NOT MATCHED")
            print(f"    Tried keys:")
            for k in set(key_candidates):  # 去重显示
                print(f"      - '{k}'")

    print("\n" + "=" * 80)
    print("📊 Summary:")
    print("=" * 80)
    print(f"Matched:   {matched}/{args.num_samples} ({matched/max(args.num_samples,1)*100:.1f}%)")
    print(f"Unmatched: {unmatched}/{args.num_samples} ({unmatched/max(args.num_samples,1)*100:.1f}%)")

    if unmatched > 0:
        print("\n" + "=" * 80)
        print("💡 Recommendations:")
        print("=" * 80)
        print("\n1. Check if JSON keys match your file structure:")
        print(f"   - JSON keys look like: {sample_keys[0] if sample_keys else 'N/A'}")
        print(f"   - Image paths look like: {rgb_path}")

        print("\n2. Common issues:")
        print("   a) JSON keys use full path, but code expects basename")
        print("   b) JSON keys use basename, but code provides full path")
        print("   c) File extensions mismatch (.jpg vs .png)")
        print("   d) Dataset name prefix in keys (e.g., 'sunrgbd/image/...')")

        print("\n3. Suggested fixes:")
        print("   a) Normalize JSON keys to use basename only:")
        print("      python utils/normalize_imglabels_keys.py \\")
        print(f"          --input {args.image_labels_json} \\")
        print(f"          --output {args.image_labels_json}.normalized.json")

        print("\n   b) Or modify RGBXDataset.py to try more key variations")

    else:
        print("\n✅ All samples matched successfully!")
        print("   Your imglabels configuration should work correctly.")


if __name__ == "__main__":
    main()
