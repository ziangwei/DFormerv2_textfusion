#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from pathlib import Path
from collections import Counter


def analyze_labels(json_file):
    """分析标签 JSON 文件的统计信息"""

    # 读取 JSON
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 统计每张图的标签数
    label_counts = []
    for img_path, labels in data.items():
        label_counts.append(len(labels))

    if not label_counts:
        print("❌ JSON 文件为空")
        return

    # 基本统计
    total_images = len(label_counts)
    total_labels = sum(label_counts)
    avg_labels = total_labels / total_images
    max_labels = max(label_counts)
    min_labels = min(label_counts)

    # 标签数量分布
    count_distribution = Counter(label_counts)

    # 输出结果
    print("=" * 60)
    print(f"📊 标签统计报告")
    print("=" * 60)
    print(f"📁 文件: {json_file}")
    print(f"🖼️  总图片数: {total_images}")
    print(f"🏷️  总标签数: {total_labels}")
    print(f"📈 平均每张图标签数: {avg_labels:.2f}")
    print(f"📉 最少标签数: {min_labels}")
    print(f"📊 最多标签数: {max_labels}")
    print()

    print("=" * 60)
    print("📋 标签数量分布:")
    print("=" * 60)
    for num_labels in sorted(count_distribution.keys()):
        count = count_distribution[num_labels]
        percentage = (count / total_images) * 100
        bar = "█" * int(percentage / 2)
        print(f"{num_labels:2d} 个标签: {count:4d} 张图 ({percentage:5.1f}%) {bar}")

    print()
    print("=" * 60)
    print("🔝 标签频率统计 (Top 20):")
    print("=" * 60)

    # 统计所有标签出现频率
    all_labels = []
    for labels in data.values():
        all_labels.extend(labels)

    label_freq = Counter(all_labels)
    for i, (label, count) in enumerate(label_freq.most_common(20), 1):
        percentage = (count / total_images) * 100
        print(f"{i:2d}. {label:20s}: {count:4d} 次 (出现在 {percentage:5.1f}% 的图中)")


def main():
    parser = argparse.ArgumentParser(description='统计标签 JSON 文件')
    parser.add_argument('json_file', type=str, help='标签 JSON 文件路径')
    args = parser.parse_args()

    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"❌ 文件不存在: {json_path}")
        return

    analyze_labels(json_path)


if __name__ == "__main__":
    main()