#!/usr/bin/env python3
"""
验证去重策略不会错误合并不同的标签
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from utils.prompt_utils import _normalize_label

def test_deduplication_logic():
    """测试去重逻辑"""

    print("=" * 60)
    print("去重策略验证")
    print("=" * 60)

    # 模拟各种可能的标签输入
    test_cases = [
        # (原始标签, 标准化后, 说明)
        ("sofa", "sofa", "标准词"),
        ("Sofa", "sofa", "大小写变体 - 会去重"),
        ("SOFA", "sofa", "全大写 - 会去重"),
        ("sofa ", "sofa", "末尾空格 - 会去重"),
        ("  sofa  ", "sofa", "多余空格 - 会去重"),

        ("couch", "couch", "近义词但字符串不同 - 不会去重"),
        ("chair", "chair", "不同的词"),
        ("chairs", "chairs", "单复数不同 - 不会去重"),
        ("desk", "desk", "不同的词"),
        ("table", "table", "语义相近但不同 - 不会去重"),

        ("television", "tv", "别名映射 - 会合并为tv"),
        ("tv", "tv", "缩写形式"),
        ("TV", "tv", "大写缩写 - 会去重"),

        ("floor mat", "rug", "别名映射 - 会合并为rug"),
        ("rug", "rug", "目标词"),

        ("bookshelf", "bookcase", "别名映射"),
        ("bookcase", "bookcase", "目标词"),

        ("living room chair", "living room chair", "组合词 - 独立编码"),
        ("desk lamp", "desk lamp", "组合词 - 独立编码"),
    ]

    print("\n测试标准化结果:")
    print("-" * 60)

    unique_labels = []
    seen = set()
    mapping = {}

    for original, expected, description in test_cases:
        normalized = _normalize_label(original)
        is_duplicate = normalized in seen

        # 验证标准化结果
        assert normalized == expected, f"标准化错误: {original} -> {normalized}, 预期: {expected}"

        status = "❌ 去重" if is_duplicate else "✅ 保留"
        print(f"{status} | '{original:20s}' → '{normalized:15s}' | {description}")

        if not is_duplicate:
            unique_labels.append(normalized)
            seen.add(normalized)
            mapping[normalized] = []

        mapping[normalized].append(original)

    print("\n" + "=" * 60)
    print(f"总结: {len(test_cases)} 个输入 → {len(unique_labels)} 个唯一标签")
    print("=" * 60)

    print("\n唯一标签及其原始变体:")
    for label in unique_labels:
        variants = mapping[label]
        print(f"  '{label}': {variants}")

    print("\n" + "=" * 60)
    print("关键结论:")
    print("=" * 60)
    print("✅ 不同的词（如 sofa/couch, chair/chairs, desk/table）都被独立保留")
    print("✅ 只有大小写/空格差异的才去重（如 Sofa/sofa/SOFA）")
    print("✅ 只有显式别名映射才合并（如 television→tv, floor mat→rug）")
    print("✅ 组合词/新词都独立编码（如 living room chair, desk lamp）")
    print("\n👉 结论: 不会因为语义相似而错误合并，只去重完全相同的标签")


if __name__ == "__main__":
    test_deduplication_logic()
