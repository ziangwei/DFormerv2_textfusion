#!/usr/bin/env python3
"""
诊断脚本：检查注意力可视化的配置是否正确

用法：
    python utils/check_attention_setup.py --config configs.sunrgbd.your_config
"""
import argparse
import importlib
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Check attention visualization setup")
    parser.add_argument("--config", required=True, help="Config module path (e.g., configs.sunrgbd.my_cfg)")
    args = parser.parse_args()

    # 加载配置
    try:
        C = getattr(importlib.import_module(args.config), "C")
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return

    print("=" * 80)
    print("🔍 注意力可视化配置检查")
    print("=" * 80)

    # 1. 检查 text guidance 是否开启
    enable_text = getattr(C, "enable_text_guidance", False)
    print(f"\n1. Text Guidance: {'✅ ENABLED' if enable_text else '❌ DISABLED (必须开启才能可视化注意力)'}")

    if not enable_text:
        print("   解决方案：在配置文件中设置 C.enable_text_guidance = True")
        return

    # 2. 检查 text source 配置
    text_source = getattr(C, "text_source", "both")
    print(f"\n2. Text Source: {text_source}")

    # 3. 检查标签文件
    label_txt_path = getattr(C, "label_txt_path", None)
    if label_txt_path and os.path.exists(label_txt_path):
        with open(label_txt_path, 'r', encoding='utf-8') as f:
            labels = [line.strip() for line in f if line.strip()]
        print(f"   ✅ Label file: {label_txt_path}")
        print(f"   📋 Found {len(labels)} class labels:")
        print(f"      {', '.join(labels[:10])}{'...' if len(labels) > 10 else ''}")
    else:
        print(f"   ⚠️  Label file not found: {label_txt_path}")

    # 4. 检查 caption 配置
    caption_json_path = getattr(C, "caption_json_path", None)
    if text_source in ("captions", "both"):
        if caption_json_path and os.path.exists(caption_json_path):
            print(f"   ✅ Caption file: {caption_json_path}")
        else:
            print(f"   ⚠️  Caption file not found: {caption_json_path}")

    # 5. 检查 image labels 配置
    image_labels_json = getattr(C, "image_labels_json_path", None)
    if text_source == "imglabels":
        if image_labels_json and os.path.exists(image_labels_json):
            print(f"   ✅ Image labels file: {image_labels_json}")
            # 检查 JSON 格式
            try:
                import json
                with open(image_labels_json, 'r', encoding='utf-8') as f:
                    img_labels = json.load(f)
                print(f"      Total images in JSON: {len(img_labels)}")
                sample_keys = list(img_labels.keys())[:3]
                print(f"      Sample keys:")
                for k in sample_keys:
                    labels = img_labels[k]
                    if isinstance(labels, list):
                        print(f"        '{k}' -> {len(labels)} labels")
                    else:
                        print(f"        '{k}' -> {labels}")
            except Exception as e:
                print(f"      ⚠️ Failed to parse JSON: {e}")
        else:
            print(f"   ❌ Image labels file not found: {image_labels_json}")
            print(f"      This is REQUIRED for text_source='imglabels' mode!")
            return

    # 6. 检查 SAM 配置
    sam_enc_stages = getattr(C, "sam_enc_stages", None)
    sam_dec_stages = getattr(C, "sam_dec_stages", None)
    print(f"\n3. SAM Configuration:")
    print(f"   Encoder stages: {sam_enc_stages}")
    print(f"   Decoder stages: {sam_dec_stages}")

    if not sam_enc_stages and not sam_dec_stages:
        print("   ⚠️  No SAM stages configured! Attention visualization may not work.")

    # 7. Token 数量估算
    print(f"\n4. Token Count Estimation:")
    cap_k = getattr(C, "caption_topk", 0)
    cap_max = getattr(C, "max_caption_sentences", 0)
    cap_tokens = cap_k if (isinstance(cap_k, int) and cap_k > 0) else cap_max

    if text_source == "labels":
        total_tokens = len(labels) if labels else C.num_classes
        print(f"   Mode: labels only")
        print(f"   Expected tokens per image: {total_tokens}")
    elif text_source == "captions":
        print(f"   Mode: captions only")
        print(f"   Expected tokens per image: {cap_tokens}")
    elif text_source == "imglabels":
        max_img_labels = getattr(C, "max_image_labels", 0)
        print(f"   Mode: per-image labels")
        print(f"   Max tokens per image: {max_img_labels or 'variable'}")
    else:  # both
        total_tokens = (len(labels) if labels else C.num_classes) + cap_tokens
        print(f"   Mode: labels + captions")
        print(f"   Expected tokens per image: {total_tokens}")
        print(f"      - Class tokens: {len(labels) if labels else C.num_classes}")
        print(f"      - Caption tokens: {cap_tokens}")

    print("\n" + "=" * 80)
    print("💡 使用建议：")
    print("=" * 80)

    if text_source == "imglabels":
        print("\n⚠️  当前使用 imglabels 模式，请先诊断 key 匹配：")
        print(f"\n   python utils/diagnose_imglabels.py \\")
        print(f"       --image-labels-json {image_labels_json or 'YOUR_JSON'} \\")
        eval_src = getattr(C, "eval_source", "datasets/sunrgbd/test.txt")
        print(f"       --eval-source {eval_src} \\")
        rgb_root = getattr(C, "rgb_root", "datasets/sunrgbd/SUNRGBD")
        print(f"       --rgb-root {rgb_root} \\")
        rgb_fmt = getattr(C, "rgb_format", ".jpg")
        print(f"       --rgb-format {rgb_fmt}")

        print("\n   如果发现 key 不匹配，可以规范化 JSON：")
        print(f"\n   python utils/normalize_imglabels_keys.py \\")
        print(f"       --input {image_labels_json or 'YOUR_JSON'} \\")
        print(f"       --output {image_labels_json or 'YOUR_JSON'}.normalized.json")
        print("\n")

    print("\n1. 可视化所有 tokens（能量排序）：")
    print(f"   python utils/infer.py --config {args.config} \\")
    print("       --save-attention --save_path ./vis_output \\")
    print("       --vis-stage enc --vis-stage-idx 0 --num-images 10")

    print("\n2. 只可视化特定 tokens（例如 floor 和 wall）：")
    print(f"   python utils/infer.py --config {args.config} \\")
    print("       --save-attention --save_path ./vis_output \\")
    print("       --vis-stage enc --vis-stage-idx 0 --num-images 10 \\")
    print("       --filter-tokens 'floor,wall'")

    print("\n3. 调整可视化参数：")
    print(f"   python utils/infer.py --config {args.config} \\")
    print("       --save-attention --save_path ./vis_output \\")
    print("       --attention-alpha 0.6 \\        # 叠加透明度")
    print("       --attention-threshold 0.1 \\    # 过滤低响应区域")
    print("       --attention-smooth 1.0          # 高斯平滑")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
