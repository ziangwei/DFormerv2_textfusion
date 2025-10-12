#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, argparse, sys
from pathlib import Path

# ---- NYUv2 40 类，去掉最后 3 个模糊类 = 37 类（与之前生成ID的顺序完全一致）----
NYU40 = [
    "wall","floor","cabinet","bed","chair","sofa","table","door","window","bookshelf",
    "picture","counter","blinds","desk","shelves","curtain","dresser","pillow","mirror","floor mat",
    "clothes","ceiling","books","refrigerator","television","paper","towel","shower curtain","box","whiteboard",
    "person","night stand","toilet","sink","lamp","bathtub","bag","otherstructure","otherfurniture","otherprop"
]
AMBIG = {"otherstructure","otherfurniture","otherprop"}
NYU37 = [c for c in NYU40 if c not in AMBIG]  # index: 0..36

def ids_to_labels(ids):
    labels = []
    for i in ids:
        if not isinstance(i, int) or i < 0 or i >= len(NYU37):
            # 遇到非法ID时给出占位并继续
            labels.append(f"<invalid_id:{i}>")
        else:
            labels.append(NYU37[i])
    return labels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="输入：图片→ID数组 的 JSON 路径")
    ap.add_argument("--out", dest="out", required=True, help="输出：图片→文本标签数组 的 JSON 路径")
    ap.add_argument("--mode", choices=["text","both"], default="text",
                    help="text=仅输出文本; both=同时包含ids与labels")
    args = ap.parse_args()

    in_path = Path(args.inp)
    out_path = Path(args.out)

    if not in_path.exists():
        print(f"[ERROR] 输入文件不存在: {in_path}", file=sys.stderr)
        sys.exit(1)

    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        print("[ERROR] 输入 JSON 格式应为 {image_rel_path: [id,...]}", file=sys.stderr)
        sys.exit(1)

    out_dict = {}
    bad = 0
    for img, ids in data.items():
        if not isinstance(ids, list):
            bad += 1
            continue
        labels = ids_to_labels(ids)
        if args.mode == "text":
            out_dict[img] = labels
        else:  # both
            out_dict[img] = {"ids": ids, "labels": labels}

        # 统计非法ID
        bad += sum(1 for x in labels if x.startswith("<invalid_id:"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, ensure_ascii=False, indent=2)

    print(f"[DONE] 写入 {out_path}")
    if bad > 0:
        print(f"[WARN] 发现 {bad} 处非法ID，已用 <invalid_id:*> 占位。")

if __name__ == "__main__":
    main()
