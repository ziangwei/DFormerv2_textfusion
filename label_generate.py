import os
import json
import numpy as np
from PIL import Image

# 1) label→名称 映射
label_map = {
    0: "unlabeled",
    1: "wall",
    2: "floor",
    3: "cabinet",
    4: "bed",
    5: "chair",
    6: "sofa",
    7: "table",
    8: "door",
    9: "window",
    10: "bookshelf",
    11: "picture",
    12: "counter",
    13: "blinds",
    14: "desk",
    15: "shelves",
    16: "curtain",
    17: "dresser",
    18: "pillow",
    19: "mirror",
    20: "floor mat",
    21: "clothes",
    22: "ceiling",
    23: "books",
    24: "refridgerator",
    25: "television",
    26: "paper",
    27: "towel",
    28: "shower curtain",
    29: "box",
    30: "whiteboard",
    31: "person",
    32: "night stand",
    33: "toilet",
    34: "sink",
    35: "lamp",
    36: "bathtub",
    37: "bag",
    38: "otherstructure",
    39: "otherfurniture",
    40: "otherprop"
}
ignore_ids = {0, 38, 39, 40}

# 2) 标签图文件夹
label_folder = 'datasets/NYUDepthv2/Label'

result = {}
for fn in os.listdir(label_folder):
    if not fn.lower().endswith('.png'):
        continue
    arr = np.array(Image.open(os.path.join(label_folder, fn)))
    u, c = np.unique(arr, return_counts=True)

    # 先过滤掉不需要的 id
    valid = [(i, cnt) for i, cnt in zip(u, c) if i not in ignore_ids]

    # 再按出现次数排前 5
    top5 = sorted(valid, key=lambda x: x[1], reverse=True)[:5]
    # 最终取名字
    names_top5 = [label_map.get(int(idx), f"class_{idx}") for idx, _ in top5]
    result[fn] = names_top5

# 保存到上级目录
out_path = os.path.join(os.path.dirname(label_folder), 'top5_labels_per_image.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=4, ensure_ascii=False)

print("Top-5 labels per image saved to:", out_path)
print(json.dumps(result, indent=2, ensure_ascii=False))
