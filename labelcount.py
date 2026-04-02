import json

json_path = "datasets/NYUDepthv2/top6_labels_per_image.json"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 统计每个子条目的词条数量
lengths = [len(v) for v in data.values() if isinstance(v, list)]

if len(lengths) == 0:
    print("没有可统计的条目")
else:
    avg_num = sum(lengths) / len(lengths)
    print(f"总条目数: {len(lengths)}")
    print(f"平均每个条目的词条数: {avg_num:.2f}")