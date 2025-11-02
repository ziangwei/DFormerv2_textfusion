#!/bin/bash
#SBATCH --job-name=dataprocess    # 任务名，方便您识别
#SBATCH --partition=lrz-cpu           # 关键：指定使用CPU分区
#SBATCH --qos=cpu                     # 新增：为CPU分区明确指定cpu QOS
#SBATCH --nodes=1                     # 申请1个节点
#SBATCH --ntasks=1                    # 在这个节点上运行1个任务
#SBATCH --cpus-per-task=4             # 为这个任务申请4个CPU核心，处理图片I/O多给一点核心有好处
#SBATCH --mem=64G                     # 申请64GB内存，和您之前设的一样
#SBATCH --time=00:15:00               # 任务最长运行时长，设置为8小时，应该足够了

# 读深度图生成的label真值标签（用作性能对比）
#python label_generate.py

# 如果输出的json里是数字可以用这个转换
#python ids2text_nyu37.py \
#  --in datasets/NYUDepthv2/topk_labels6_internvl3.json \
#  --out datasets/NYUDepthv2/topk_labels6_text_internvl3.json

# 取多个大模型生成的文本的交集
python intersect_labels.py \
    datasets/SUNRGBD/out.json \
    datasets/SUNRGBD/sunrgbd_image_labels_internvl3.json \
    datasets/SUNRGBD/sunrgbd_image_labels_qwen3vl.json

# 计算某json与真值标签的重合度
#python compare_json_overlap.py \
#    datasets/NYUDepthv2/out.json \
#    datasets/NYUDepthv2/top5_labels_per_image.json

# 计算某json的标签平均数
#python stats_labels.py datasets/NYUDepthv2/out.json

# 画图
#python nyu_overlay.py \
#  --rgb datasets/NYUDepthv2/RGB/976.jpg \
#  --label datasets/NYUDepthv2/Label/976.png \
#  --colormap utils/nyucmap.npy \
#  --style paper-soft \
#  --out out/f2.png
