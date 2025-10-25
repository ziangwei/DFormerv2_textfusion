#!/bin/bash
#SBATCH --job-name=cpu_stuff    # 任务名，方便您识别
#SBATCH --partition=lrz-cpu           # 关键：指定使用CPU分区
#SBATCH --qos=cpu                     # 新增：为CPU分区明确指定cpu QOS
#SBATCH --nodes=1                     # 申请1个节点
#SBATCH --ntasks=1                    # 在这个节点上运行1个任务
#SBATCH --cpus-per-task=1             # 为这个任务申请1个CPU核心
#SBATCH --mem=64G                     # 申请64GB内存，和您之前设的一样
#SBATCH --time=00:15:00               # 任务最长运行时长

#python check.py
python ids2text_nyu37.py \
  --in datasets/NYUDepthv2/topk_labels2_internvl3.json \
  --out datasets/NYUDepthv2/topk_labels2_text_internvl3.json
