#!/bin/bash
#SBATCH --job-name=generate_labels_top5    # 任务名
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --nodes=1                     # 申请1个节点
#SBATCH --gpus-per-node=1             # 为每个节点申请1个GPU
#SBATCH --mem=64G                     # 申请64GB内存，和您之前设的一样
#SBATCH --time=01:20:00               # 任务最长运行时长，设置为8小时，应该足够了

CACHE_DIR="/dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/huggingface_cache"
mkdir -p ${CACHE_DIR}
export HF_HOME=${CACHE_DIR}

python generate_tags_internvl3.py \
  --model_id OpenGVLab/InternVL3-38B \
  --batch_size 2 \
  --max_new_tokens 64


echo "标签生成完成！"
