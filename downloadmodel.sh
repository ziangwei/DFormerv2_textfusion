#!/bin/bash
#SBATCH --job-name=model_download   # 任务名，方便您识别
#SBATCH --partition=lrz-cpu           # 关键：指定使用CPU分区
#SBATCH --qos=cpu                     # 新增：为CPU分区明确指定cpu QOS
#SBATCH --nodes=1                     # 申请1个节点
#SBATCH --ntasks=1                    # 在这个节点上运行1个任务
#SBATCH --cpus-per-task=4             # 为这个任务申请4个CPU核心，处理图片I/O多给一点核心有好处
#SBATCH --mem=64G                     # 申请64GB内存，和您之前设的一样
#SBATCH --time=02:00:00               # 任务最长运行时长，设置为8小时，应该足够了

CACHE_DIR="/dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/huggingface_cache"
mkdir -p ${CACHE_DIR}
export HF_HOME=${CACHE_DIR}
