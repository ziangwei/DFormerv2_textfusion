#!/bin/bash
#SBATCH --job-name=blip_generate
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:2       # 请求1个GPU
#SBATCH --time=14:00:00    # 运行时间限制
#SBATCH --mem=64G          # 内存需求

srun python3 blip_generate.py