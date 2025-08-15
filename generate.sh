#!/bin/bash
#SBATCH --job-name=text_generate
#SBATCH --partition=lrz-v100x2
#SBATCH --gres=gpu:2       # 请求1个GPU
#SBATCH --time=14:00:00    # 运行时间限制
#SBATCH --mem=64G          # 内存需求

# srun python3 blip_generate.py

# srun python3 generate_clip_prompts.py

# srun python3 check.py

srun python3 label_generate.py
