#!/bin/bash
# Evaluation script for DFormerv2
# Usage: bash eval.sh

# ===== GPU Configuration =====
GPUS=2  # Number of GPUs to use (set to 8 for full-scale evaluation)
NNODES=1
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29158}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

export CUDA_VISIBLE_DEVICES="0,1"  # Specify GPU IDs
export TORCHDYNAMO_VERBOSE=1

# ===== Model Configuration =====
CONFIG="local_configs.NYUDepthv2.DFormerv2_S"
CHECKPOINT="checkpoints/NYUDepthv2_DFormerv2_S_CLIPsoftmax(2)_60.66/epoch-508_miou_60.5.pth"

# ===== Evaluation Settings =====
# Text guidance settings are read from config file by default
# You can override them with command-line arguments if needed:
#   --text-source imglabels
#   --text-encoder clip
#   --image-labels-json-path datasets/NYUDepthv2/out.json
#   --max-templates-per-label 3

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    utils/eval.py \
    --config=$CONFIG \
    --gpus=$GPUS \
    --continue_fpath="$CHECKPOINT" \
    --sliding \
    --no-compile \
    --syncbn \
    --mst \
    --amp \
    --compile_mode="reduce-overhead"

# ===== Notes =====
#
# Dataset-specific settings:
#   - NYUDepthv2: No --pad_SUNRGBD needed
#   - SUNRGBD: Add --pad_SUNRGBD flag
#
# Model configs:
#   - NYUv2: local_configs.NYUDepthv2.DFormerv2_{S/B/L}
#   - SUNRGBD: local_configs.SUNRGBD.DFormerv2_{S/B/L}
#
# Common optional arguments:
#   --no-mst              # Disable multi-scale testing (faster, lower accuracy)
#   --no-sliding          # Disable sliding window inference
#   --no-amp              # Disable mixed precision (higher memory usage)
#   --compile             # Enable torch.compile (may speed up inference)
#
# Text guidance overrides (if you want to change from config):
#   --text-source {labels|captions|both|imglabels}
#   --text-encoder {clip|jinaclip}
#   --image-labels-json-path PATH
#   --caption-json-path PATH
#   --max-templates-per-label N

