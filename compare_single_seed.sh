#!/bin/bash
# 单seed对照实验：快速验证
# 用法: bash compare_single_seed.sh [seed] [config] [gpus]

SEED=${1:-42}
CONFIG=${2:-"local_configs.NYUDepthv2.DFormerv2_S"}
GPUS=${3:-2}

echo "=========================================="
echo "单Seed对照实验"
echo "  Config: $CONFIG"
echo "  Seed:   $SEED"
echo "  GPUs:   $GPUS"
echo "=========================================="

# Baseline (无文本)
echo -e "\n[1/2] 启动 Baseline (无文本)..."
torchrun --nproc_per_node=$GPUS --master_port=$((29500 + SEED % 1000)) \
    -m utils.train \
    --config $CONFIG \
    --seed $SEED \
    --no-enable-text-guidance \
    --gpus $GPUS \
    --pad_SUNRGBD --amp &

BASELINE_PID=$!
sleep 5

# TextFusion (有文本)
echo "[2/2] 启动 TextFusion (有文本)..."
torchrun --nproc_per_node=$GPUS --master_port=$((30000 + SEED % 1000)) \
    -m utils.train \
    --config $CONFIG \
    --seed $SEED \
    --enable-text-guidance \
    --gpus $GPUS \
    --pad_SUNRGBD --amp &

TEXTFUSION_PID=$!

echo "=========================================="
echo "两个实验已启动："
echo "  Baseline PID:    $BASELINE_PID"
echo "  TextFusion PID:  $TEXTFUSION_PID"
echo "=========================================="

wait
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 对照实验完成！"
