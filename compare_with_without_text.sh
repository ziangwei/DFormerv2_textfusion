#!/bin/bash
# 对照实验：相同seed下对比有/无文本fusion的效果
# 用法: bash compare_with_without_text.sh

# ============================================
# 实验配置
# ============================================
CONFIG="local_configs.NYUDepthv2.DFormerv2_S"
GPUS=2
SEEDS=(42 123 456 789 1024)  # 多seed取平均，提高可信度

# 可选参数
EXTRA_ARGS="--pad_SUNRGBD --amp"

echo "=========================================="
echo "对照实验：Text Fusion vs Baseline"
echo "配置: $CONFIG"
echo "Seeds: ${SEEDS[@]}"
echo "每个seed运行2次实验："
echo "  1. Baseline (无文本)"
echo "  2. TextFusion (有文本)"
echo "=========================================="

# ============================================
# 实验1：Baseline（无文本）
# ============================================
echo -e "\n[实验组1] Baseline - 无文本fusion"
echo "=========================================="

for seed in "${SEEDS[@]}"; do
    echo "[$(date '+%H:%M:%S')] 启动 Baseline seed=$seed"

    torchrun --nproc_per_node=$GPUS --master_port=$((29500 + seed % 1000)) \
        -m utils.train \
        --config $CONFIG \
        --seed $seed \
        --no-enable-text-guidance \
        --gpus $GPUS \
        $EXTRA_ARGS &

    sleep 3  # 避免同时启动
done

echo "Baseline组全部启动，等待5秒后启动TextFusion组..."
sleep 5

# ============================================
# 实验2：TextFusion（有文本）
# ============================================
echo -e "\n[实验组2] TextFusion - 有文本fusion"
echo "=========================================="

for seed in "${SEEDS[@]}"; do
    echo "[$(date '+%H:%M:%S')] 启动 TextFusion seed=$seed"

    torchrun --nproc_per_node=$GPUS --master_port=$((30000 + seed % 1000)) \
        -m utils.train \
        --config $CONFIG \
        --seed $seed \
        --enable-text-guidance \
        --gpus $GPUS \
        $EXTRA_ARGS &

    sleep 3  # 避免同时启动
done

echo "=========================================="
echo "所有实验任务已启动！"
echo ""
echo "监控命令："
echo "  查看进程: ps aux | grep train"
echo "  查看GPU:  nvidia-smi"
echo "  查看日志: tail -f checkpoints/*/log_*.log"
echo ""
echo "预计运行时间: ~500 epochs × 训练时间"
echo "完成后对比两组的 mean ± std"
echo "=========================================="

wait
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 所有实验完成！"
