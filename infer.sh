# CUDA_VISIBLE_DEVICES=0,1
# config -> which model config
# continue_fpath -> the trained pth path
GPUS=2
NNODES=1
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29958}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    utils/infer.py \
    --config=local_configs.NYUDepthv2.DFormer_Small \
    --continue_fpath=checkpoints/NYUDepthv2_DFormerv2_S_20251030-182749/epoch-330_miou_58.0.pth \
    --save_path "output/" \
    --gpus=$GPUS \
    --text-source=imglabels \
    --text-encoder=clip \
    --sam-enc-stages=1,2,3 \
    --sam-dec-stages=0,1,2,3

# choose the dataset and DFormer for evaluating

# NYUv2 DFormers
# --config=local_configs.NYUDepthv2.DFormer_Large/Base/Small/Tiny
# --continue_fpath=checkpoints/trained/NYUv2_DFormer_Large/Base/Small/Tiny.pth

# SUNRGBD DFormers
# --config=local_configs.SUNRGBD.DFormer_Large/Base/Small/Tiny
# --continue_fpath=checkpoints/trained/SUNRGBD_DFormer_Large/Base/Small/Tiny.pth

