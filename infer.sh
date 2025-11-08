# CUDA_VISIBLE_DEVICES=0,1
# config -> which model config
# continue_fpath -> the trained pth path
GPUS=1
NNODES=1
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29928}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

CACHE_DIR="/dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/huggingface_cache"
mkdir -p ${CACHE_DIR}
export HF_HOME=${CACHE_DIR}
echo "Hugging Face cache has been set to: ${HF_HOME}"

export TEXT_EMBED_CACHE="$(pwd)/datasets/.text_cache"
mkdir -p "$TEXT_EMBED_CACHE"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    utils/infer.py \
    --config=local_configs.NYUDepthv2.DFormerv2_S \
    --continue_fpath=checkpoints/NYU_S_57.53_seed53384_clip_ori/epoch-251_miou_57.49.pth \
    --save_path output \
    --gpus=$GPUS \
    --text-source imglabels \
    --image-labels-json-path datasets/NYUDepthv2/out.json \
    --text-encoder clip \
    --sam-enc-stages 1,2,3 \
    --sam-dec-stages 1,2,3 \
    --superpower \
    --save-attention \
    --vis-stage dec \
    --vis-stage-idx 2 \
    --vis-block-idx -1 \
    --num-images 3

#   --random-sample \
# choose the dataset and DFormer for evaluating

# NYUv2 DFormers
# --config=local_configs.NYUDepthv2.DFormer_Large/Base/Small/Tiny
# --continue_fpath=checkpoints/trained/NYUv2_DFormer_Large/Base/Small/Tiny.pth

# SUNRGBD DFormers
# --config=local_configs.SUNRGBD.DFormer_Large/Base/Small/Tiny
# --continue_fpath=checkpoints/trained/SUNRGBD_DFormer_Large/Base/Small/Tiny.pth

