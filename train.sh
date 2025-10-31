GPUS=2
NNODES=1
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29759} #158
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

CACHE_DIR="/dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/huggingface_cache"
mkdir -p ${CACHE_DIR}
export HF_HOME=${CACHE_DIR}
echo "Hugging Face cache has been set to: ${HF_HOME}"

export CUDA_VISIBLE_DEVICES="0,1"
export TORCHDYNAMO_VERBOSE=1

PYTHONPATH="$(dirname $0)/..":"$(dirname $0)":$PYTHONPATH \
    torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    utils/train.py \
    --config=local_configs.NYUDepthv2.DFormerv2_S --gpus=$GPUS \
    --text-source imglabels \
    --text-encoder clip \
    --sam-enc-stages 1,2,3 \
    --sam-dec-stages 1,2,3 \
    --no-sliding \
    --no-compile \
    --syncbn \
    --mst \
    --compile_mode="default" \
    --no-amp \
    --val_amp \
    --pad_SUNRGBD \
    --no-use_seed \
    --superpower \

# --text-source imglabels \
# 文本来源：labels / captions / both / imglabels
# --text-encoder jinaclip \
# 选择文本编码器：clip / jinaclip
# Encoder 侧在哪些 stage 启用 SAM（按 0/1/2/3）
# --sam-enc-stages 0,2 \
# Decoder 侧在哪些 stage 启用 SAM（按 1/2/3）
# --sam-dec-stages 1,3 \
# 例子: 1，3的时候 decoder启用了对于H/8和H/32的stage输出的decoder内的SAM
# --superpower \ 代指block内部的结构



# config for DFormers on NYUDepthv2
# local_configs.NYUDepthv2.DFormer_Large
# local_configs.NYUDepthv2.DFormer_Base
# local_configs.NYUDepthv2.DFormer_Small
# local_configs.NYUDepthv2.DFormer_Tiny
# local_configs.NYUDepthv2.DFormer_v2_S
# local_configs.NYUDepthv2.DFormer_v2_B
# local_configs.NYUDepthv2.DFormer_v2_L

# config for DFormers on SUNRGBD
# local_configs.SUNRGBD.DFormer_Large
# local_configs.SUNRGBD.DFormer_Base
# local_configs.SUNRGBD.DFormer_Small
# local_configs.SUNRGBD.DFormer_Tiny
# local_configs.SUNRGBD.DFormer_v2_S
# local_configs.SUNRGBD.DFormer_v2_B
# local_configs.SUNRGBD.DFormer_v2_L
