# SAM Attention 模式控制说明

## 📋 概述

DFormerv2 支持对 Encoder 和 Decoder 中的 SAM (Semantic Alignment Module) 分别控制注意力机制的两个维度：
1. **注意力类型**: 普通注意力 vs 余弦相似度注意力
2. **温度缩放**: 固定温度 vs 可学习温度

## 🎛️ 可控参数

### Decoder 参数
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--sam-decoder-use-cosine` | bool | `True` | 使用余弦相似度注意力 (L2归一化Q/K) |
| `--no-sam-decoder-use-cosine` | bool | - | 使用普通点积注意力 |
| `--sam-decoder-learnable-temp` | bool | `True` | 温度参数可学习 |
| `--no-sam-decoder-learnable-temp` | bool | - | 温度参数固定 |
| `--sam-decoder-logit-init` | float | `14.285714` (≈1/0.07) | 温度初始化值 |

### Encoder 参数
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--sam-encoder-use-cosine` | bool | `False` | 使用余弦相似度注意力 |
| `--no-sam-encoder-use-cosine` | bool | - | 使用普通点积注意力 (默认) |
| `--sam-encoder-learnable-temp` | bool | `False` | 温度参数可学习 |
| `--no-sam-encoder-learnable-temp` | bool | - | 温度参数固定 (默认) |
| `--sam-encoder-logit-init` | float | `1.0` | 温度初始化值 |

## 🧪 常用配置

### 1️⃣ 默认配置 (推荐)
**不需要传任何参数**，默认值为：
- **Encoder**: 固定温度 (`1.0`) + 普通注意力
- **Decoder**: 可学习温度 (`1/0.07`) + 余弦注意力

```bash
# 不传参，使用默认配置
./train.sh
```

### 2️⃣ Encoder 和 Decoder 都用余弦 + 可学习温度
适合文本-图像对齐要求高的任务：
```bash
./train.sh \
    --sam-encoder-use-cosine \
    --sam-encoder-learnable-temp \
    --sam-encoder-logit-init 14.285714
```

### 3️⃣ 都用普通注意力 + 固定温度
适合快速baseline或消融实验：
```bash
./train.sh \
    --no-sam-decoder-use-cosine \
    --no-sam-decoder-learnable-temp \
    --sam-decoder-logit-init 1.0
```

### 4️⃣ 交换 Encoder/Decoder 模式
消融实验：测试哪一侧更需要余弦注意力：
```bash
./train.sh \
    --no-sam-decoder-use-cosine \
    --no-sam-decoder-learnable-temp \
    --sam-decoder-logit-init 1.0 \
    --sam-encoder-use-cosine \
    --sam-encoder-learnable-temp \
    --sam-encoder-logit-init 14.285714
```

## 📝 参数说明

### 余弦注意力 vs 普通注意力
- **普通注意力**: `Q @ K.T / sqrt(d_k)`
- **余弦注意力**: `normalize(Q) @ normalize(K).T / sqrt(d_k)`
  - 更稳定，适合跨模态对齐 (文本-视觉)
  - 归一化后只关注方向，忽略幅度

### 可学习温度 vs 固定温度
- **固定温度**: `scale = exp(logit_init) / sqrt(d_k)`
- **可学习温度**: `scale = exp(clamp(learnable_logit)) / sqrt(d_k)`
  - 网络可以自适应调节注意力的锐度
  - 有梯度裁剪 (`clamp_logit=2.0`) 防止梯度爆炸

### 温度初始化值建议
- **普通注意力**: `1.0` (标准缩放)
- **余弦注意力**: `14.285714 ≈ 1/0.07` (常用于对比学习，如CLIP)

## 🔍 代码位置

- **SAM模块实现**: `models/blocks/semantic_alignment.py`
  - `_decoder_scale()` (L128-132): Decoder温度计算
  - `_encoder_scale()` (L134-138): Encoder温度计算
  - `forward()` (L141-218): Decoder注意力
  - `forward_ssa()` (L221-298): Encoder注意力

- **参数传递链路**:
  - CLI → `utils/train.py` (L69-94)
  - Config → `models/builder.py` (L61-66)
  - Builder → `models/encoders/DFormerv2.py`
  - Builder → `models/decoders/hsg_head.py`

## ✅ 向后兼容性

**完全向后兼容！** 默认参数值与之前的行为一致：
- 之前的实验不需要修改任何配置
- 新参数都是可选的
- 不传参数 = 保持原有行为

## 🚀 快速开始

参考 `train_examples.sh` 查看 6 种预设实验配置：
```bash
# 运行实验 1 (默认配置)
./train_examples.sh 1

# 运行实验 2 (都用余弦+可学习)
./train_examples.sh 2

# ... 依此类推
```
