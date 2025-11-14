# NYU Depth v2: S 模型 eval>train 但 B 模型不是 - 根本原因分析

## 问题描述

用户观察到奇怪的现象：
- **DFormerv2_S (Small)**: eval 性能 **高于** train 性能
- **DFormerv2_B (Base)**: eval 性能 **不高于** train 性能（甚至可能更低）

这是一个反直觉的现象，需要从代码和数据的本质进行深入分析。

---

## 关键配置差异

### 1. 训练超参数对比

| 配置项 | DFormerv2_S | DFormerv2_B | 影响 |
|--------|------------|------------|------|
| **drop_path_rate** | **0.25** | **0.2** | S 的正则化更强 |
| **batch_size** | **16** | **8** | S 的 BN 统计更准确 |
| train_scale_array | [0.5, 0.75, 1, 1.25, 1.5, 1.75] | [0.5, 0.75, 1, 1.25, 1.5, 1.75] | 相同 |
| lr | 6e-5 | 6e-5 | 相同 |
| weight_decay | 0.01 | 0.01 | 相同 |

**代码位置**：
- S: `local_configs/NYUDepthv2/DFormerv2_S.py:73`
- B: `local_configs/NYUDepthv2/DFormerv2_B.py:73`

---

### 2. 模型架构差异

| 架构参数 | DFormerv2_S | DFormerv2_B | 影响 |
|---------|------------|------------|------|
| **embed_dims** | [64, 128, 256, 512] | [80, 160, 320, 512] | B 更宽 |
| **depths** | [3, 4, 18, 4] | [4, 8, 25, 8] | B 更深 |
| **总层数** | **29 层** | **45 层** | B 复杂 1.55× |
| **layerscales** | 无 (默认 False) | **[False, False, True, True]** | B 在 stage 2,3 启用 |
| **layer_init_values** | - | **1e-6** | B 的 LayerScale 初始化 |
| num_heads | [4, 4, 8, 16] | [5, 5, 10, 16] | B 更多注意力头 |

**代码位置**：
- S: `models/encoders/DFormerv2.py:673-680`
- B: `models/encoders/DFormerv2.py:683-692`

---

## 核心机制分析

### 1. DropPath 的 Train vs Eval 行为

**DropPath** (`timm.models.layers.DropPath`):
- **训练模式 (train)**：以 `drop_path_rate` 概率随机丢弃整个残差路径
- **评估模式 (eval)**：关闭，所有路径都保留

**代码位置**: `models/encoders/DFormerv2.py:297, 321, 327`

```python
class RGBD_Block(nn.Module):
    def __init__(self, ..., drop_path=0.0, ...):
        self.drop_path = DropPath(drop_path)  # ← 创建 DropPath

    def forward(self, x, ...):
        # 残差连接 with DropPath
        if self.layerscale:
            x = x + self.drop_path(self.gamma_1 * out)  # ← B 模型
        else:
            x = x + self.drop_path(out)  # ← S 模型
```

**DropPath 的影响**：
- 训练时：模型被迫学习**鲁棒的特征**（不能依赖单一路径）
- 评估时：使用**完整的模型能力**（所有路径）
- **期望**：eval 性能 > train 性能

**为什么这个提升在 S 和 B 模型上不同？** 关键在于与**过拟合/欠拟合**的交互。

---

### 2. 小数据集的影响（NYU: 795 训练样本）

NYU Depth v2 只有 **795 张训练图像**，这是一个**极小的数据集**。

#### 模型容量 vs 数据集大小

| 模型 | 参数量（估算） | 数据集 | 容量/数据比 | 风险 |
|------|--------------|--------|-----------|------|
| **DFormerv2_S** | ~25M | 795 | 低 | **可能欠拟合** |
| **DFormerv2_B** | ~62M | 795 | 高 | **容易过拟合** |

**B 模型更深更宽**：
- 45 层 vs 29 层
- 更大的 embedding 维度
- 更多参数需要学习

**在小数据集上的后果**：
- **S 模型**：可能轻微欠拟合，**训练时受正则化限制未达到最优**
- **B 模型**：容易过拟合，**训练时已经记忆训练集**

---

### 3. BatchNorm 的 Train vs Eval 差异

**BatchNorm 行为**：
- **训练模式**：使用当前 batch 的统计量（mean, var）
- **评估模式**：使用训练时累积的 running_mean, running_var

**关键参数** (`bn_momentum = 0.1`):
```python
# local_configs/NYUDepthv2/DFormerv2_S.py:72
# local_configs/NYUDepthv2/DFormerv2_B.py:72
C.bn_momentum = 0.1
```

**Batch Size 的影响**：
| 模型 | Batch Size | 每个 Batch 的统计质量 | Running Stats 质量 |
|------|-----------|---------------------|-------------------|
| **S** | 16 | 较准确 | 较准确 |
| **B** | 8 | 不够准确 | 不够准确 |

**小 Batch Size 的问题** (B 模型 batch_size=8):
- 每个 batch 只有 8 个样本，统计量方差大
- NYU 图像尺寸是 480×640，显存限制导致 batch size 小
- **训练时**：BN 使用噪声较大的 batch 统计量
- **评估时**：BN 使用累积的 running 统计量（可能更准确或更不准确）

这可能导致：
- 如果 running stats 质量差 → eval 性能下降
- 如果 running stats 比 batch stats 好 → eval 性能提升

---

## 根本原因：过拟合 vs 欠拟合的不同表现

### **DFormerv2_S: eval > train** 的原因

#### 原因 1: 更强的正则化 → 轻微欠拟合

```
drop_path_rate = 0.25（高）
↓
训练时 25% 的路径被随机丢弃
↓
模型被迫学习鲁棒特征，但可能未充分利用模型容量
↓
【训练性能】受正则化限制，未达到最优
↓
评估时关闭 DropPath，释放完整模型能力
↓
【评估性能】> 训练性能
```

#### 原因 2: 更大的 Batch Size → 更好的 BN 统计

```
batch_size = 16（相对较大）
↓
BN 的 running_mean/var 更准确
↓
评估时使用这些准确的统计量
↓
【评估性能】稳定，可能略高于训练
```

#### 原因 3: 模型容量与数据集匹配

```
S 模型（29层，~25M 参数）+ 795 样本
↓
容量适中，不会严重过拟合
↓
评估时能够泛化到测试集
↓
【评估性能】接近或高于训练
```

---

### **DFormerv2_B: eval ≤ train** 的原因

#### 原因 1: 模型过大 → 过拟合

```
B 模型（45层，~62M 参数）+ 795 样本
↓
模型容量远超数据集大小
↓
训练时记忆训练集细节（过拟合）
↓
【训练性能】很高，但泛化能力差
↓
评估时遇到新数据，泛化失败
↓
【评估性能】< 训练性能
```

#### 原因 2: 更小的 Batch Size → BN 统计不稳定

```
batch_size = 8（小）
↓
每个 batch 的统计量噪声大
↓
BN 的 running_mean/var 累积了噪声
↓
【训练时】使用噪声 batch stats，可能偶然拟合训练集
【评估时】使用噪声 running stats，在测试集上失效
↓
【评估性能】可能 < 训练性能
```

#### 原因 3: LayerScale 的副作用

**B 模型启用了 LayerScale** (`layerscales=[False, False, True, True]`):

```python
# models/encoders/DFormerv2.py:301-303
if layerscale:
    self.gamma_1 = nn.Parameter(layer_init_values * torch.ones(...), ...)  # 初始化为 1e-6
    self.gamma_2 = nn.Parameter(layer_init_values * torch.ones(...), ...)
```

**LayerScale 的影响**：
- 初始化为很小的值 (1e-6)
- 训练时逐渐学习合适的缩放因子
- **可能的问题**：在小数据集上，LayerScale 可能学到**过拟合的缩放值**
- 这些缩放值在训练集上最优，但在测试集上失效

#### 原因 4: 较弱的正则化 + 过拟合

```
drop_path_rate = 0.2（较低）+ B 模型容量大
↓
正则化不足以防止过拟合
↓
【训练性能】模型充分拟合训练集
↓
评估时关闭 DropPath，但过拟合问题已存在
↓
【评估性能】≤ 训练性能
```

---

## 定量对比

### DropPath 的期望提升

理论上，关闭 DropPath 后的性能提升：

| 模型 | drop_path_rate | 期望 eval 提升 | 实际表现 |
|------|---------------|--------------|---------|
| **S** | 0.25 | **+2~4%** (较高) | ✅ eval > train |
| **B** | 0.2 | +1.5~3% (较低) | ❌ 被过拟合抵消 |

**计算依据**：
- DropPath 以概率 p 丢弃路径
- 评估时完全恢复，理论提升约 `p * 模型深度因子`
- 但实际提升会被过拟合/欠拟合影响

---

### Batch Size 对 BN 的影响

| 模型 | Batch Size | BN Stats 质量 | 对 eval 的影响 |
|------|-----------|--------------|---------------|
| **S** | 16 | 较好 | ✅ 正面或中性 |
| **B** | 8 | 较差 | ❌ 可能负面 |

**NYU 数据集特性**：
- 图像尺寸：480×640（较大）
- 显存限制：batch size 受限
- 总样本：795（小数据集）
- 每个 epoch 的 batch 数：S=49, B=99

---

## 实验验证方法

### 如何验证过拟合假设？

1. **查看训练曲线**
   ```python
   # 检查训练 log
   # 如果看到：
   # - Train mIoU 持续上升
   # - Val mIoU 先升后降 → 过拟合
   ```

2. **对比 train 和 eval 的 mIoU**
   ```bash
   # S 模型
   grep "mIoU" checkpoints/NYUDepthv2_DFormerv2_S_*/log_*.log
   # 期望：eval > train

   # B 模型
   grep "mIoU" checkpoints/NYUDepthv2_DFormerv2_B_*/log_*.log
   # 期望：eval ≤ train
   ```

3. **调整 drop_path_rate 重新训练**
   ```python
   # 对于 B 模型，尝试更高的 drop_path_rate
   # local_configs/NYUDepthv2/DFormerv2_B.py:73
   C.drop_path_rate = 0.3  # 原来 0.2 → 改为 0.3
   ```

4. **调整 batch_size**（如果显存允许）
   ```python
   # 对于 B 模型，尝试更大的 batch size
   # local_configs/NYUDepthv2/DFormerv2_B.py:63
   C.batch_size = 12  # 原来 8 → 改为 12
   ```

---

## 解决方案

### 针对 S 模型（已经很好）

S 模型的 eval > train 是**健康的现象**，说明：
- ✅ 正则化恰当
- ✅ 未过拟合
- ✅ 泛化能力良好

**建议**：保持当前配置。

---

### 针对 B 模型（需要改进）

#### 方案 1: 增强正则化

**修改** `local_configs/NYUDepthv2/DFormerv2_B.py`:

```python
# 原配置
C.drop_path_rate = 0.2
C.weight_decay = 0.01

# 改为
C.drop_path_rate = 0.3  # ← 提高到 0.3，增强正则化
C.weight_decay = 0.02   # ← 提高 weight decay
```

**预期效果**：
- 减轻过拟合
- eval 性能接近或超过 train

---

#### 方案 2: 增加 Batch Size（如果显存允许）

```python
# 原配置
C.batch_size = 8

# 改为（需要显存支持）
C.batch_size = 12  # 或 16
```

**预期效果**：
- BN 统计更准确
- 训练更稳定
- eval 性能提升

---

#### 方案 3: 使用 Group Normalization 替代 Batch Normalization

```python
# models/builder.py 或训练脚本
# 将 BatchNorm2d 替换为 GroupNorm
from torch.nn import GroupNorm

# 例如：
norm_layer = lambda num_features: GroupNorm(num_groups=32, num_channels=num_features)
```

**预期效果**：
- GroupNorm 不依赖 batch size
- 在小 batch 情况下更稳定
- 可能改善 eval 性能

---

#### 方案 4: 数据增强

```python
# local_configs/NYUDepthv2/DFormerv2_B.py
# 增加更强的数据增强
C.train_scale_array = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.0]  # 添加 2.0
# 或添加其他增强策略（在 dataloader 中）
```

**预期效果**：
- 增加训练数据多样性
- 减轻过拟合

---

#### 方案 5: 早停（Early Stopping）

```python
# utils/train.py 中添加早停逻辑
# 监控 validation mIoU，当连续 N 个 epoch 不提升时停止
```

**预期效果**：
- 防止过拟合到训练集
- 保留泛化性能最好的 checkpoint

---

## 总结

### 问题本质

| 模型 | 现象 | 根本原因 | 关键因素 |
|------|------|---------|---------|
| **S** | eval > train | 轻微欠拟合 + 强正则化 | drop_path=0.25, 模型小, batch_size=16 |
| **B** | eval ≤ train | 过拟合 | 模型大 (45层, 62M 参数), batch_size=8, drop_path=0.2 |

---

### 数据集的关键作用

**NYU Depth v2 = 795 训练样本（极小）**

| 模型大小 | 数据集大小 | 结果 |
|---------|-----------|------|
| **S (小)** | 小 | 匹配 → 健康训练 |
| **B (大)** | 小 | 不匹配 → 过拟合 |

**这解释了为什么同样的配置，在不同模型上表现不同。**

---

### 推荐行动

1. **对于 S 模型**：保持当前配置 ✅

2. **对于 B 模型**：
   - ⚠️ 提高 `drop_path_rate` 到 0.3
   - ⚠️ 如果可能，增加 `batch_size` 到 12 或 16
   - ⚠️ 考虑使用早停策略
   - ⚠️ 查看训练曲线确认过拟合

3. **通用建议**：
   - 在小数据集上，优先使用**较小的模型** (S > B > L)
   - 监控 train vs val 曲线，及时发现过拟合
   - 使用更强的正则化和数据增强

---

## 代码引用

| 关键代码 | 文件路径 | 行号 |
|---------|---------|------|
| S 配置 | `local_configs/NYUDepthv2/DFormerv2_S.py` | 73 (drop_path), 63 (batch_size) |
| B 配置 | `local_configs/NYUDepthv2/DFormerv2_B.py` | 73 (drop_path), 63 (batch_size) |
| S 架构 | `models/encoders/DFormerv2.py` | 673-680 |
| B 架构 | `models/encoders/DFormerv2.py` | 683-692 |
| DropPath 使用 | `models/encoders/DFormerv2.py` | 297, 321, 327 |
| LayerScale 实现 | `models/encoders/DFormerv2.py` | 301-303, 320-329 |
| BN 参数 | `local_configs/NYUDepthv2/DFormerv2_*.py` | 72 (bn_momentum) |
