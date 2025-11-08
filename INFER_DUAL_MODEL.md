# 双模型对比功能 (Dual-Model Comparison Mode)

## 概述

`--dual-model` 功能允许你在**一次运行**中，对**同一批图片**使用**两个不同的模型**进行推理对比：

- **模型1**：带文本引导的模型 → 生成 attention 可视化
- **模型2**：纯视觉的模型 → 生成分割预测

这解决了随机选择图片时，两次运行无法保证选中相同图片的问题。

## 核心优势

✅ **一次选择，两次推理**：随机选择的图片在两个模型中保持一致
✅ **不同模型对比**：可以加载两个不同 checkpoint 的模型
✅ **不同训练方式**：模型1用文本引导训练，模型2纯视觉训练
✅ **自动化流程**：无需手动记录 indices 或两次运行
✅ **显存优化**：模型1运行完后自动清理，再加载模型2

## 使用方法

### 基础用法

```bash
python utils/infer.py \
    --config local_configs/NYUDepthv2/DFormerv2_S.py \
    --gpus 1 \
    --continue_fpath checkpoints/model_text_guided.pth \
    --dual-model \
    --model2-path checkpoints/model_visual_only.pth \
    --num-images 10 \
    --random-select
```

### 完整示例（指定输出路径）

```bash
python utils/infer.py \
    --config local_configs/NYUDepthv2/DFormerv2_S.py \
    --gpus 1 \
    --continue_fpath checkpoints/text_guided_best.pth \
    --dual-model \
    --model2-path checkpoints/visual_only_best.pth \
    --save_path ./comparison/model1_text \
    --model2-save-path ./comparison/model2_visual \
    --num-images 20 \
    --random-select
```

### 使用指定图片索引

```bash
python utils/infer.py \
    --config local_configs/NYUDepthv2/DFormerv2_S.py \
    --gpus 1 \
    --continue_fpath checkpoints/model1.pth \
    --dual-model \
    --model2-path checkpoints/model2.pth \
    --image-indices "0,5,10,15,20"
```

## 参数说明

### 必需参数

| 参数 | 说明 |
|------|------|
| `--dual-model` | 启用双模型对比模式（bool） |
| `--model2-path` | 模型2的 checkpoint 路径（必须存在） |
| `--continue_fpath` | 模型1的 checkpoint 路径（原有参数） |

### 可选参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--save_path` | `./dual_model_model1_attention` | 模型1的输出目录 |
| `--model2-save-path` | `<save_path>_model2_visual` | 模型2的输出目录 |

### 图片选择参数（可组合）

| 参数 | 说明 |
|------|------|
| `--num-images N` | 选择 N 张图片 |
| `--random-select` | 随机选择（而非顺序选择） |
| `--image-indices "0,5,10"` | 指定具体索引 |
| `--image-paths file.txt` | 从文件读取路径列表 |

## 运行流程

当启用 `--dual-model` 时，程序会执行以下步骤：

```
1. 验证参数
   ├─ 检查 --model2-path 是否提供
   ├─ 检查两个 checkpoint 是否存在
   └─ 设置默认输出路径

2. 图片选择
   └─ 根据 --num-images, --random-select 等参数选择图片

3. 加载模型1（文本引导）
   ├─ 设置 config.enable_text_guidance = True
   ├─ 加载 --continue_fpath
   └─ 运行 attention 可视化

4. 清理模型1
   ├─ 删除模型1对象
   └─ 清空 GPU 缓存

5. 加载模型2（纯视觉）
   ├─ 设置 config.enable_text_guidance = False
   ├─ 加载 --model2-path
   └─ 使用相同的图片选择

6. 运行模型2评估
   ├─ 多尺度+翻转评估
   └─ 保存分割结果到 --model2-save-path

7. 输出对比总结
```

## 输出结构

### 模型1输出（文本引导 + Attention）

```
<save_path>/
├── RGB_0/
│   ├── 00_original.png
│   ├── 01_segmentation.png
│   ├── 02_attn_wall.png
│   ├── 03_attn_floor.png
│   └── ...
├── RGB_5/
│   └── ...
```

### 模型2输出（纯视觉 + Predictions）

```
<model2-save-path>/
├── RGB_0_pred.png
├── RGB_5_pred.png
├── RGB_10_pred.png
└── ...
```

## 日志示例

```
====================================================================================================
DUAL-MODEL COMPARISON MODE ENABLED
Model 1 (Text-Guided): checkpoints/text_guided_best.pth
  Output: ./comparison/model1_text
  Mode: Attention visualization (--save-attention)
Model 2 (Visual-Only): checkpoints/visual_only_best.pth
  Output: ./comparison/model2_visual
  Mode: Prediction only (--save-predictions)
====================================================================================================

Randomly selected 10 images from 654

====================================================================================================
STEP 1/2: Running Model 1 (Text-Guided + Attention Visualization)
Checkpoint: checkpoints/text_guided_best.pth
Output: ./comparison/model1_text
====================================================================================================

[... Model 1 运行日志 ...]

mIoU: 0.5834

====================================================================================================
STEP 2/2: Running Model 2 (Visual-Only + Prediction Saving)
Checkpoint: checkpoints/visual_only_best.pth
Output: ./comparison/model2_visual
====================================================================================================

Clearing Model 1 from GPU memory...
✓ Model 1 cleared

Reconfiguring for visual-only mode...
✓ config.enable_text_guidance = False

Loading Model 2...
✓ Model 2 loaded successfully

✓ Using same 10 images

[... Model 2 运行日志 ...]

====================================================================================================
MODEL 2 RESULTS (Visual-Only):
mIoU: 0.5621
mAcc: 0.7234
mF1: 0.6812
====================================================================================================

====================================================================================================
DUAL-MODEL COMPARISON COMPLETED
====================================================================================================
Model 1 (Text-Guided) outputs: ./comparison/model1_text
Model 2 (Visual-Only) outputs: ./comparison/model2_visual
====================================================================================================
```

## 注意事项

### 1. 模型配置要求

- **模型1**：应该是使用 `enable_text_guidance=True` 训练的模型
- **模型2**：应该是使用 `enable_text_guidance=False` 训练的模型
- 两个模型的其他配置（backbone, decoder 等）可以相同或不同

### 2. 显存管理

- 程序会在加载模型2前自动清理模型1，释放显存
- 如果显存充足，峰值占用约等于单个模型的显存 + 数据加载
- 不需要同时在显存中保存两个模型

### 3. 图片选择一致性

- 两个模型使用**完全相同**的图片选择
- 即使使用 `--random-select`，种子是固定的（基于首次选择）
- 可以安全地对比同一张图在两个模型下的表现

### 4. 输出格式差异

- **模型1**：使用文件夹结构（每张图一个文件夹，包含多个文件）
- **模型2**：使用扁平结构（每张图一个 `_pred.png` 文件）
- 这是因为两种模式的用途不同（详细分析 vs 快速对比）

## 应用场景

### 1. 消融实验

对比文本引导对模型性能的影响：

```bash
python utils/infer.py \
    --dual-model \
    --continue_fpath checkpoints/with_text.pth \
    --model2-path checkpoints/without_text.pth \
    --num-images 50 \
    --random-select
```

### 2. 模型演进对比

对比训练过程中的不同 checkpoint：

```bash
python utils/infer.py \
    --dual-model \
    --continue_fpath checkpoints/epoch_100.pth \
    --model2-path checkpoints/epoch_50.pth \
    --image-indices "0,10,20,30"
```

### 3. 架构对比

对比不同架构的模型（需要使用相同的 config 或手动修改）：

```bash
# 假设两个模型使用相同的 config 格式
python utils/infer.py \
    --dual-model \
    --continue_fpath checkpoints/dformer_large.pth \
    --model2-path checkpoints/dformer_small.pth \
    --num-images 100
```

## 常见问题

### Q1: 两个模型必须使用相同的 config 吗？

**A:** 是的，当前实现中两个模型共用同一个 config 文件。唯一的区别是 `enable_text_guidance` 会自动设置（模型1=True，模型2=False）。

### Q2: 可以两个模型都用 attention 模式吗？

**A:** 当前设计中，`--dual-model` 强制设置：
- 模型1 = attention 模式
- 模型2 = prediction 模式

如果需要两个模型都用 attention，建议分两次运行，使用 `--image-indices` 指定相同的图片。

### Q3: 显存不足怎么办？

**A:** 程序已经优化了显存使用：
1. 模型1运行完后会**立即删除**并清空缓存
2. 模型2加载时，显存中只有模型2
3. 如果仍然不足，可以：
   - 减少 `--num-images` 的数量
   - 使用更小的 batch size（需要修改 config）

### Q4: 如何确保两次运行选中相同的图片？

**A:** 这正是 `--dual-model` 的核心价值！程序内部会保存首次选择的 indices，模型2会使用完全相同的图片集合。

### Q5: 可以用于分布式训练吗？

**A:** 可以。代码已经支持 `engine.distributed` 模式，会正确处理多 GPU 场景。

## 与其他模式的对比

| 特性 | `--dual-model` | 两次运行 `--save-attention` + `--save-predictions` |
|------|----------------|---------------------------------------------------|
| 图片选择一致性 | ✅ 保证相同 | ❌ 随机选择时无法保证 |
| 手动操作 | ✅ 一次运行 | ❌ 需要两次运行 |
| 模型差异 | ✅ 支持不同 checkpoint | ✅ 支持不同 checkpoint |
| 显存占用 | ✅ 串行加载，占用低 | ⚠️ 分两次，总占用相同 |
| 灵活性 | ⚠️ 固定为 attention + prediction | ✅ 可以自由选择模式 |

## 技术实现细节

### 关键设计

1. **参数验证**（line 864-893）
   - 检查 `--model2-path` 是否存在
   - 自动设置默认输出路径
   - 强制启用 `--save-attention` 和禁用 `--save-predictions`

2. **模型1运行**（line 1062-1124）
   - 使用已加载的模型1
   - 运行 `evaluate_with_attention()`

3. **显存清理**（line 1135-1139）
   ```python
   del model
   torch.cuda.empty_cache()
   ```

4. **模型2加载**（line 1146-1169）
   - 重新配置 `config.enable_text_guidance = False`
   - 创建新模型并加载权重
   - 移到 GPU

5. **复用数据集**（line 1173-1175）
   - 直接使用已过滤的 `val_loader`
   - 无需重新计算 indices

6. **模型2评估**（line 1182-1231）
   - 运行 `evaluate_msf()` 并保存到 `--model2-save-path`

### 代码位置

- 参数定义：`utils/infer.py:82-87`
- 参数验证：`utils/infer.py:864-893`
- 双模型逻辑：`utils/infer.py:1045-1238`

## 相关文档

- [INFER_OUTPUT_STRUCTURE.md](./INFER_OUTPUT_STRUCTURE.md) - 模型1的输出结构
- [INFER_VISUAL_ONLY_MODE.md](./INFER_VISUAL_ONLY_MODE.md) - 模型2的纯视觉模式
- [INFER_ADVANCED_FEATURES.md](./INFER_ADVANCED_FEATURES.md) - Attention 可视化高级功能

## 版本历史

- **2025-01-08**: 初始实现，支持双模型对比功能
