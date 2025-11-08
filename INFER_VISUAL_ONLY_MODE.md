# Infer.py 纯视觉模式

## 概述

**纯视觉模式**允许你在不开启文本引导的情况下，仅使用视觉端（RGB + 深度/模态数据）运行推理，并保存分割结果。

这对以下场景很有用：
- 调试纯视觉baseline
- 对比文本引导和纯视觉的性能
- 快速生成分割结果而不需要attention可视化
- 在没有文本标注的数据集上运行

---

## 快速开始

### 基础使用（纯视觉推理）

```bash
python utils/infer.py \
    --config local_configs/NYUDepthv2/DFormerv2_S.py \
    --gpus 1 \
    --continue_fpath checkpoints/best.pth \
    --save-predictions
```

**输出**：
```
infer_predictions/
├── RGB_0_pred.png
├── RGB_1_pred.png
├── RGB_2_pred.png
└── ...
```

### 指定输出路径

```bash
python utils/infer.py \
    --config local_configs/NYUDepthv2/DFormerv2_S.py \
    --gpus 1 \
    --continue_fpath checkpoints/best.pth \
    --save-predictions \
    --save_path ./my_predictions
```

---

## 两种模式对比

### 模式1: 纯视觉模式（新增）

**特点**：
- ❌ 不需要文本引导
- ❌ 不需要attention可视化
- ✅ 只保存分割结果
- ✅ 使用多尺度+翻转评估（最高精度）

**命令**：
```bash
python utils/infer.py \
    --config ... \
    --continue_fpath ... \
    --save-predictions
```

**输出结构（扁平）**：
```
<save_path>/
├── RGB_0_pred.png
├── RGB_1_pred.png
└── ...
```

**文件内容**：
- 每张图1个文件：彩色分割结果

---

### 模式2: Attention可视化模式（原有）

**特点**：
- ✅ 需要文本引导（enable_text_guidance=True）
- ✅ 保存原图、分割、attention
- ✅ 详细的token级可视化

**命令**：
```bash
python utils/infer.py \
    --config ... \
    --continue_fpath ... \
    --save-attention \
    --save_path ./output
```

**输出结构（文件夹）**：
```
output/
├── RGB_0/
│   ├── 00_original.png
│   ├── 01_segmentation.png
│   ├── 02_attn_wall.png
│   ├── 03_attn_floor.png
│   └── ...
└── RGB_1/
    └── ...
```

**文件内容**：
- 每张图多个文件：原图 + 分割 + N个attention

---

## 配置要求

### 纯视觉模式

**Config 文件要求**：
```python
# 可以设置为 False 或不设置
C.enable_text_guidance = False  # 或者完全不设置这个字段

# 其他正常配置
C.backbone = "DFormerv2_S"
C.num_classes = 40
# ...
```

### Attention模式

**Config 文件要求**：
```python
# 必须设置为 True
C.enable_text_guidance = True

# 必须配置文本相关参数
C.text_source = "imglabels"
C.label_txt_path = "datasets/NYUDepthv2/nyu40_labels.txt"
C.image_labels_json_path = "datasets/NYUDepthv2/out.json"
# ...
```

---

## 使用示例

### 示例1: 纯视觉baseline（完整验证集）

```bash
python utils/infer.py \
    --config local_configs/NYUDepthv2/DFormerv2_S.py \
    --gpus 1 \
    --continue_fpath checkpoints/visual_only.pth \
    --save-predictions \
    --save_path ./visual_baseline
```

**输出**：
- mIoU: XX.XX
- 654个分割结果图（NYU test set）

---

### 示例2: 对比文本引导 vs 纯视觉

```bash
# 1. 纯视觉
python utils/infer.py \
    --config local_configs/NYUDepthv2/DFormerv2_S.py \
    --gpus 1 \
    --continue_fpath checkpoints/model.pth \
    --save-predictions \
    --save_path ./comparison/visual_only

# 2. 文本引导
python utils/infer.py \
    --config local_configs/NYUDepthv2/DFormerv2_S.py \
    --gpus 1 \
    --continue_fpath checkpoints/model.pth \
    --save-attention \
    --save_path ./comparison/text_guided \
    --num-images 50  # 只可视化50张图
```

**对比**：
- `comparison/visual_only/` - 纯视觉的分割结果（扁平结构）
- `comparison/text_guided/` - 文本引导的详细可视化（文件夹结构）

---

### 示例3: 随机选择图片（纯视觉）

```bash
python utils/infer.py \
    --config local_configs/NYUDepthv2/DFormerv2_S.py \
    --gpus 1 \
    --continue_fpath checkpoints/model.pth \
    --save-predictions \
    --num-images 100 \
    --random-select
```

**输出**：
- 100张随机选择的图片的分割结果

---

### 示例4: 特定图片列表（纯视觉）

```bash
# 创建图片列表
cat > selected.txt << EOF
RGB/0.jpg
RGB/10.jpg
RGB/100.jpg
EOF

# 运行推理
python utils/infer.py \
    --config local_configs/NYUDepthv2/DFormerv2_S.py \
    --gpus 1 \
    --continue_fpath checkpoints/model.pth \
    --save-predictions \
    --image-paths selected.txt
```

---

## 参数说明

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `--save-predictions` | flag | 是 | 启用纯视觉模式，保存分割结果 |
| `--save_path` | str | 否 | 输出目录（默认：`./infer_predictions`） |
| `--num-images` | int | 否 | 处理图片数量（默认：全部） |
| `--random-select` | flag | 否 | 随机选择图片 |
| `--image-indices` | str | 否 | 指定图片索引（如："0,5,10"） |
| `--image-paths` | str | 否 | 图片路径列表文件 |

---

## 输出文件格式

### 文件命名

```
<save_path>/<image_identifier>_pred.png
```

**示例**：
- `RGB_0_pred.png`
- `RGB_test_123_pred.png`
- `datasets_NYUDepthv2_RGB_456_pred.png`

### 彩色编码

分割结果使用数据集特定的调色板：

| 数据集 | 调色板 |
|--------|--------|
| NYUDepthv2 | `utils/nyucmap.npy` |
| SUNRGBD | `utils/nyucmap.npy` |
| KITTI-360 | Cityscapes palette |
| EventScape | Cityscapes palette |
| MFNet | MFNet palette |

---

## 性能对比

### 评估模式

两种模式都使用相同的评估策略：
- 多尺度：`[0.5, 0.75, 1.0, 1.25, 1.5]`
- 水平翻转：`True`

因此mIoU结果应该一致（前提是模型支持纯视觉模式）。

### 速度对比

| 模式 | 相对速度 | 原因 |
|------|---------|------|
| 纯视觉 | 快 ⚡ | 只保存分割结果 |
| Attention | 慢 🐢 | 需要提取attention + 多个文件保存 |

**示例**（NYU 654张图）：
- 纯视觉：~5分钟
- Attention：~15分钟（取决于token数量）

---

## 常见问题

### Q1: 为什么需要纯视觉模式？

**A:** 几个主要原因：

1. **Baseline对比**：评估文本引导的提升效果
2. **调试**：排除文本引导的干扰，专注视觉端问题
3. **无文本数据集**：某些数据集没有文本标注
4. **快速预测**：只需要分割结果，不需要详细分析

---

### Q2: 纯视觉模式的mIoU会更低吗？

**A:** 取决于模型设计：

- 如果模型**严格依赖**文本引导（如SAM模块），纯视觉性能会下降
- 如果模型有**fallback机制**（没有文本时用视觉特征），性能下降较小
- 最好的设计是**可选文本引导**，两种模式都能工作

---

### Q3: 如何知道我的模型支持纯视觉模式？

**A:** 运行测试：

```bash
# 测试纯视觉
python utils/infer.py \
    --config <your_config> \
    --continue_fpath <checkpoint> \
    --save-predictions \
    --num-images 10

# 如果没有错误且mIoU合理（>10），说明支持
```

如果报错或mIoU接近0，说明模型强依赖文本引导。

---

### Q4: 可以同时输出纯视觉和文本引导的结果吗？

**A:** 需要运行两次：

```bash
# 第一次：纯视觉
python utils/infer.py \
    --config ... \
    --continue_fpath ... \
    --save-predictions \
    --save_path ./visual_only

# 第二次：文本引导
python utils/infer.py \
    --config ... \
    --continue_fpath ... \
    --save-attention \
    --save_path ./text_guided
```

---

### Q5: 纯视觉模式可以保存原图吗？

**A:** 目前不支持。纯视觉模式只保存分割结果。

如果需要原图，使用attention模式（会自动保存原图+分割+attention）。

或者你可以修改 `val_mm.py` 的保存逻辑添加原图保存。

---

### Q6: 输出格式为什么不同？

**A:** 设计理念：

- **纯视觉模式**：只有1个文件/图 → 扁平结构更简洁
- **Attention模式**：多个文件/图 → 文件夹结构更清晰

如果你需要统一格式，可以手动整理：

```bash
# 将扁平结构转为文件夹结构
for f in predictions/*_pred.png; do
    name=$(basename $f _pred.png)
    mkdir -p output/$name
    cp $f output/$name/01_segmentation.png
done
```

---

## 技术细节

### 实现原理

1. **参数处理**：
   - 检测 `--save-predictions` 标志
   - 如果没有 `--save_path`，设置默认路径 `./infer_predictions`

2. **评估流程**：
   - 使用 `evaluate_msf`（多尺度+翻转）
   - 传入 `save_dir` 参数触发保存

3. **保存逻辑**（在 `val_mm.py`）：
   - 每个batch后，将预测结果通过调色板映射
   - 保存为PNG文件

### 代码位置

- **参数定义**：`utils/infer.py:80`
- **默认路径设置**：`utils/infer.py:1003-1005`
- **保存逻辑**：`utils/val_mm.py:79-132`

---

## 总结

| 特性 | 纯视觉模式 | Attention模式 |
|------|-----------|--------------|
| **文本引导** | ❌ 不需要 | ✅ 需要 |
| **输出内容** | 分割结果 | 原图+分割+attention |
| **文件结构** | 扁平 | 文件夹 |
| **速度** | 快 ⚡ | 慢 🐢 |
| **用途** | Baseline/快速预测 | 详细分析/论文可视化 |
| **命令** | `--save-predictions` | `--save-attention` |

**推荐使用场景**：
- 只需要分割结果 → 纯视觉模式
- 需要详细分析 → Attention模式
- 对比实验 → 两种都用
