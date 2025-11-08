# GT 对比功能与三模式集成 (GT Comparison & Triple-Mode Integration)

## 概述

新增的 `--save-gt` 和增强的 `--dual-model` 功能，让你可以在**一次推理**中生成完整的对比材料：

✅ **GT 标签**（Ground Truth）
✅ **模型1 预测**（文本引导 + Attention 可视化）
✅ **模型2 预测**（纯视觉）

这三种结果会被**集成到同一个文件夹**中，方便直接对比和论文可视化！

## 核心优势

🎯 **一次运行，三种结果**：GT + 文本引导 + 纯视觉
📁 **集成输出结构**：所有结果在同一个文件夹，易于对比
🔢 **智能编号**：自动排序，浏览器中顺序查看
🎨 **统一配色**：GT 和预测使用相同的 palette，颜色一致
📊 **完整指标**：同时输出两个模型的 mIoU、mAcc、mF1

## 使用方法

### 完整对比模式（推荐）

```bash
python utils/infer.py \
    --config local_configs/NYUDepthv2/DFormerv2_S.py \
    --gpus 1 \
    --continue_fpath checkpoints/text_guided.pth \
    --dual-model \
    --model2-path checkpoints/visual_only.pth \
    --save_path ./comparison_results \
    --num-images 10 \
    --random-select
```

**输出结构**：

```
comparison_results/
├── RGB_0/
│   ├── 00_original.png             # 原图
│   ├── 01_GT.png                   # Ground Truth 标签
│   ├── 02_pred_model1_text.png     # 模型1预测（文本引导）
│   ├── 03_pred_model2_visual.png   # 模型2预测（纯视觉）
│   ├── 04_attn_wall.png            # Attention map: wall
│   ├── 05_attn_floor.png           # Attention map: floor
│   ├── 06_attn_ceiling.png         # Attention map: ceiling
│   └── ...                         # 其他 attention maps
├── RGB_5/
│   └── ...（相同结构）
└── ...
```

### 只保存 GT（不使用双模型）

```bash
python utils/infer.py \
    --config local_configs/NYUDepthv2/DFormerv2_S.py \
    --gpus 1 \
    --continue_fpath checkpoints/model.pth \
    --save-attention \
    --save-gt \
    --save_path ./results_with_gt \
    --num-images 5
```

**输出结构**：

```
results_with_gt/
├── RGB_0/
│   ├── 00_original.png
│   ├── 01_GT.png
│   ├── 02_segmentation.png         # 模型预测
│   ├── 03_attn_wall.png
│   └── ...
```

## 参数说明

### 新增参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--save-gt` | bool | False | 保存 Ground Truth 标签（自动颜色映射） |

### 与 --dual-model 组合

当启用 `--dual-model` 时：
- ✅ 自动启用 `--save-gt`（无需手动指定）
- ✅ 自动集成输出（GT + Model1 + Model2 在同一文件夹）
- ✅ 自动清理临时文件

## 文件编号规则

文件编号会根据启用的功能动态调整：

| 编号 | 文件名 | 条件 |
|------|--------|------|
| 00 | original.png | 总是生成 |
| 01 | GT.png | 如果 `--save-gt` |
| 02 | pred_model1_text.png | 如果 `--dual-model` |
|    | segmentation.png | 否则（单模型） |
| 03 | pred_model2_visual.png | 如果 `--dual-model` |
| 04+ | attn_*.png | Attention maps |

**示例**：

- **单模型 + GT**：
  ```
  00_original.png
  01_GT.png
  02_segmentation.png
  03_attn_wall.png
  04_attn_floor.png
  ```

- **双模型（自动包含GT）**：
  ```
  00_original.png
  01_GT.png
  02_pred_model1_text.png
  03_pred_model2_visual.png
  04_attn_wall.png
  05_attn_floor.png
  ```

- **单模型，无GT**：
  ```
  00_original.png
  01_segmentation.png
  02_attn_wall.png
  03_attn_floor.png
  ```

## GT 颜色映射

GT 标签使用与预测相同的 palette 进行颜色映射，确保颜色一致性：

- **NYUDepthv2/SUNRGBD**：使用 `utils/nyucmap.npy`
- **Cityscapes/其他**：使用 Cityscapes 标准配色

这样可以直接对比 GT 和预测，颜色代表的类别完全一致。

## 日志输出示例

### 双模型集成模式

```
====================================================================================================
DUAL-MODEL COMPARISON MODE ENABLED
Model 1 (Text-Guided): checkpoints/text_guided.pth
Model 2 (Visual-Only): checkpoints/visual_only.pth
Output: ./comparison_results
  Mode: Integrated (GT + Model1 + Model2 + Attention in same folders)
====================================================================================================

Randomly selected 10 images from 654

====================================================================================================
STEP 1/2: Running Model 1 (Text-Guided + Attention Visualization)
====================================================================================================
[... Model 1 运行 ...]
mIoU: 0.5834

====================================================================================================
STEP 2/2: Running Model 2 (Visual-Only + Prediction Saving)
====================================================================================================
Clearing Model 1 from GPU memory...
✓ Model 1 cleared

Reconfiguring for visual-only mode...
✓ config.enable_text_guidance = False

Loading Model 2...
✓ Model 2 loaded successfully

Running multi-scale+flip evaluation for Model 2...
  Integrated mode: using temporary directory ./comparison_results_model2_temp

====================================================================================================
MODEL 2 RESULTS (Visual-Only):
mIoU: 0.5621
mAcc: 0.7234
mF1: 0.6812
====================================================================================================

================================================================================
Merging Model 2 predictions into Model 1 folders...
  ✓ Merged RGB_0
  ✓ Merged RGB_5
  ✓ Merged RGB_10
  ...
✓ Cleaned up temporary directory: ./comparison_results_model2_temp
================================================================================

====================================================================================================
DUAL-MODEL COMPARISON COMPLETED
====================================================================================================
Integrated outputs (GT + Model1 + Model2 + Attention): ./comparison_results
  Each image folder contains:
    - 00_original.png
    - 01_GT.png
    - 02_pred_model1_text.png
    - 03_pred_model2_visual.png
    - 04+ attention maps
====================================================================================================
```

## 应用场景

### 1. 论文可视化

一次生成所有对比材料：

```bash
python utils/infer.py \
    --dual-model \
    --continue_fpath checkpoints/ours_with_text.pth \
    --model2-path checkpoints/ours_without_text.pth \
    --save_path ./paper_figures \
    --image-indices "0,10,20,30,40"  # 选择特定的图片
```

在 `./paper_figures/RGB_X/` 中直接获取所有需要的图片，无需后处理。

### 2. 定性分析

随机选择样本，对比三种结果：

```bash
python utils/infer.py \
    --dual-model \
    --continue_fpath checkpoints/model_a.pth \
    --model2-path checkpoints/model_b.pth \
    --save_path ./qualitative_analysis \
    --num-images 50 \
    --random-select
```

浏览器打开文件夹，顺序查看：原图 → GT → 模型1 → 模型2 → Attention

### 3. 消融实验

对比文本引导的影响：

```bash
python utils/infer.py \
    --dual-model \
    --continue_fpath checkpoints/with_text_guidance.pth \
    --model2-path checkpoints/without_text_guidance.pth \
    --save_path ./ablation_text_guidance \
    --num-images 100
```

可以直观看到每个样本上，文本引导带来的改进。

### 4. 错误分析

找出预测与 GT 差异大的样本：

```bash
python utils/infer.py \
    --dual-model \
    --continue_fpath checkpoints/model.pth \
    --model2-path checkpoints/baseline.pth \
    --save_path ./error_analysis \
    --num-images 200
```

浏览所有结果，找出问题样本，分析原因。

## 技术实现细节

### GT 读取与颜色映射

```python
# 从 dataloader 直接获取 GT label（无需额外读文件）
label_np = labels[b].cpu().numpy().astype(np.uint8)

# 使用相同的 palette 进行颜色映射
gt_colored = palette[label_np]

# 保存
gt_path = os.path.join(img_output_dir, f"{file_counter:02d}_GT.png")
plt.imsave(gt_path, gt_colored)
```

### 集成模式文件合并

双模型集成模式的实现：

1. 模型1运行，保存到 `args.save_path`
2. 模型2运行，保存到临时目录 `args.save_path + "_model2_temp"`
3. 后处理：遍历临时目录，将每个 `*_pred.png` 移动到对应的模型1文件夹
4. 重命名为 `03_pred_model2_visual.png`
5. 删除临时目录

```python
# 伪代码
for each file in temp_dir:
    base_name = extract_base_name(file)
    model1_folder = find_model1_folder(base_name)
    copy_and_rename(file, model1_folder, "03_pred_model2_visual.png")
cleanup(temp_dir)
```

### 动态文件编号

使用 `file_counter` 动态分配编号：

```python
file_counter = 1

if save_gt:
    save_gt_as(f"{file_counter:02d}_GT.png")
    file_counter += 1

if model_name:
    save_pred_as(f"{file_counter:02d}_pred_{model_name}.png")
else:
    save_pred_as(f"{file_counter:02d}_segmentation.png")
file_counter += 1

# Attention maps start from file_counter
for token in tokens:
    save_attn_as(f"{file_counter:02d}_attn_{token}.png")
    file_counter += 1
```

## 与其他模式的对比

| 特性 | `--save-gt` 单独使用 | `--dual-model`（自动包含GT） |
|------|---------------------|----------------------------|
| GT 标签 | ✅ | ✅ |
| 模型1预测 | ✅ | ✅ |
| 模型2预测 | ❌ | ✅ |
| Attention maps | ✅（如果启用） | ✅ |
| 输出结构 | 单一文件夹 | 集成文件夹（或独立） |
| 适用场景 | 单模型分析 | 多模型对比 |

## 常见问题

### Q1: 是否可以只保存 GT，不保存 attention？

**A:** 可以。不使用 `--save-attention` 即可，但这样只会运行标准评估，不会保存任何可视化。建议使用：

```bash
python utils/infer.py \
    --save-predictions \
    --save-gt \
    --save_path ./gt_and_pred
```

但注意 `--save-predictions` 是扁平结构，不是文件夹结构。

### Q2: GT 颜色为什么和我的可视化工具不一样？

**A:** 因为使用了数据集特定的 palette。确保你使用的是正确的颜色映射文件（如 `nyucmap.npy`）。

### Q3: 双模型集成模式可以禁用吗？

**A:** 可以。显式指定 `--model2-save-path` 即可使用独立输出：

```bash
python utils/infer.py \
    --dual-model \
    --continue_fpath checkpoints/m1.pth \
    --model2-path checkpoints/m2.pth \
    --save_path ./model1_output \
    --model2-save-path ./model2_output
```

这样模型1和模型2的输出会分开保存。

### Q4: 文件编号会自动调整吗？

**A:** 是的。编号完全动态，取决于启用的功能：
- 有 GT：从 01 开始
- 无 GT：从 01 开始（但是 segmentation）
- Attention 总是从前面的编号继续

### Q5: 集成模式下，模型2的临时文件会自动清理吗？

**A:** 是的。合并完成后会自动删除临时目录，无需手动清理。

## 代码位置

- 参数定义：`utils/infer.py:82`（`--save-gt`）
- GT 保存逻辑：`utils/infer.py:685-692`
- 动态编号：`utils/infer.py:683-703`
- 模型2集成：`utils/infer.py:1265-1295`

## 相关文档

- [INFER_DUAL_MODEL.md](./INFER_DUAL_MODEL.md) - 双模型对比功能详解
- [INFER_OUTPUT_STRUCTURE.md](./INFER_OUTPUT_STRUCTURE.md) - 输出结构说明
- [INFER_ADVANCED_FEATURES.md](./INFER_ADVANCED_FEATURES.md) - Attention 高级功能

## 版本历史

- **2025-01-08**: 初始实现
  - 添加 `--save-gt` 参数
  - 双模型模式自动包含 GT
  - 集成输出结构
  - 动态文件编号
