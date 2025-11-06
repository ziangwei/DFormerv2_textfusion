# Infer.py 使用指南

## 新功能概览

### 1. 精确图片选择

**问题：** 之前只能顺序取前N张，无法指定具体图片

**解决方案：** 新增3种选择模式

#### 模式A：指定索引 (`--image-indices`)
```bash
python -m utils.infer \
    --config xxx \
    --continue_fpath checkpoint.pth \
    --save-seg-overlay \
    --image-indices "0,5,10,42,100"
```
**适用场景：** 知道数据集中第几张图，想看特定case

#### 模式B：匹配文件名 (`--image-names`)
```bash
python -m utils.infer \
    --config xxx \
    --continue_fpath checkpoint.pth \
    --save-seg-overlay \
    --image-names "img_0001,img_0042,scene_kitchen"
```
**适用场景：** 知道文件名，想看特定场景

**匹配规则：** 子串匹配（不需要完整路径）
- `"img_0001"` 会匹配 `datasets/NYUDepthv2/rgb/img_0001.png`
- `"kitchen"` 会匹配所有包含"kitchen"的文件

#### 模式C：随机采样 (`--random-images`)
```bash
python -m utils.infer \
    --config xxx \
    --continue_fpath checkpoint.pth \
    --save-seg-overlay \
    --num-images 10 \
    --random-images
```
**适用场景：** 想看不同图片的整体效果，避免总是看前N张

### 2. 分割结果可视化 (`--save-seg-overlay`)

**问题：** 之前只有attention map，没有最核心的分割预测结果

**解决方案：** 自动保存3种可视化

**输出文件结构：**
```
results/
  segmentation/
    img_0001/
      ├── original.png      # 原图
      ├── pred_mask.png     # 纯分割结果（彩色mask）
      └── overlay.png       # 叠加图（mask + 原图）
    img_0042/
      ├── ...
```

**使用：**
```bash
python -m utils.infer \
    --save-seg-overlay \
    --attention-alpha 0.5  # 控制叠加透明度（0.0=全原图, 1.0=全mask）
```

### 3. 优先级规则

当多个参数同时存在时，按以下优先级：

1. `--image-indices` （最高优先级）
2. `--image-names`
3. `--random-images`
4. `--num-images`（顺序取前N张）

**示例：**
```bash
# 这会使用indices，忽略num-images
python -m utils.infer \
    --num-images 10 \
    --image-indices "5,10,15"  # 只处理这3张
```

## 常见使用场景

### 场景1：论文插图制作

**需求：** 选择效果最好的5张图片，保存高质量可视化

```bash
# 步骤1：先随机看20张，找到好的case
python -m utils.infer \
    --save-seg-overlay \
    --num-images 20 \
    --random-images

# 步骤2：记录好的图片索引（假设是 5,23,47,89,156）
# 步骤3：重新生成这些图片的高质量可视化
python -m utils.infer \
    --save-attention \
    --save-seg-overlay \
    --image-indices "5,23,47,89,156" \
    --filter-tokens "floor,wall,ceiling,window,bed" \
    --attention-alpha 0.6 \
    --attention-threshold 0.3 \
    --attention-smooth 1.0
```

### 场景2：错误分析

**需求：** 查看模型失败的特定case

```bash
# 已知某些图片分割失败（比如包含"bathroom"的场景）
python -m utils.infer \
    --save-seg-overlay \
    --image-names "bathroom"  # 匹配所有bathroom场景
```

### 场景3：模型对比

**需求：** 对比baseline和text fusion在相同图片上的效果

```bash
# 先确定测试图片（使用随机采样）
INDICES="5,10,15,20,25,30,35,40,45,50"

# Baseline
python -m utils.infer \
    --config baseline_config \
    --continue_fpath baseline.pth \
    --save-path results/baseline \
    --save-seg-overlay \
    --image-indices $INDICES

# Text Fusion
python -m utils.infer \
    --config textfusion_config \
    --continue_fpath textfusion.pth \
    --save-path results/textfusion \
    --save-seg-overlay \
    --image-indices $INDICES

# 现在可以对比同一张图的两个结果
```

### 场景4：快速验证

**需求：** 训练中途快速看效果，不需要attention细节

```bash
# 只保存分割结果，速度快
python -m utils.infer \
    --save-seg-overlay \
    --num-images 10  # 前10张
    # 注意：不加 --save-attention
```

## 参数完整列表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--num-images` | int | None | 处理多少张图片（None=全部） |
| `--image-indices` | str | None | 指定图片索引，逗号分隔（例如："0,5,10"） |
| `--image-names` | str | None | 匹配文件名，逗号分隔（例如："img_0001,scene_kitchen"） |
| `--random-images` | flag | False | 随机采样（需配合--num-images） |
| `--save-seg-overlay` | flag | False | 保存分割结果叠加图 |
| `--save-attention` | flag | False | 保存注意力图 |
| `--attention-alpha` | float | 0.5 | 叠加透明度（0.0~1.0） |
| `--filter-tokens` | str | None | 只可视化指定token（例如："floor,wall"） |

## 输出文件结构

```
results/
  ├── segmentation/              # 分割结果（--save-seg-overlay）
  │   ├── img_0001/
  │   │   ├── original.png       # 原图
  │   │   ├── pred_mask.png      # 彩色分割mask
  │   │   └── overlay.png        # 叠加图
  │   └── img_0002/
  │       └── ...
  │
  └── attention/                 # 注意力图（--save-attention）
      ├── enc_stage1_block0/
      │   ├── img_0001__label_floor_attn.png
      │   ├── img_0001__label_wall_attn.png
      │   └── ...
      └── dec_stage2/
          └── ...
```

## 注意事项

1. **内存管理：**
   - 指定图片时不会提前终止，会遍历整个dataset
   - 如果dataset很大但只可视化少量图片，建议使用`--image-indices`且索引靠前

2. **文件名匹配：**
   - 大小写不敏感
   - 支持部分匹配（子串）
   - 路径分隔符会被自动处理

3. **性能优化：**
   - 只需要分割结果时，不要加`--save-attention`（快很多）
   - attention可视化会消耗大量时间和磁盘空间

4. **分布式推理：**
   - 所有新功能都支持多GPU推理
   - 建议单GPU推理（`--gpus 1`）以确保图片索引对应正确
