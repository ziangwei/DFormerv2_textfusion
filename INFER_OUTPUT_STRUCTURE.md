# Infer.py 输出文件结构说明

## 核心设计理念

**每张图片 = 一个文件夹**

对于每张输入图片，所有相关的可视化结果都保存在同一个文件夹中，方便查看和对比。

---

## 文件组织结构

### 完整示例

假设处理3张图片，每张图有6个文本token：

```
output/
├── RGB_0/                          # 第一张图片
│   ├── 00_original.png             # 原始图片
│   ├── 01_segmentation.png         # 分割结果（彩色）
│   ├── 02_attn_wall.png            # Token 1: wall 的attention
│   ├── 03_attn_floor.png           # Token 2: floor 的attention
│   ├── 04_attn_ceiling.png         # Token 3: ceiling 的attention
│   ├── 05_attn_bed.png             # Token 4: bed 的attention
│   ├── 06_attn_chair.png           # Token 5: chair 的attention
│   └── 07_attn_table.png           # Token 6: table 的attention
│
├── RGB_1/                          # 第二张图片
│   ├── 00_original.png
│   ├── 01_segmentation.png
│   ├── 02_attn_wall.png
│   ├── 03_attn_floor.png
│   ├── 04_attn_window.png
│   ├── 05_attn_door.png
│   ├── 06_attn_cabinet.png
│   └── 07_attn_sofa.png
│
└── RGB_2/                          # 第三张图片
    ├── 00_original.png
    ├── 01_segmentation.png
    ├── 02_attn_floor.png
    ├── 03_attn_wall.png
    └── 04_attn_desk.png
```

### 文件命名规则

| 编号前缀 | 文件名格式 | 说明 |
|---------|-----------|------|
| `00_` | `00_original.png` | 原始RGB图片（反规范化后） |
| `01_` | `01_segmentation.png` | 最终分割结果（彩色调色板） |
| `02-99` | `XX_attn_<token_name>.png` | 各个token的attention可视化 |

**编号规则**：
- 固定顺序：00 原图 → 01 分割 → 02+ attention
- Attention maps 按 token 在列表中的顺序编号
- Token名称经过 slugify 处理（去除特殊字符，保留中英文数字）

---

## 每张图的输出内容

### 1. 原始图片 (00_original.png)

**内容**：反规范化后的原始RGB图像

**用途**：
- 对比原图和分割结果
- 检查attention是否聚焦在正确区域
- 论文插图的原始素材

**技术细节**：
```python
# 反规范化公式
rgb_denorm = (rgb_normalized * std + mean) * 255
```

---

### 2. 分割结果 (01_segmentation.png)

**内容**：模型最终预测的彩色分割图

**调色板**：
- NYUDepthv2/SUNRGBD: 使用 `utils/nyucmap.npy`
- 其他数据集: Cityscapes 风格调色板

**用途**：
- 直观查看分割效果
- 与Ground Truth对比
- 定性分析模型性能

**生成过程**：
```
模型输出 logits → argmax → 类别索引 → 调色板映射 → 彩色图
```

---

### 3. Token Attention Maps (02-XX_attn_*.png)

**内容**：每个文本token的注意力热力图叠加在原图上

**命名示例**：
- `02_attn_wall.png`
- `03_attn_floor.png`
- `04_attn_ceiling.png`

**可视化特性**：
- ✅ 热力图与原图混合（默认alpha=0.6）
- ✅ 可选竞争性归一化（突出token特定区域）
- ✅ 高斯平滑（减少噪声）
- ✅ Gamma校正（增强可见度）
- ✅ 阈值过滤（去除低响应）

**用途**：
- 理解模型如何关注不同语义区域
- 调试文本引导机制
- 论文可视化展示

---

## 使用示例

### 示例1: 基础可视化（单stage）

```bash
python utils/infer.py \
    --config local_configs/NYUDepthv2/DFormerv2_S.py \
    --gpus 1 \
    --continue_fpath checkpoints/best.pth \
    --save-attention \
    --save_path ./output \
    --num-images 10 \
    --vis-stage enc \
    --vis-stage-idx "2"
```

**输出**：
```
output/
├── RGB_0/
│   ├── 00_original.png
│   ├── 01_segmentation.png
│   ├── 02_attn_wall.png
│   ├── 03_attn_floor.png
│   └── ... (所有tokens)
├── RGB_1/
│   └── ...
...
```

---

### 示例2: 只保存特定tokens

```bash
python utils/infer.py \
    --config local_configs/NYUDepthv2/DFormerv2_S.py \
    --gpus 1 \
    --continue_fpath checkpoints/best.pth \
    --save-attention \
    --save_path ./output \
    --num-images 5 \
    --filter-tokens "wall,floor,ceiling" \
    --vis-stage enc \
    --vis-stage-idx "2"
```

**输出**（每张图只有3个attention maps）：
```
output/
├── RGB_0/
│   ├── 00_original.png
│   ├── 01_segmentation.png
│   ├── 02_attn_wall.png
│   ├── 03_attn_floor.png
│   └── 04_attn_ceiling.png
```

---

### 示例3: 多stage聚合

```bash
python utils/infer.py \
    --config local_configs/NYUDepthv2/DFormerv2_S.py \
    --gpus 1 \
    --continue_fpath checkpoints/best.pth \
    --save-attention \
    --save_path ./output \
    --num-images 5 \
    --vis-stage enc \
    --vis-stage-idx "all" \
    --vis-aggregate weighted
```

**输出**（包含聚合结果 + 各stage原始attention）：
```
output/
├── RGB_0/
│   ├── 00_original.png
│   ├── 01_segmentation.png
│   ├── 02_attn_wall.png      # 聚合后的 wall attention
│   ├── 03_attn_floor.png     # 聚合后的 floor attention
│   └── ... (其他tokens)
```

> **注意**：聚合模式下，保存的是多个stage融合后的attention，能更全面地反映模型的关注模式。

---

## 文件数量统计

### 计算公式

对于每张图片：
```
总文件数 = 2 (原图+分割) + N (tokens数量)
```

### 实际例子

| 场景 | Tokens数 | 总文件数/图 |
|------|---------|------------|
| NYU 全类别 | 40 | 42 |
| Per-image labels (avg) | 6 | 8 |
| Filtered tokens | 3 | 5 |

### 批量处理示例

处理100张图片，每张平均6个tokens：
```
总文件数 = 100 × (2 + 6) = 800 个文件
```

---

## 文件命名的 Slugify 处理

为了确保文件名在各操作系统上都能正常使用，token名称会经过处理：

### 处理规则

1. **保留**：中文、英文、数字、`._-`
2. **替换**：其他特殊字符 → `_`
3. **合并**：连续的 `_` → 单个 `_`
4. **截断**：最长120字符

### 示例

| 原始Token名称 | 处理后文件名 |
|--------------|-------------|
| `wall` | `02_attn_wall.png` |
| `floor mat` | `03_attn_floor_mat.png` |
| `night stand` | `04_attn_night_stand.png` |
| `墙壁` | `02_attn_墙壁.png` |
| `<pad>` | (跳过，不保存) |

---

## 优势分析

### ✅ 为什么这样设计？

#### 1. **一图一文件夹 = 直观清晰**

```
# 好：所有相关文件在一起
RGB_0/
├── 00_original.png
├── 01_segmentation.png
└── 02_attn_wall.png

# 差：文件分散在不同地方
predictions/RGB_0_pred.png
attention/wall/RGB_0.png
originals/RGB_0.png
```

#### 2. **编号前缀 = 自动排序**

文件浏览器会按文件名自动排序：
```
00_original.png        ← 第一个看到
01_segmentation.png    ← 第二个看到
02_attn_wall.png       ← 依次是各个attention
03_attn_floor.png
...
```

#### 3. **Token名称 = 一目了然**

不需要额外的JSON文件，从文件名就知道这是哪个token的attention。

#### 4. **易于批处理**

```bash
# 查看所有图片的分割结果
ls */01_segmentation.png

# 查看所有"wall"的attention
ls */02_attn_wall.png

# 统计每张图有多少tokens
for dir in RGB_*/; do
    echo "$dir: $(ls $dir/*_attn_*.png | wc -l) tokens"
done
```

---

## 与之前版本的对比

### 旧版本结构（已废弃）

```
output/
├── predictions/
│   ├── RGB_0_pred.png
│   └── RGB_1_pred.png
├── token_info/
│   ├── RGB_0_tokens.json
│   └── RGB_1_tokens.json
└── attention/
    ├── enc_stage2_block2/
    │   ├── RGB_0__imglabel_wall_attn.png
    │   └── RGB_0__imglabel_floor_attn.png
```

**问题**：
- ❌ 文件分散在3个不同目录
- ❌ 需要JSON文件才知道有哪些tokens
- ❌ 文件名冗长且包含重复信息
- ❌ 难以快速浏览单张图的所有结果

### 新版本结构（当前）

```
output/
├── RGB_0/
│   ├── 00_original.png
│   ├── 01_segmentation.png
│   ├── 02_attn_wall.png
│   └── 03_attn_floor.png
```

**优势**：
- ✅ 所有相关文件集中在一起
- ✅ 编号自动排序，查看顺序固定
- ✅ 文件名简洁明了
- ✅ 不需要额外的JSON文件

---

## 常见问题

### Q1: 如果只想要分割结果，不要attention？

**A**: 目前必须开启 `--save-attention` 才会保存。如果只想要分割结果，可以用 `--filter-tokens ""` 来跳过attention保存。

或者使用标准eval模式（不加 `--save-attention`），然后在 `val_mm.py` 中启用保存功能。

### Q2: 文件夹名称能自定义吗？

**A**: 文件夹名称由图片路径自动生成。例如：
- `datasets/NYUDepthv2/RGB/0.jpg` → `RGB_0`
- `datasets/test/image_100.png` → `test_image_100`

### Q3: 为什么有些图片的token数量不同？

**A**: 如果使用 `text_source=imglabels`，每张图的labels是不同的，所以token数量会变化。

示例：
- `RGB_0`: 6个tokens（wall, floor, ceiling, bed, chair, table）
- `RGB_1`: 4个tokens（wall, floor, window, door）

### Q4: 能只保存原图和分割结果吗？

**A**: 可以，使用 `--filter-tokens "none"` 或者修改代码跳过attention循环。

### Q5: 聚合模式下文件数量会变吗？

**A**: 不会。聚合模式保存的是融合后的attention，文件数量不变：
- 仍然是 `2 (原图+分割) + N (tokens)`
- 但attention质量更高（多stage融合）

---

## 性能考虑

### 磁盘空间

每个文件平均大小（480×640图像）：
- 原图：~100KB
- 分割：~50KB
- Attention：~80KB

**示例**：
- 100张图 × 8文件/图 × 80KB = ~64MB

### 处理速度

保存文件不会显著影响速度：
- 主要时间在模型forward
- 文件I/O相对较快

### 推荐配置

如果磁盘空间有限：
```bash
# 只保存关键tokens
--filter-tokens "wall,floor,ceiling"

# 减少图片数量
--num-images 50
```

---

## 总结

| 特性 | 说明 |
|------|------|
| **组织方式** | 一张图一个文件夹 |
| **文件数量** | 2 (原图+分割) + N (tokens) |
| **命名规则** | 编号前缀 + 描述性名称 |
| **自动排序** | ✅ 按编号顺序 |
| **易于浏览** | ✅ 所有相关文件在一起 |
| **易于批处理** | ✅ 支持shell glob模式 |

**核心优势**：简洁、直观、易用！
