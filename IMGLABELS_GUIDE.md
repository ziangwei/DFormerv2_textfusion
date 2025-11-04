# ImgLabels 模式使用指南

## 问题背景

当使用 `text_source="imglabels"` 模式时，每张图片应该有自己特定的标签列表（例如某张图只有 "floor", "wall", "bed" 这3个标签）。但如果 JSON 的 key 格式和代码中的图片路径不匹配，就会导致：

- ❌ 推理时无法找到图片对应的标签
- ❌ 回退到全局词汇表的前几个类别
- ❌ 注意力可视化显示错误的 token 名称
- ❌ IoU 性能下降

## 解决方案

我们提供了一套完整的诊断和修复工具。

---

## 🔍 步骤1：检查配置

首先运行配置检查工具：

```bash
python utils/check_attention_setup.py --config configs.sunrgbd.your_config
```

**输出示例**：
```
================================================================================
🔍 注意力可视化配置检查
================================================================================

1. Text Guidance: ✅ ENABLED

2. Text Source: imglabels
   ✅ Image labels file: datasets/sunrgbd/image_labels.json
      Total images in JSON: 5285
      Sample keys:
        'image/img_0001.jpg' -> 5 labels
        'image/img_0002.jpg' -> 8 labels
        'image/img_0003.jpg' -> 3 labels
```

**检查要点**：
- `enable_text_guidance` 必须是 `True`
- `image_labels_json_path` 必须存在且可读
- 注意 JSON 中 key 的格式（完整路径？basename？有无扩展名？）

---

## 🔬 步骤2：诊断 Key 匹配

运行诊断工具，检查 JSON keys 是否能匹配到实际的图片路径：

```bash
python utils/diagnose_imglabels.py \
    --image-labels-json datasets/sunrgbd/image_labels.json \
    --eval-source datasets/sunrgbd/test.txt \
    --rgb-root datasets/sunrgbd/SUNRGBD \
    --rgb-format .jpg \
    --num-samples 20
```

**输出示例（匹配成功）**：
```
================================================================================
🔍 ImgLabels Key Matching Diagnosis
================================================================================

1. Loaded image_labels.json: datasets/sunrgbd/image_labels.json
   Total entries: 5285
   Sample keys in JSON:
      'img_0001.jpg' -> [floor, wall, window, ... (5 total)]
      'img_0002.jpg' -> [floor, wall, bed, table, ... (8 total)]

2. Loaded eval source: datasets/sunrgbd/test.txt
   Total items: 5050

3. Testing key matching for first 20 images:
================================================================================

[1] item_name: 0001
    rgb_path: datasets/sunrgbd/SUNRGBD/img_0001.jpg
    ✅ MATCHED with key: 'img_0001.jpg'
    Labels: [floor, wall, window, bed, nightstand]

[2] item_name: 0002
    rgb_path: datasets/sunrgbd/SUNRGBD/img_0002.jpg
    ✅ MATCHED with key: 'img_0002.jpg'
    Labels: [floor, wall, bed, table, chair, lamp, picture, curtain]

...

================================================================================
📊 Summary:
================================================================================
Matched:   20/20 (100.0%)
Unmatched: 0/20 (0.0%)

✅ All samples matched successfully!
   Your imglabels configuration should work correctly.
```

**输出示例（匹配失败）**：
```
[1] item_name: 0001
    rgb_path: datasets/sunrgbd/SUNRGBD/img_0001.jpg
    ❌ NOT MATCHED
    Tried keys:
      - 'datasets/sunrgbd/SUNRGBD/img_0001.jpg'
      - 'img_0001.jpg'
      - 'img_0001'
      - '0001'

================================================================================
📊 Summary:
================================================================================
Matched:   0/20 (0.0%)
Unmatched: 20/20 (100.0%)

💡 Recommendations:
1. Check if JSON keys match your file structure:
   - JSON keys look like: image/img_0001.jpg
   - Image paths look like: datasets/sunrgbd/SUNRGBD/img_0001.jpg

2. Common issues:
   a) JSON keys use full path, but code expects basename
   b) JSON keys have 'image/' prefix not in actual paths
   c) File extensions mismatch (.jpg vs .png)

3. Suggested fix: Normalize JSON keys
```

---

## 🔧 步骤3：修复 Key 不匹配

如果诊断工具显示 **Unmatched > 0**，说明 key 格式不对，需要规范化 JSON：

```bash
python utils/normalize_imglabels_keys.py \
    --input datasets/sunrgbd/image_labels.json \
    --output datasets/sunrgbd/image_labels.normalized.json
```

**这个工具会**：
1. 读取原始 JSON
2. 为每个条目生成多个 key 变体：
   - 完整路径：`image/img_0001.jpg`
   - Basename：`img_0001.jpg`
   - 无扩展名：`img_0001`
   - 各种扩展名：`img_0001.png`, `img_0001.JPG` 等

3. 输出规范化的 JSON，确保能匹配各种路径格式

**输出示例**：
```
📖 Loading: datasets/sunrgbd/image_labels.json
   Original entries: 5285
   Normalized entries: 5285 (from 5285 original)
   Total key variants: 26425
   Average variants per entry: 5.0

📋 Sample key mappings:

   Original key: 'image/img_0001.jpg'
   Labels: [floor, wall, window, bed, nightstand]
   Generated variants:
      - 'image/img_0001.jpg'
      - 'img_0001.jpg'
      - 'img_0001'
      - 'img_0001.png'
      - 'img_0001.JPG'

💾 Saving to: datasets/sunrgbd/image_labels.normalized.json
✅ Done!

💡 Next steps:
   1. Update your config to use the normalized JSON:
      C.image_labels_json_path = 'datasets/sunrgbd/image_labels.normalized.json'

   2. Or replace the original file:
      mv datasets/sunrgbd/image_labels.normalized.json datasets/sunrgbd/image_labels.json
```

**更新配置**：

方法1：修改配置文件
```python
# configs/sunrgbd/your_config.py
C.image_labels_json_path = "datasets/sunrgbd/image_labels.normalized.json"
```

方法2：推理时覆盖
```bash
python utils/infer.py --config configs.sunrgbd.your_config \
    --image-labels-json-path datasets/sunrgbd/image_labels.normalized.json \
    ...
```

---

## 🎯 步骤4：验证修复

再次运行诊断工具，确认 100% 匹配：

```bash
python utils/diagnose_imglabels.py \
    --image-labels-json datasets/sunrgbd/image_labels.normalized.json \
    --eval-source datasets/sunrgbd/test.txt \
    --rgb-root datasets/sunrgbd/SUNRGBD \
    --rgb-format .jpg \
    --num-samples 50
```

应该看到：
```
Matched:   50/50 (100.0%)
Unmatched: 0/50 (0.0%)
✅ All samples matched successfully!
```

---

## 📊 步骤5：推理和可视化

现在可以正常推理了，每张图片会使用自己的标签：

### 标准推理（不可视化）
```bash
python utils/infer.py \
    --config configs.sunrgbd.your_config \
    --continue_fpath checkpoints/your_model.pth \
    --save_path ./eval_output
```

### 可视化所有 tokens（能量排序）
```bash
python utils/infer.py \
    --config configs.sunrgbd.your_config \
    --continue_fpath checkpoints/your_model.pth \
    --save-attention \
    --save_path ./vis_all_tokens \
    --vis-stage enc \
    --vis-stage-idx 0 \
    --num-images 20
```

### 只可视化特定 tokens
```bash
# 假设某张图有标签：[floor, wall, bed, window, nightstand]
# 只可视化 floor 和 wall
python utils/infer.py \
    --config configs.sunrgbd.your_config \
    --continue_fpath checkpoints/your_model.pth \
    --save-attention \
    --save_path ./vis_floor_wall \
    --vis-stage enc \
    --vis-stage-idx 0 \
    --num-images 20 \
    --filter-tokens 'floor,wall'
```

**输出结构**：
```
vis_floor_wall/
└── attention/
    └── enc_stage0_block0/
        ├── img_0001__class_floor_heatmap.png
        ├── img_0001__class_floor_overlay.png
        ├── img_0001__class_wall_heatmap.png
        ├── img_0001__class_wall_overlay.png
        ├── img_0002__class_floor_heatmap.png
        ...
```

---

## 🐛 调试技巧

### 1. 查看详细日志

代码会自动记录前5次和每100次的匹配失败：

```
[WARNING] [ImgLabels] No match for image (miss #1)
  rgb_path: datasets/sunrgbd/SUNRGBD/img_0001.jpg
  item_name: 0001
  Tried keys: ['datasets/sunrgbd/SUNRGBD/img_0001.jpg', 'img_0001.jpg', 'img_0001', '0001']
  Available keys sample: ['image/img_0001.jpg', 'image/img_0002.jpg', 'image/img_0003.jpg']
```

**对比 "Tried keys" 和 "Available keys"**，看出格式差异：
- Tried: `img_0001.jpg`
- Available: `image/img_0001.jpg` ← 多了 `image/` 前缀！

### 2. 手动检查 JSON

```bash
# 查看 JSON 的前几个 keys
python -c "import json; f=open('datasets/sunrgbd/image_labels.json'); d=json.load(f); print(list(d.keys())[:10])"
```

### 3. 检查数据集文件结构

```bash
# 查看评估列表的格式
head -n 10 datasets/sunrgbd/test.txt

# 查看实际图片路径
ls datasets/sunrgbd/SUNRGBD/*.jpg | head -n 10
```

---

## ✅ 常见问题

### Q1: 诊断工具显示 100% 匹配，但推理时还是显示词汇表？

**可能原因**：
- 推理时的配置和诊断时不一致
- `text_source` 设置错误（应该是 "imglabels"）
- `enable_text_guidance` 未开启

**解决方案**：
```bash
# 确保推理时强制使用正确的配置
python utils/infer.py \
    --config configs.sunrgbd.your_config \
    --text-source imglabels \
    --image-labels-json-path datasets/sunrgbd/image_labels.normalized.json \
    --save-attention \
    ...
```

### Q2: 为什么有些图片的 token 数量不一样？

这是正常的！`imglabels` 模式的特点就是**每张图有自己的标签数量**：
- 图1：`[floor, wall, window]` → 3 tokens
- 图2：`[floor, wall, bed, table, chair]` → 5 tokens
- 图3：`[floor, ceiling]` → 2 tokens

代码会自动 pad 到统一长度（`_imglabel_tokens`），多余的会填充零向量。

### Q3: IoU 还是很低怎么办？

如果 key 匹配正确但 IoU 还是低，检查：

1. **模型权重**：是否加载了正确的 checkpoint？
   ```bash
   --continue_fpath /path/to/correct/checkpoint.pth
   ```

2. **训练时的配置**：推理时的配置要和训练时一致
   - `text_source`
   - `text_encoder`
   - `text_feature_dim`
   - `max_image_labels`

3. **SAM 配置**：encoder/decoder stages 是否正确？
   ```bash
   --sam-enc-stages 0,2 --sam-dec-stages 1,3
   ```

4. **Superpower 模式**：训练时开了要推理时也开
   ```bash
   --superpower
   ```

---

## 📁 文件清单

修复后新增的工具：

| 文件 | 用途 |
|------|------|
| `utils/diagnose_imglabels.py` | 诊断 JSON key 匹配问题 |
| `utils/normalize_imglabels_keys.py` | 规范化 JSON，生成多个 key 变体 |
| `utils/check_attention_setup.py` | 检查整体配置（已更新） |
| `IMGLABELS_GUIDE.md` | 本指南 |

---

## 🚀 快速开始（完整流程）

```bash
# 1. 检查配置
python utils/check_attention_setup.py --config configs.sunrgbd.your_config

# 2. 诊断 key 匹配
python utils/diagnose_imglabels.py \
    --image-labels-json datasets/sunrgbd/image_labels.json \
    --eval-source datasets/sunrgbd/test.txt \
    --rgb-root datasets/sunrgbd/SUNRGBD \
    --rgb-format .jpg

# 3. 如果不匹配，规范化 JSON
python utils/normalize_imglabels_keys.py \
    --input datasets/sunrgbd/image_labels.json \
    --output datasets/sunrgbd/image_labels.normalized.json

# 4. 再次验证
python utils/diagnose_imglabels.py \
    --image-labels-json datasets/sunrgbd/image_labels.normalized.json \
    --eval-source datasets/sunrgbd/test.txt \
    --rgb-root datasets/sunrgbd/SUNRGBD \
    --rgb-format .jpg

# 5. 推理 + 可视化特定 tokens
python utils/infer.py \
    --config configs.sunrgbd.your_config \
    --continue_fpath checkpoints/your_model.pth \
    --image-labels-json-path datasets/sunrgbd/image_labels.normalized.json \
    --save-attention \
    --save_path ./vis_output \
    --filter-tokens 'floor,wall,bed' \
    --num-images 20
```

---

## 💡 最后提示

- **训练和推理配置要一致**：特别是 `text_source`, `text_encoder`, `image_labels_json_path`
- **先诊断再推理**：确保 100% 匹配后再运行完整推理
- **使用 --filter-tokens**：大幅减少输出文件数量，专注于关心的类别
- **检查日志**：推理时留意 `[ImgLabels] No match` 警告

如果还有问题，欢迎提供：
1. 诊断工具的完整输出
2. JSON 的示例 keys（前3个）
3. test.txt 的示例行（前3行）
4. 实际图片路径的示例

祝调试顺利！🎉
