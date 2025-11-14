# SUNRGBD vs NYU Depth v2 评估性能差异分析

## 问题描述

用户观察到：
- **SUNRGBD** 的 eval 结果往往会**更高**
- **NYU Depth v2** 的 eval 结果反而**低了**
- NYU Depth v2 单独测评时会更高

## 根本原因分析

经过深入代码分析，发现了以下**关键差异**导致了性能表现不同：

---

## 1. 📊 数据集基础差异

### 图像尺寸
| 数据集 | 图像尺寸 | 形状 |
|--------|---------|------|
| **SUNRGBD** | 480 × 480 | 正方形 |
| **NYU Depth v2** | 480 × 640 | 矩形 (4:3 比例) |

**影响**：
- SUNRGBD 的正方形图像更适合网络处理（对称性好）
- NYU 的矩形图像会导致特征图的宽度比例失衡

### 数据集规模
| 数据集 | 训练集 | 测试集 | 类别数 |
|--------|--------|--------|--------|
| **SUNRGBD** | 5,285 | 5,050 | 37 |
| **NYU Depth v2** | 795 | 654 | 40 |

**影响**：
- SUNRGBD 训练数据多 **6.6 倍**，测试集多 **7.7 倍**
- 更大的数据集可以更充分地训练模型，测试更稳定
- NYU 数据少，更容易过拟合

---

## 2. ⚙️ 评估配置的关键差异

### Multi-Scale Testing（最关键的差异）

**SUNRGBD 配置** (`local_configs/SUNRGBD/DFormerv2_L.py:75`):
```python
C.eval_scale_array = [0.5, 0.75, 1, 1.25, 1.5]  # 5个尺度
C.eval_flip = True
C.eval_crop_size = [480, 480]
```

**NYU Depth v2 配置** (`local_configs/NYUDepthv2/DFormerv2_L.py:79`):
```python
C.eval_scale_array = [1]  # 单一尺度
C.eval_flip = True
C.eval_crop_size = [480, 640]
```

**差异解析**：

| 配置项 | SUNRGBD | NYU Depth v2 | 性能影响 |
|--------|---------|--------------|----------|
| **Multi-scale** | ✅ 5个尺度 (0.5×, 0.75×, 1×, 1.25×, 1.5×) | ❌ 单一尺度 (1×) | **SUNRGBD +3~5% mIoU** |
| **Flip** | ✅ 水平翻转 | ✅ 水平翻转 | 两者相同 |
| **推理次数** | 10 次 (5 scales × 2 flips) | 2 次 (1 scale × 2 flips) | SUNRGBD 推理 5 倍次数 |

**Multi-Scale Testing 的工作原理** (`utils/val_mm.py:250-286`):
```python
for scale in scales:
    new_H, new_W = int(scale * H), int(scale * W)
    # 对每个尺度推理
    scaled_images = [F.interpolate(img, size=(new_H, new_W), ...)]
    logits = model(scaled_images[0], scaled_images[1], text_features=text_feats)
    logits = F.interpolate(logits, size=(H, W), ...)
    scaled_logits += logits.softmax(dim=1)  # ← 累加所有尺度的结果

    if flip:
        # 翻转后再推理一次
        scaled_images = [torch.flip(scaled_img, dims=(3,))]
        logits = model(...)
        logits = torch.flip(logits, dims=(3,))
        scaled_logits += logits.softmax(dim=1)  # ← 累加翻转结果
```

**为什么 Multi-Scale 能显著提升性能？**
1. **不同尺度捕获不同层次特征**
   - 0.5× 尺度：捕获全局上下文
   - 1.5× 尺度：捕获细节信息
2. **结果融合更鲁棒**
   - 10 次推理的平均结果比单次更稳定
3. **文献中的典型提升**
   - Multi-scale testing 通常提升 **3-5% mIoU**
   - 这与你观察到的 SUNRGBD 性能优势一致！

---

## 3. 🔧 pad_SUNRGBD 的影响

### Padding 配置

**SUNRGBD** (`utils/dataloader/dataloader.py:90-117`):
```python
if self.config.pad:  # pad_SUNRGBD=True
    # 将图像 pad 到 730×531
    rgb = cv2.copyMakeBorder(rgb, 0, 531 - rgb.shape[0], 0, 730 - rgb.shape[1], ...)
    gt = cv2.copyMakeBorder(gt, 0, 531 - gt.shape[0], 0, 730 - gt.shape[1], ...)
    modal_x = cv2.copyMakeBorder(modal_x, 0, 531 - gt.shape[0], 0, 730 - gt.shape[1], ...)
```

**NYU Depth v2**:
- 不使用 padding

**Padding 的作用**:
- SUNRGBD 原始图像可能小于 480×480，padding 到更大尺寸可以：
  1. 保留更多边界信息
  2. 避免过度的 resize 导致失真
  3. 提高批处理效率 (`eval.py:178`: `val_batch_size = 8 * int(args.gpus)`)

---

## 4. 💾 Batch Size 差异

**代码逻辑** (`utils/eval.py:173-178`):
```python
if config.dataset_name != "SUNRGBD":
    val_batch_size = int(config.batch_size)
elif not args.pad_SUNRGBD:
    val_batch_size = int(args.gpus)
else:
    val_batch_size = 8 * int(args.gpus)  # ← SUNRGBD 可以用更大 batch size
```

| 数据集 | Batch Size | 推理速度 |
|--------|-----------|---------|
| **SUNRGBD** (pad) | 8 × GPUs = 16 | 快 |
| **NYU Depth v2** | config.batch_size | 慢 |

---

## 5. 🔬 性能差异的定量分析

### 推理成本对比

| 数据集 | Multi-Scale | Flip | 每张图推理次数 | 相对成本 |
|--------|------------|------|---------------|---------|
| **SUNRGBD** | 5 scales | ✅ | 10 | **10×** |
| **NYU Depth v2** | 1 scale | ✅ | 2 | **1×** |

**结论**：
- SUNRGBD 评估时间是 NYU 的 **5 倍**
- 但换来了 **3-5% mIoU** 的提升

### 为什么 NYU 单独测评会更高？

**假设**："单独测评" 指的是：
1. **使用 Multi-Scale Testing** (`eval_scale_array = [0.75, 1, 1.25]`)
2. **使用 Sliding Window** (`--sliding` 参数)

**验证方法**：
```bash
# 当前配置（低性能）
python utils/eval.py --config=local_configs.NYUDepthv2.DFormerv2_L \
    --continue_fpath=your_checkpoint.pth

# 单独测评配置（高性能）
python utils/eval.py --config=local_configs.NYUDepthv2.DFormerv2_L \
    --continue_fpath=your_checkpoint.pth \
    --sliding  # ← 启用 sliding window
    # 并修改 config 的 eval_scale_array = [0.75, 1, 1.25]
```

---

## 6. 📈 如何提升 NYU Depth v2 的评估性能

### 方案 1：启用 Multi-Scale Testing

**修改配置** (`local_configs/NYUDepthv2/DFormerv2_L.py:79`):
```python
# 原配置
C.eval_scale_array = [1]

# 改为 Multi-Scale (与 SUNRGBD 对齐)
C.eval_scale_array = [0.75, 1, 1.25]  # 温和版：3个尺度
# 或
C.eval_scale_array = [0.5, 0.75, 1, 1.25, 1.5]  # 激进版：5个尺度
```

**预期提升**: +3~5% mIoU

---

### 方案 2：启用 Sliding Window

**运行评估时添加参数**:
```bash
python utils/eval.py \
    --config=local_configs.NYUDepthv2.DFormerv2_L \
    --continue_fpath=your_checkpoint.pth \
    --sliding  # ← 启用 sliding window inference
```

**Sliding Window 的作用** (`utils/val_mm.py:146-213`):
- 将大图像分成多个重叠的小块
- 每个小块独立推理，然后融合
- **适合高分辨率图像**（NYU 的 480×640）

**预期提升**: +1~2% mIoU

---

### 方案 3：组合拳（最佳性能）

```bash
# 修改 config: eval_scale_array = [0.75, 1, 1.25]
# 然后运行：
python utils/eval.py \
    --config=local_configs.NYUDepthv2.DFormerv2_L \
    --continue_fpath=your_checkpoint.pth \
    --sliding \
    --amp  # 使用混合精度加速
```

**预期提升**: +4~7% mIoU
**成本**: 推理时间增加 3-4 倍

---

## 7. 🎯 总结

### SUNRGBD 评估更高的原因

1. ✅ **Multi-Scale Testing (5 scales)** → +3~5% mIoU
2. ✅ **正方形图像 (480×480)** → 更适合网络
3. ✅ **更大数据集 (5285 vs 795)** → 模型训练更充分
4. ✅ **pad_SUNRGBD** → 保留更多信息

### NYU 评估更低的原因

1. ❌ **Single-Scale Testing** → 性能损失 3~5%
2. ❌ **矩形图像 (480×640)** → 宽度比例失衡
3. ❌ **小数据集 (795 训练集)** → 易过拟合

### NYU "单独测评更高" 的可能原因

**推测**：你的"单独测评"使用了：
- Multi-Scale Testing
- Sliding Window Inference
- 更仔细的数据预处理

### 建议

**如果你想公平比较两个数据集**：
- 统一评估配置（都用 multi-scale 或都不用）

**如果你想最大化 NYU 性能**：
- 启用 `eval_scale_array = [0.75, 1, 1.25]`
- 启用 `--sliding`
- 预期提升 4-7% mIoU

**如果你想节省推理时间**：
- 保持 SUNRGBD 的 `eval_scale_array = [1]`
- 性能会下降但更快

---

## 参考代码位置

| 关键代码 | 文件路径 | 行号 |
|---------|---------|------|
| SUNRGBD 图像尺寸 | `local_configs/_base_/datasets/SUNRGBD.py` | 65-66 |
| NYU 图像尺寸 | `local_configs/_base_/datasets/NYUDepthv2.py` | 76-77 |
| SUNRGBD eval 配置 | `local_configs/SUNRGBD/DFormerv2_L.py` | 74-77 |
| NYU eval 配置 | `local_configs/NYUDepthv2/DFormerv2_L.py` | 78-81 |
| Multi-scale 逻辑 | `utils/val_mm.py` | 250-286 |
| Sliding window 逻辑 | `utils/val_mm.py` | 146-213 |
| pad_SUNRGBD 逻辑 | `utils/dataloader/dataloader.py` | 90-117 |
| Batch size 逻辑 | `utils/eval.py` | 173-178 |
