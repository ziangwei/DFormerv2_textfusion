# 双模型模式故障排除指南 (Troubleshooting Dual-Model Mode)

## 常见问题：模型架构不兼容

### 问题描述

当使用 `--dual-model` 时，可能会遇到以下错误：

```
RuntimeError: Error(s) in loading state_dict for DFormer:
Missing key(s) in state_dict: ...
Unexpected key(s) in state_dict: ...
```

或者：

```
AttributeError: ...
KeyError: ...
```

### 问题原因

这是因为**模型1**（文本引导）和**模型2**（纯视觉）的训练方式不同，导致：

1. **文本侧结构不同**：
   - 模型1训练时：`enable_text_guidance=True`，有完整的文本编码器和 SAM 模块
   - 模型2训练时：`enable_text_guidance=False`，没有文本相关模块

2. **Checkpoint 权重不匹配**：
   - 模型1的 checkpoint 包含文本侧的权重
   - 模型2的 checkpoint 可能没有文本侧权重，或者有但结构不同

3. **Config 冲突**：
   - 即使设置 `config.enable_text_guidance=False`，如果 config 中还有其他文本相关配置（如 `text_encoder`、`text_feature_dim` 等），模型创建时可能还是会初始化文本模块

## 解决方案

### 方案1：使用自动清理（推荐，默认行为）

**无需额外操作**，程序会自动：

1. 清理 config 中的所有文本相关配置
2. 使用 `strict=False` 加载权重（忽略不匹配的部分）
3. 显示详细的加载日志

**示例日志**：

```
====================================================================================================
STEP 2/2: Running Model 2 (Visual-Only + Prediction Saving)
====================================================================================================
Clearing Model 1 from GPU memory...
✓ Model 1 cleared

Reconfiguring shared config for visual-only mode...
  Removing all text-related configurations...
✓ config.enable_text_guidance = False
✓ All text-related configs cleared

Creating Model 2 architecture...
✓ Model 2 architecture created successfully

Loading Model 2 weights (strict=False, ignoring mismatched keys)...
  Missing keys (not loaded from checkpoint): 12
    - sam_encoder.text_proj.weight
    - sam_encoder.text_proj.bias
    - sam_decoder.cross_attn.q_proj.weight
    ...
  Unexpected keys (in checkpoint but not in model): 0
  ✓ Weights loaded with partial matching (this is expected for different architectures)
```

**这是正常的**！Missing keys 说明模型2的架构中有些层（文本相关）没有从 checkpoint 中加载权重，但这些层在纯视觉模式下不会被使用，所以不影响推理。

### 方案2：使用独立的 Config 文件（高级）

如果自动清理还是不够（比如你的两个模型架构差异很大），可以为模型2指定独立的配置文件：

```bash
python utils/infer.py \
    --config local_configs/NYUDepthv2/DFormerv2_S_TextGuided.py \
    --continue_fpath checkpoints/text_guided.pth \
    --dual-model \
    --model2-path checkpoints/visual_only.pth \
    --model2-config local_configs/NYUDepthv2/DFormerv2_S_VisualOnly.py \
    --num-images 10 --random-select
```

**创建 `DFormerv2_S_VisualOnly.py`**（示例）：

```python
# 从原始配置复制
from .DFormerv2_S import C

# 确保文本引导完全关闭
C.enable_text_guidance = False

# 移除所有文本相关配置
C.text_source = None
C.text_encoder = None
C.text_encoder_name = None
C.label_txt_path = None
C.image_labels_json_path = None
C.caption_json_path = None
C.text_template_set = None
```

### 方案3：检查权重加载日志

运行后查看日志中的 `Missing keys` 和 `Unexpected keys`：

**正常情况**（可以忽略）：
```
Missing keys (not loaded from checkpoint): 12
  - sam_encoder.text_proj.weight      # 文本投影层
  - sam_decoder.cross_attn.weight     # 交叉注意力层
  ...
```

这些都是文本相关的层，在纯视觉模式下不会被使用。

**异常情况**（需要处理）：
```
Missing keys (not loaded from checkpoint): 156
  - encoder.layer1.conv1.weight       # 主干网络的权重缺失！
  - decoder.head.weight               # 解码器头部缺失！
  ...
```

如果缺失的是主干网络或解码器的权重，说明 checkpoint 和模型架构严重不匹配，需要检查：
- Checkpoint 文件是否正确
- 是否使用了正确的 config

## 常见错误及解决

### 错误1：RuntimeError: size mismatch

```
RuntimeError: Error(s) in loading state_dict for DFormer:
size mismatch for decoder.head.weight: copying a param with shape torch.Size([40, 512])
from checkpoint, the shape in current model is torch.Size([13, 512]).
```

**原因**：模型1和模型2使用了不同的类别数（40 vs 13）。

**解决**：使用 `--model2-config` 指定正确的配置，或者修改 config 中的 `num_classes`。

### 错误2：AttributeError: 'Config' object has no attribute 'xxx'

```
AttributeError: 'Config' object has no attribute 'text_encoder'
```

**原因**：模型代码中访问了不存在的配置项。

**解决**：
1. 检查模型代码，确保在访问 `config.text_encoder` 前先检查是否存在
2. 或者在 config 中添加默认值：`C.text_encoder = None`

### 错误3：KeyError during forward pass

```
KeyError: 'text_features'
```

**原因**：dataloader 返回的数据中没有 `text_features`，但模型前向传播时尝试访问。

**解决**：
1. 确保 `config.enable_text_guidance=False` 生效
2. 检查模型前向传播代码，确保在 `enable_text_guidance=False` 时不访问文本特征

## 最佳实践

### 训练时

1. **明确设置 `enable_text_guidance`**：
   ```python
   # 文本引导模型
   C.enable_text_guidance = True

   # 纯视觉模型
   C.enable_text_guidance = False
   ```

2. **使用统一的主干配置**：
   - 即使是纯视觉模型，也使用相同的 backbone、decoder 等配置
   - 只在文本侧有差异

3. **保存完整的 config**：
   ```python
   torch.save({
       'model': model.state_dict(),
       'config': config,  # 保存配置，方便推理时加载
   }, 'checkpoint.pth')
   ```

### 推理时

1. **优先使用自动清理**：
   - 大多数情况下，默认的自动清理就足够了
   - 查看日志中的 `Missing keys`，确认都是文本相关的层

2. **必要时使用独立 config**：
   - 如果架构差异很大（比如不同的 backbone），使用 `--model2-config`

3. **检查日志**：
   - 确保 `Missing keys` 都是预期的（文本相关）
   - 确保没有主干网络或解码器的权重缺失

## 参数说明

### --model2-config

为模型2指定独立的配置文件。

**使用场景**：
- 两个模型架构差异很大
- 自动清理后仍然报错
- 想要完全隔离两个模型的配置

**示例**：
```bash
--model2-config local_configs/NYUDepthv2/DFormerv2_S_VisualOnly.py
```

**不使用时**：
- 默认使用相同的 config
- 自动清理所有文本相关配置
- 适用于大多数情况

## 调试技巧

### 1. 查看 checkpoint 内容

```python
import torch

# 加载 checkpoint
ckpt = torch.load('checkpoints/model.pth', map_location='cpu')

# 查看所有 key
print("Keys in checkpoint:")
for key in ckpt['model'].keys():
    print(f"  {key}: {ckpt['model'][key].shape}")

# 查看是否有文本相关的权重
text_keys = [k for k in ckpt['model'].keys() if 'text' in k or 'sam' in k]
print(f"\nText-related keys: {len(text_keys)}")
for key in text_keys:
    print(f"  {key}")
```

### 2. 对比两个 checkpoint

```python
ckpt1 = torch.load('checkpoints/model1.pth', map_location='cpu')
ckpt2 = torch.load('checkpoints/model2.pth', map_location='cpu')

keys1 = set(ckpt1['model'].keys())
keys2 = set(ckpt2['model'].keys())

print(f"Only in model1: {keys1 - keys2}")
print(f"Only in model2: {keys2 - keys1}")
print(f"Common keys: {keys1 & keys2}")
```

### 3. 测试模型创建

```python
from importlib import import_module
from model.semseg import segmodel
import torch.nn as nn

# 加载 config
config = getattr(import_module('local_configs.NYUDepthv2.DFormerv2_S'), "C")

# 清理文本配置
config.enable_text_guidance = False
config.text_source = None
config.text_encoder = None
# ... 其他清理

# 尝试创建模型
try:
    model = segmodel(cfg=config, norm_layer=nn.BatchNorm2d)
    print("✓ Model created successfully")

    # 查看模型结构
    for name, module in model.named_modules():
        if 'text' in name.lower() or 'sam' in name.lower():
            print(f"  Found text-related module: {name}")
except Exception as e:
    print(f"✗ Failed to create model: {e}")
```

## FAQ

### Q1: Missing keys 是否会影响推理结果？

**A:** 取决于缺失的是哪些 keys：
- **文本相关的层**（如 `sam_encoder.text_proj`）：不影响，这些层在纯视觉模式下不会被使用
- **主干网络或解码器**：严重影响，需要修复

### Q2: Unexpected keys 是否需要处理？

**A:** 一般不需要。Unexpected keys 说明 checkpoint 中有些权重在当前模型中不存在，使用 `strict=False` 后会自动忽略。

### Q3: 为什么不直接重新训练一个 baseline？

**A:**
- 重新训练需要很长时间
- 使用现有的 checkpoint，配合自动清理和 `strict=False`，可以直接使用
- 即使架构有小差异，只要主干网络和解码器一致，推理结果就是可信的

### Q4: 如何确认模型2的推理结果是正确的？

**A:**
1. 检查 mIoU 等指标是否合理（应该接近训练时的结果）
2. 可视化几张预测结果，看是否正常
3. 对比 GT 和预测，确认没有明显的异常

### Q5: 能否使用完全不同架构的两个模型？

**A:**
- 可以，但需要使用 `--model2-config` 指定独立的配置
- 两个模型必须使用相同的 dataloader（相同的数据预处理）
- 输出类别数必须相同（使用相同的 palette）

## 技术细节

### 权重加载的 strict 参数

```python
# strict=True (默认)
# - 所有 checkpoint 中的 key 必须在模型中存在
# - 所有模型中的 key 必须在 checkpoint 中存在
# - 任何不匹配都会报错

# strict=False
# - 忽略 checkpoint 中多余的 key (unexpected keys)
# - 允许模型中有些 key 不在 checkpoint 中 (missing keys)
# - 只加载匹配的部分，其他的保持初始化状态

model.load_state_dict(checkpoint, strict=False)
```

### Config 清理的实现

```python
# 完全禁用文本引导
config.enable_text_guidance = False

# 清理所有文本相关配置
text_related_attrs = [
    'text_source', 'text_encoder', 'text_encoder_name',
    'text_feature_dim', 'label_txt_path', 'image_labels_json_path',
    'caption_json_path', 'text_template_set'
]

for attr in text_related_attrs:
    if hasattr(config, attr):
        setattr(config, attr, None)
```

## 相关文档

- [INFER_DUAL_MODEL.md](./INFER_DUAL_MODEL.md) - 双模型对比功能
- [INFER_GT_COMPARISON.md](./INFER_GT_COMPARISON.md) - GT 对比功能

## 总结

**不需要重新训练 baseline**！使用以下策略可以直接使用现有的 checkpoint：

1. ✅ 使用自动清理（默认）
2. ✅ 检查日志中的 Missing/Unexpected keys
3. ✅ 必要时使用 `--model2-config`
4. ✅ 验证推理结果（mIoU、可视化）

只要主干网络和解码器的权重正确加载，文本侧的差异不会影响纯视觉模式的推理结果。
