# mIoU不增长问题诊断指南

## 问题现象
训练SUNRGBD数据集时，mIoU一直输出同一个值，完全不增长。

## 可能原因分析

### 1. 模型预测collapse（最常见）
**症状**：模型始终预测同一个类别
**原因**：
- 学习率过小/过大
- 类别不平衡严重
- 初始化问题
- Loss计算错误

### 2. 混淆矩阵计算错误
**症状**：mIoU计算逻辑有bug
**原因**：
- `num_classes` 设置错误
- `ignore_label` 处理不当
- 混淆矩阵索引越界

### 3. 梯度问题
**症状**：Loss不下降或Loss=NaN
**原因**：
- 梯度消失/爆炸
- 学习率设置不当
- 数值溢出

### 4. 数据问题
**症状**：标签加载错误
**原因**：
- 标签值超出范围
- 标签全是ignore_label
- 数据增强导致标签错位

---

## 诊断步骤

### 步骤1: 检查模型预测多样性（最关键）

在 `utils/val_mm.py` 第67行附近添加：

```python
# 原代码
preds = model(images[0], images[1], text_features=text_feats).softmax(dim=1)

# === 添加诊断代码 ===
pred_classes = preds.argmax(dim=1)  # [B, H, W]
unique_preds = torch.unique(pred_classes)

print(f"\n[🔍 诊断] Batch {idx}/{len(dataloader)}:")
print(f"  预测的唯一类别数: {len(unique_preds)}/{n_classes}")
print(f"  预测的类别ID: {sorted(unique_preds.tolist())}")

# 检查预测分布
pred_flat = pred_classes.flatten()
for cls in unique_preds[:10]:
    count = (pred_flat == cls).sum().item()
    ratio = count / pred_flat.numel() * 100
    print(f"  类别{cls:2d}: {count:6d}像素 ({ratio:5.1f}%)")

# ⚠️ 关键检查：是否只预测一个类
if len(unique_preds) == 1:
    print(f"  ❌ 错误：模型只预测了类别 {unique_preds[0].item()}！")
    print(f"     这是典型的模型collapse，需要检查训练过程")
elif len(unique_preds) < 5:
    print(f"  ⚠️  警告：预测类别过少，模型可能收敛到局部最优")
else:
    print(f"  ✅ 正常：预测类别多样")
# === 诊断代码结束 ===

metrics.update(preds, labels)
```

**期望输出**：
```
[🔍 诊断] Batch 0/100:
  预测的唯一类别数: 25/37     ← 应该>5
  预测的类别ID: [0, 1, 2, 3, 4, ...]
  类别 0: 12000像素 (52.2%)
  类别 1:  5000像素 (21.7%)
  类别 2:  3000像素 (13.0%)
  ...
  ✅ 正常：预测类别多样
```

**如果只有1个类别**：
```
[🔍 诊断] Batch 0/100:
  预测的唯一类别数: 1/37       ← ❌ 问题！
  预测的类别ID: [0]
  类别 0: 230400像素 (100.0%)   ← ❌ 全部预测为类0
  ❌ 错误：模型只预测了类别 0！
```

---

### 步骤2: 检查Loss和梯度

在 `utils/train.py` 第480行附近添加：

```python
# 原代码
loss = model(imgs, modal_xs, label=gts, text_features=text_feats)
loss.backward()

# === 添加诊断代码 ===
if train_iteration % 50 == 0:  # 每50次迭代打印一次
    print(f"\n[🔍 诊断] Iteration {train_iteration}:")
    print(f"  Loss: {loss.item():.6f}")

    # 检查梯度范数
    total_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    print(f"  梯度总范数: {total_norm:.6f}")

    # 梯度异常检查
    if total_norm < 1e-7:
        print(f"  ❌ 错误：梯度过小 ({total_norm:.2e})，可能梯度消失！")
        print(f"     解决：检查学习率、激活函数、层数")
    elif total_norm > 1000:
        print(f"  ❌ 错误：梯度过大 ({total_norm:.2e})，可能梯度爆炸！")
        print(f"     解决：降低学习率、使用梯度裁剪")
    elif torch.isnan(torch.tensor(total_norm)):
        print(f"  ❌ 错误：梯度为NaN！")
        print(f"     解决：检查输入数据范围、学习率")
    else:
        print(f"  ✅ 正常：梯度范数合理")
# === 诊断代码结束 ===
```

**期望输出**：
```
[🔍 诊断] Iteration 50:
  Loss: 1.234567
  梯度总范数: 5.432100
  ✅ 正常：梯度范数合理
```

**异常情况**：
```
[🔍 诊断] Iteration 50:
  Loss: 0.000001              ← Loss过小
  梯度总范数: 0.000000001     ← 梯度接近0
  ❌ 错误：梯度过小，可能梯度消失！
```

---

### 步骤3: 检查混淆矩阵计算

在 `utils/metrics_new.py` 第24-31行添加诊断：

```python
def compute_iou(self) -> Tuple[Tensor, Tensor]:
    # === 添加诊断代码 ===
    print(f"\n[🔍 混淆矩阵诊断]:")
    print(f"  矩阵shape: {self.hist.shape}")
    print(f"  矩阵总和: {self.hist.sum().item()}")
    print(f"  对角线和: {self.hist.diag().sum().item()}")

    # 检查每个类别的样本数
    class_totals = self.hist.sum(dim=1)  # 每个类的真实样本数
    pred_totals = self.hist.sum(dim=0)   # 每个类的预测数
    for i in range(min(10, self.num_classes)):
        if class_totals[i] > 0 or pred_totals[i] > 0:
            print(f"  类别{i:2d}: GT={class_totals[i]:6.0f}, Pred={pred_totals[i]:6.0f}, "
                  f"Correct={self.hist[i,i]:6.0f}")
    # === 诊断代码结束 ===

    ious = self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1) - self.hist.diag())
    ious[ious.isnan()] = 0.0
    miou = ious.mean().item()
    ious *= 100
    miou *= 100
    return ious.cpu().numpy().round(2).tolist(), round(miou, 2)
```

**期望输出**：
```
[🔍 混淆矩阵诊断]:
  矩阵shape: torch.Size([37, 37])
  矩阵总和: 2304000.0
  对角线和: 1152000.0
  类别 0: GT=500000, Pred=450000, Correct=400000
  类别 1: GT=300000, Pred=350000, Correct=250000
  ...
```

**异常情况1（模型collapse）**：
```
[🔍 混淆矩阵诊断]:
  矩阵shape: torch.Size([37, 37])
  矩阵总和: 2304000.0
  对角线和: 500000.0
  类别 0: GT=500000, Pred=2304000, Correct=500000  ← 所有预测都是类0
  类别 1: GT=300000, Pred=0, Correct=0             ← 从未预测类1
  类别 2: GT=200000, Pred=0, Correct=0
  ...
```

**异常情况2（混淆矩阵全零）**：
```
[🔍 混淆矩阵诊断]:
  矩阵shape: torch.Size([37, 37])
  矩阵总和: 0.0              ← 矩阵为空！
  对角线和: 0.0
```
如果出现这个，检查：
- `metrics.update()` 是否被正确调用
- `ignore_label` 设置是否正确（SUNRGBD应该是-1）

---

### 步骤4: 检查数据加载

在训练循环开始时添加：

```python
# 在 utils/train.py 第460行附近
for imgs, modal_xs, gts, text_feats in trainloader:

    # === 只在第一个batch打印 ===
    if train_iteration == 1:
        print(f"\n[🔍 数据加载诊断]:")
        print(f"  图片shape: {imgs.shape}")
        print(f"  标签shape: {gts.shape}")
        print(f"  标签范围: [{gts.min().item()}, {gts.max().item()}]")
        print(f"  标签唯一值: {torch.unique(gts).tolist()}")

        # 检查ignore_label
        ignore_count = (gts == config.background).sum().item()
        total_pixels = gts.numel()
        print(f"  ignore_label ({config.background}) 占比: {ignore_count}/{total_pixels} ({ignore_count/total_pixels*100:.1f}%)")

        # 检查类别分布
        print(f"  类别分布（前10个）:")
        for cls in range(min(10, config.num_classes)):
            count = (gts == cls).sum().item()
            if count > 0:
                print(f"    类别{cls:2d}: {count:6d}像素 ({count/total_pixels*100:5.2f}%)")

        print(f"  config.num_classes: {config.num_classes}")
        print(f"  config.background: {config.background}")
    # === 诊断代码结束 ===
```

**期望输出**：
```
[🔍 数据加载诊断]:
  图片shape: torch.Size([16, 3, 480, 480])
  标签shape: torch.Size([16, 480, 480])
  标签范围: [-1, 36]                    ← 正常：-1是ignore, 0-36是类别
  标签唯一值: [-1, 0, 1, 2, 3, ..., 36]
  ignore_label (-1) 占比: 500000/3686400 (13.6%)
  类别分布（前10个）:
    类别 0: 1200000像素 (32.56%)
    类别 1:  800000像素 (21.70%)
    ...
  config.num_classes: 37
  config.background: -1
```

**异常情况**：
```
[🔍 数据加载诊断]:
  图片shape: torch.Size([16, 3, 480, 480])
  标签shape: torch.Size([16, 480, 480])
  标签范围: [0, 37]                     ← ❌ 超出范围！应该是0-36
  标签唯一值: [0, 1, 2, ..., 37]
  config.num_classes: 37               ← 类别数37，但标签有37（0-36）
```

---

## 快速检查清单

运行训练前，确认：

- [ ] `config.num_classes = 37` (SUNRGBD有37类)
- [ ] `config.background = -1` (ignore_label)
- [ ] 学习率合理（通常8e-5）
- [ ] 预训练权重加载成功
- [ ] 数据增强不会破坏标签

运行训练时，观察：

- [ ] Loss是否在下降（前几个epoch应该明显下降）
- [ ] 梯度范数是否合理（不要太小<1e-7或太大>1000）
- [ ] 模型预测是否多样化（unique类别数应该>10）
- [ ] mIoU是否增长（即使很慢也应该有变化）

---

## 常见问题及解决

### 问题1：mIoU固定在2.7%
**原因**：模型只预测类0（100/37 ≈ 2.7%）
**解决**：
1. 检查Loss是否在下降
2. 降低学习率（从8e-5降到2e-5）
3. 检查类别权重是否设置
4. 延长warmup周期

### 问题2：mIoU=0.0
**原因**：混淆矩阵全零或计算错误
**解决**：
1. 检查`metrics.update()`是否被调用
2. 检查`ignore_label`设置
3. 检查标签值范围

### 问题3：Loss=NaN
**原因**：数值溢出
**解决**：
1. 降低学习率10倍
2. 使用梯度裁剪
3. 检查输入归一化

### 问题4：Loss不下降
**原因**：学习率过小或梯度消失
**解决**：
1. 增大学习率
2. 检查预训练权重是否加载
3. 检查BN层是否冻结

---

## 一键诊断命令

在训练启动时添加 `--debug` 参数，然后在代码中：

```python
if args.debug:
    # 自动启用所有诊断代码
    import utils.debug_mode
    utils.debug_mode.enable_all_diagnostics(model, config)
```

---

## 联系信息

如果以上步骤都无法解决问题，请提供：

1. 完整的训练日志（前100个iteration）
2. 诊断输出（预测多样性、Loss、梯度）
3. config配置文件
4. 使用的预训练权重

这样可以更快定位问题！
