#!/usr/bin/env python3
"""
诊断mIoU不增长问题的脚本

可能的原因：
1. 模型预测全是同一个类（模型collapse）
2. 混淆矩阵计算错误
3. 标签预处理问题（ignore_label处理）
4. 梯度消失/爆炸
5. 学习率问题
"""

import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.metrics_new import Metrics


def diagnose_confusion_matrix():
    """诊断混淆矩阵计算逻辑"""
    print("=" * 60)
    print("1. 测试混淆矩阵计算")
    print("=" * 60)

    num_classes = 37  # SUNRGBD有37类
    ignore_label = -1
    device = torch.device("cpu")

    metric = Metrics(num_classes, ignore_label, device)

    # 测试1：完美预测
    print("\n[测试1] 完美预测（所有像素预测正确）")
    pred = torch.zeros(1, num_classes, 10, 10)  # [B, C, H, W]
    pred[:, 5, :, :] = 100  # 全部预测为类5
    target = torch.full((1, 10, 10), 5, dtype=torch.long)  # 全部真值为类5

    metric.hist.zero_()  # 重置
    metric.update(pred, target)

    print(f"混淆矩阵对角线（类5）: {metric.hist[5, 5].item()}")
    print(f"预期: 100 (10x10像素)")
    ious, miou = metric.compute_iou()
    print(f"类5的IoU: {ious[5]:.2f}%")
    print(f"mIoU: {miou:.2f}%")
    print(f"✅ 预期mIoU: 100.00% (因为其他类IoU为0，平均后会下降)")

    # 测试2：全预测同一类（模型collapse）
    print("\n[测试2] 模型collapse（全预测为类0）")
    pred = torch.zeros(1, num_classes, 10, 10)
    pred[:, 0, :, :] = 100  # 全部预测为类0
    target = torch.randint(0, num_classes, (1, 10, 10), dtype=torch.long)  # 真值随机

    metric.hist.zero_()
    metric.update(pred, target)

    print(f"混淆矩阵第0行（预测为类0的分布）:")
    print(f"  总像素数: {metric.hist[0, :].sum().item()}")
    print(f"  正确预测（对角线）: {metric.hist[0, 0].item()}")
    ious, miou = metric.compute_iou()
    print(f"mIoU: {miou:.2f}%")
    print(f"⚠️  mIoU应该很低且固定（因为预测始终是类0）")

    # 测试3：ignore_label处理
    print("\n[测试3] ignore_label处理")
    pred = torch.zeros(1, num_classes, 10, 10)
    pred[:, 5, :, :] = 100
    target = torch.full((1, 10, 10), 5, dtype=torch.long)
    target[:, :5, :] = -1  # 前5行标记为ignore

    metric.hist.zero_()
    metric.update(pred, target)

    print(f"混淆矩阵对角线（类5）: {metric.hist[5, 5].item()}")
    print(f"预期: 50 (后5行×10列)")
    ious, miou = metric.compute_iou()
    print(f"mIoU: {miou:.2f}%")
    print(f"✅ ignore_label被正确忽略")

    return metric


def diagnose_model_predictions():
    """诊断模型预测多样性"""
    print("\n" + "=" * 60)
    print("2. 检查模型预测多样性")
    print("=" * 60)
    print("\n请在训练循环中添加以下代码来诊断：")

    code = '''
# 在训练循环的validation部分添加（utils/val_mm.py 第67行附近）
with torch.no_grad():
    preds = model(images[0], images[1], text_features=text_feats).softmax(dim=1)

    # === 诊断代码开始 ===
    pred_classes = preds.argmax(dim=1)  # [B, H, W]
    unique_preds = torch.unique(pred_classes)

    print(f"[诊断] Batch {idx}:")
    print(f"  预测的唯一类别数: {len(unique_preds)}/{n_classes}")
    print(f"  预测的类别: {unique_preds.tolist()}")

    # 检查是否所有预测都是同一个类
    if len(unique_preds) == 1:
        print(f"  ⚠️ 警告：模型只预测了类别 {unique_preds[0].item()}！")

    # 检查预测分布
    for cls in unique_preds[:10]:  # 只显示前10个
        count = (pred_classes == cls).sum().item()
        ratio = count / pred_classes.numel() * 100
        print(f"  类别{cls}: {count}像素 ({ratio:.1f}%)")
    # === 诊断代码结束 ===

    metrics.update(preds, labels)
'''
    print(code)


def diagnose_loss_gradient():
    """诊断loss和梯度"""
    print("\n" + "=" * 60)
    print("3. 检查Loss和梯度")
    print("=" * 60)
    print("\n请在训练循环中添加以下代码来诊断：")

    code = '''
# 在训练循环中（utils/train.py 第480行附近）
loss = model(imgs, modal_xs, label=gts, text_features=text_feats)

# === 诊断代码开始 ===
print(f"[诊断] Iteration {train_iteration}:")
print(f"  Loss: {loss.item():.6f}")

# 检查loss是否在下降
if not hasattr(diagnose_loss_gradient, 'loss_history'):
    diagnose_loss_gradient.loss_history = []
diagnose_loss_gradient.loss_history.append(loss.item())

if len(diagnose_loss_gradient.loss_history) > 10:
    recent_loss = np.mean(diagnose_loss_gradient.loss_history[-10:])
    old_loss = np.mean(diagnose_loss_gradient.loss_history[-20:-10])
    if abs(recent_loss - old_loss) < 1e-6:
        print(f"  ⚠️ 警告：Loss停止下降！recent={recent_loss:.6f}, old={old_loss:.6f}")
# === 诊断代码结束 ===

loss.backward()

# === 检查梯度 ===
total_norm = 0.0
for name, param in model.named_parameters():
    if param.grad is not None:
        param_norm = param.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
total_norm = total_norm ** 0.5

print(f"  梯度总范数: {total_norm:.6f}")
if total_norm < 1e-7:
    print(f"  ⚠️ 警告：梯度过小，可能梯度消失！")
elif total_norm > 1000:
    print(f"  ⚠️ 警告：梯度过大，可能梯度爆炸！")
# === 诊断代码结束 ===
'''
    print(code)


def diagnose_data_loading():
    """诊断数据加载"""
    print("\n" + "=" * 60)
    print("4. 检查数据加载")
    print("=" * 60)
    print("\n请在训练循环中添加以下代码来诊断：")

    code = '''
# 在训练循环开始时（utils/train.py 第460行附近）
for imgs, modal_xs, gts in trainloader:

    # === 诊断代码开始 ===
    print(f"[诊断] 数据加载:")
    print(f"  图片shape: {imgs.shape}")
    print(f"  模态shape: {modal_xs.shape}")
    print(f"  标签shape: {gts.shape}")
    print(f"  标签唯一值: {torch.unique(gts).tolist()}")
    print(f"  标签范围: [{gts.min().item()}, {gts.max().item()}]")

    # 检查标签分布
    if hasattr(config, 'num_classes'):
        for cls in range(config.num_classes):
            count = (gts == cls).sum().item()
            if count > 0:
                print(f"    类别{cls}: {count}像素")

    # 检查ignore_label
    ignore_count = (gts == config.background).sum().item()
    print(f"  ignore_label像素数: {ignore_count}")
    # === 诊断代码结束 ===

    break  # 只检查第一个batch
'''
    print(code)


def create_quick_test_script():
    """创建快速测试脚本"""
    print("\n" + "=" * 60)
    print("5. 快速测试脚本")
    print("=" * 60)

    test_code = '''
# 保存为 test_single_batch.py
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.config import Config
from utils.dataloader import get_train_loader
from utils.metrics_new import Metrics

# 加载配置
config = Config.from_file("local_configs/SUNRGBD/DFormerv2_B.py")

# 加载数据
trainloader = get_train_loader(config, num_workers=4)

# 创建metric
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
metric = Metrics(config.num_classes, config.background, device)

print(f"数据集: {config.dataset_name}")
print(f"类别数: {config.num_classes}")
print(f"ignore_label: {config.background}")

# 测试一个batch
for batch_idx, minibatch in enumerate(trainloader):
    imgs = minibatch["data"]
    labels = minibatch["label"]
    modal_xs = minibatch["modal_x"]

    print(f"\\nBatch {batch_idx}:")
    print(f"  图片shape: {imgs.shape}")
    print(f"  标签shape: {labels.shape}")
    print(f"  标签唯一值: {torch.unique(labels).tolist()}")

    # 模拟随机预测
    B, H, W = labels.shape
    preds = torch.randn(B, config.num_classes, H, W, device=device)
    labels = labels.to(device)

    metric.update(preds, labels)
    ious, miou = metric.compute_iou()

    print(f"  mIoU (随机预测): {miou:.2f}%")
    print(f"  预期: ~{100/config.num_classes:.2f}% (随机猜测)")

    if batch_idx >= 2:
        break

print("\\n✅ 如果mIoU接近随机猜测的期望值，说明metric计算正确")
'''

    print("\n将以下代码保存为 test_single_batch.py 并运行：")
    print(test_code)


def main():
    print("\n" + "=" * 60)
    print("SUNRGBD训练mIoU不增长问题诊断工具")
    print("=" * 60)

    # 1. 测试混淆矩阵
    diagnose_confusion_matrix()

    # 2. 模型预测诊断
    diagnose_model_predictions()

    # 3. Loss和梯度诊断
    diagnose_loss_gradient()

    # 4. 数据加载诊断
    diagnose_data_loading()

    # 5. 创建快速测试脚本
    create_quick_test_script()

    # 总结
    print("\n" + "=" * 60)
    print("诊断步骤总结")
    print("=" * 60)
    print("""
1. 运行本脚本检查混淆矩阵计算是否正确
2. 在训练代码中添加预测多样性检查（第2节）
3. 在训练代码中添加loss/梯度检查（第3节）
4. 在训练代码中添加数据加载检查（第4节）
5. 运行test_single_batch.py验证基础功能

常见问题及解决方法：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
问题                               可能原因                    解决方法
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
mIoU固定在很低值                   模型预测全是同一类          检查loss、学习率、初始化
mIoU固定在0                        混淆矩阵计算错误            检查num_classes和ignore_label
mIoU=NaN                           除零错误                    检查是否有类别从未出现
Loss不下降                         学习率过小/梯度消失         增大学习率/检查梯度
Loss=NaN                           学习率过大/数值溢出         降低学习率/检查输入范围
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

推荐检查顺序：
1. 先检查数据加载（标签范围、分布是否正常）
2. 再检查模型预测（是否多样化）
3. 最后检查训练过程（loss、梯度）
""")


if __name__ == "__main__":
    main()
