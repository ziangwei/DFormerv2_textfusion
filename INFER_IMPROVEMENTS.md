# Infer.py 功能改进说明

## 修复的问题

### 1. mIoU 差异问题 (20 vs 58)

**问题原因**：
- `infer.py` 之前使用单尺度评估
- `eval.py` 使用多尺度+翻转评估 `[0.5, 0.75, 1.0, 1.25, 1.5]` + flip

**解决方案**：
- 现在 `infer.py` 在非attention模式下默认使用与 `eval.py` 相同的多尺度+翻转评估
- 确保两个脚本计算的 mIoU 一致

### 2. 图片选择功能缺失

**新增功能**：
1. **顺序选择**：选择前N张图片
2. **随机选择**：随机选择N张图片
3. **指定索引**：选择特定索引的图片
4. **指定路径**：从文件读取图片路径列表

## 使用方法

### 1. 处理所有图片（默认，使用多尺度评估）
```bash
python utils/infer.py \
    --config local_configs/NYUDepthv2/DFormerv2_S.py \
    --gpus 1 \
    --continue_fpath checkpoints/model.pth
```

### 2. 选择前N张图片
```bash
python utils/infer.py \
    --config local_configs/NYUDepthv2/DFormerv2_S.py \
    --gpus 1 \
    --continue_fpath checkpoints/model.pth \
    --num-images 10
```

### 3. 随机选择N张图片
```bash
python utils/infer.py \
    --config local_configs/NYUDepthv2/DFormerv2_S.py \
    --gpus 1 \
    --continue_fpath checkpoints/model.pth \
    --num-images 10 \
    --random-select
```

### 4. 指定图片索引
```bash
python utils/infer.py \
    --config local_configs/NYUDepthv2/DFormerv2_S.py \
    --gpus 1 \
    --continue_fpath checkpoints/model.pth \
    --image-indices "0,5,10,15,20"
```

### 5. 从文件读取图片路径
```bash
# 创建图片路径文件 selected_images.txt
cat > selected_images.txt << EOF
RGB/0.jpg
RGB/10.jpg
RGB/20.jpg
EOF

# 运行推理
python utils/infer.py \
    --config local_configs/NYUDepthv2/DFormerv2_S.py \
    --gpus 1 \
    --continue_fpath checkpoints/model.pth \
    --image-paths selected_images.txt
```

### 6. Attention 可视化 + 图片选择
```bash
python utils/infer.py \
    --config local_configs/NYUDepthv2/DFormerv2_S.py \
    --gpus 1 \
    --continue_fpath checkpoints/model.pth \
    --save-attention \
    --num-images 5 \
    --random-select \
    --save_path ./vis_output \
    --vis-stage enc \
    --vis-stage-idx 2
```

## 新增参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `--num-images` | int | 要处理的图片数量（None=全部） |
| `--random-select` | flag | 随机选择图片而非顺序选择 |
| `--image-indices` | str | 逗号分隔的图片索引列表，如 "0,5,10" |
| `--image-paths` | str | 包含图片路径的文本文件路径（每行一个） |

## 注意事项

1. **图片选择优先级**：
   - `--image-indices` 最高优先级
   - `--image-paths` 次优先级
   - `--num-images` + `--random-select` 再次
   - `--num-images` 单独使用（顺序选择）
   - 默认处理全部图片

2. **Attention 可视化**：
   - 图片过滤在数据加载器层面完成
   - `--num-images` 参数现在会在可视化前过滤数据集
   - 可以结合 `--random-select` 或 `--image-indices` 使用

3. **mIoU 评估**：
   - 标准评估模式现在默认使用多尺度+翻转
   - 结果应该与 `eval.py` 一致
   - Attention 可视化模式仍然使用单尺度（为了保持可视化的一致性）

## 技术细节

### 实现方式
- 添加了 `SubsetDataset` 包装类来过滤数据集
- 在主函数中根据参数选择图片索引
- 重新创建 DataLoader 使用过滤后的数据集
- 保持了与原始代码的兼容性

### 代码修改位置
- `utils/infer.py` 第35-45行：新增 `SubsetDataset` 类
- `utils/infer.py` 第74-81行：新增命令行参数
- `utils/infer.py` 第769-830行：图片选择逻辑
- `utils/infer.py` 第914-975行：多尺度评估修复
