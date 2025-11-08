# Infer.py 高级功能说明

## 新增功能总览

本次更新为 `infer.py` 添加了三大核心功能：

1. **分割结果可视化输出** - 保存彩色分割预测图
2. **Per-Image Token 信息记录** - 导出每张图片使用的文本token详情
3. **多Stage Attention 可视化** - 支持同时可视化多个阶段的attention或聚合展示

---

## 1. 分割结果可视化输出

### 功能描述
自动保存模型的最终分割预测结果为彩色图像，使用数据集特定的调色板。

### 输出位置
```
<save_path>/predictions/<image_name>_pred.png
```

### 使用示例
```bash
python utils/infer.py \
    --config local_configs/NYUDepthv2/DFormerv2_S.py \
    --gpus 1 \
    --continue_fpath checkpoints/model.pth \
    --save-attention \
    --save_path ./output \
    --num-images 10
```

**输出文件结构：**
```
output/
├── predictions/          # 分割结果
│   ├── RGB_0_pred.png
│   ├── RGB_1_pred.png
│   └── ...
├── attention/           # Attention 可视化
│   └── enc_stage0_block2/
│       ├── RGB_0__imglabel_wall_attn.png
│       └── ...
└── token_info/          # Token 信息
    ├── RGB_0_tokens.json
    └── ...
```

### 调色板支持
- **NYUDepthv2/SUNRGBD**: 使用 `utils/nyucmap.npy` 调色板
- **其他数据集**: 使用默认 Cityscapes 风格调色板

---

## 2. Per-Image Token 信息输出

### 功能描述
为每张图片生成一个 JSON 文件，记录该图使用的所有文本 token 及其类型。

### 输出格式
```json
{
  "image": "RGB_0",
  "num_tokens": 6,
  "tokens": [
    {
      "index": 0,
      "name": "wall",
      "type": "imglabel"
    },
    {
      "index": 1,
      "name": "floor",
      "type": "imglabel"
    },
    ...
  ]
}
```

### 字段说明
- **image**: 图片文件名
- **num_tokens**: Token 总数
- **tokens**: Token 列表
  - **index**: Token 索引
  - **name**: Token 名称（类别名、caption句子等）
  - **type**: Token 类型
    - `imglabel` - 图片级标签（per-image labels）
    - `class` - 全局类别标签
    - `caption` - Caption 句子

### 使用场景
- 调试文本引导机制
- 分析哪些token被用于特定图片
- 验证 per-image labels 是否正确加载
- 追踪 attention 可视化中的 token 来源

---

## 3. 多Stage Attention 可视化

### 核心改进

之前只能可视化**单个 stage 的单个 block**，现在支持：
- ✅ 同时可视化多个 stages
- ✅ 可视化所有 stages（自动检测）
- ✅ 聚合多个 stages 的 attention（3种模式）

### 新增参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--vis-stage-idx` | str | "0" | Stage索引：单个 "0"，多个 "0,1,2"，全部 "all" |
| `--vis-aggregate` | str | "none" | 聚合模式：none/mean/max/weighted |

### 使用示例

#### 示例1: 可视化单个 stage（原有功能）
```bash
python utils/infer.py \
    --config local_configs/NYUDepthv2/DFormerv2_S.py \
    --gpus 1 \
    --continue_fpath checkpoints/model.pth \
    --save-attention \
    --save_path ./output \
    --vis-stage enc \
    --vis-stage-idx "2"
```
**输出：** `output/attention/enc_stage2_block2/`

#### 示例2: 可视化多个 stages（分别保存）
```bash
python utils/infer.py \
    --config local_configs/NYUDepthv2/DFormerv2_S.py \
    --gpus 1 \
    --continue_fpath checkpoints/model.pth \
    --save-attention \
    --save_path ./output \
    --vis-stage enc \
    --vis-stage-idx "0,1,2"  # 同时可视化 stage 0,1,2
```

**输出：**
```
output/attention/
├── enc_stage0_block2/
│   └── ...
├── enc_stage1_block2/
│   └── ...
└── enc_stage2_block2/
    └── ...
```

#### 示例3: 可视化所有 stages
```bash
python utils/infer.py \
    --config local_configs/NYUDepthv2/DFormerv2_S.py \
    --gpus 1 \
    --continue_fpath checkpoints/model.pth \
    --save-attention \
    --save_path ./output \
    --vis-stage enc \
    --vis-stage-idx "all"  # 自动检测并可视化所有stage
```

#### 示例4: 聚合多个 stages（平均）
```bash
python utils/infer.py \
    --config local_configs/NYUDepthv2/DFormerv2_S.py \
    --gpus 1 \
    --continue_fpath checkpoints/model.pth \
    --save-attention \
    --save_path ./output \
    --vis-stage enc \
    --vis-stage-idx "0,1,2" \
    --vis-aggregate mean  # 计算多个stage的平均attention
```

**输出：**
```
output/attention/
├── aggregated_mean_stage0_stage1_stage2/  # 聚合结果
│   └── ...
├── enc_stage0_block2/  # 原始 stage 0
│   └── ...
├── enc_stage1_block2/  # 原始 stage 1
│   └── ...
└── enc_stage2_block2/  # 原始 stage 2
    └── ...
```

### 聚合模式详解

#### 1. `none` (默认)
不聚合，为每个 stage 单独保存 attention maps。

**适用场景：**
- 详细分析每个 stage 的注意力模式
- 对比不同 stage 的表现
- 论文插图需要展示多层次特征

#### 2. `mean`
计算所有 stage 的平均 attention。

**公式：** `A_agg = (A_0 + A_1 + ... + A_n) / n`

**适用场景：**
- 获得整体的语义关注趋势
- 平滑噪声
- 综合不同层次的信息

#### 3. `max`
对每个位置取所有 stage 的最大值。

**公式：** `A_agg[i,j] = max(A_0[i,j], A_1[i,j], ..., A_n[i,j])`

**适用场景：**
- 突出显示最显著的注意力区域
- 保留每个 stage 的峰值响应
- 强调关键特征区域

#### 4. `weighted`
根据 attention 强度加权平均。

**公式：**
```
w_i = softmax(sum(A_i))
A_agg = Σ(w_i * A_i)
```

**适用场景：**
- 自适应融合，强调信息量大的 stage
- 自动平衡不同 stage 的贡献
- 智能聚合多层次特征

### 聚合模式对比实验

```bash
# 创建对比实验
for mode in none mean max weighted; do
    python utils/infer.py \
        --config local_configs/NYUDepthv2/DFormerv2_S.py \
        --gpus 1 \
        --continue_fpath checkpoints/model.pth \
        --save-attention \
        --save_path ./output_${mode} \
        --vis-stage enc \
        --vis-stage-idx "all" \
        --vis-aggregate ${mode} \
        --num-images 5
done
```

---

## 完整使用示例

### 场景1: 调试 Per-Image Labels

```bash
python utils/infer.py \
    --config local_configs/NYUDepthv2/DFormerv2_S.py \
    --gpus 1 \
    --continue_fpath checkpoints/best.pth \
    --save-attention \
    --save_path ./debug_output \
    --num-images 10 \
    --random-select \
    --vis-stage enc \
    --vis-stage-idx "2" \
    --filter-tokens "wall,floor,ceiling"
```

**输出：**
- ✅ 分割结果图：`debug_output/predictions/`
- ✅ Token信息：`debug_output/token_info/` （检查每张图用了哪些token）
- ✅ Attention可视化：`debug_output/attention/` （只显示 wall, floor, ceiling 的attention）

### 场景2: 论文插图 - 多层次特征可视化

```bash
python utils/infer.py \
    --config local_configs/NYUDepthv2/DFormerv2_S.py \
    --gpus 1 \
    --continue_fpath checkpoints/best.pth \
    --save-attention \
    --save_path ./paper_figs \
    --image-indices "42,108,256" \
    --vis-stage enc \
    --vis-stage-idx "0,1,2,3" \
    --vis-aggregate weighted \
    --vis-colormap plasma \
    --attention-alpha 0.7
```

**输出：**
- 3张特定图片的：
  - 4个独立 stage 的 attention maps
  - 1个加权聚合的 attention map
  - 最终分割结果

### 场景3: 分析 Decoder Attention

```bash
python utils/infer.py \
    --config local_configs/NYUDepthv2/DFormerv2_S.py \
    --gpus 1 \
    --continue_fpath checkpoints/best.pth \
    --save-attention \
    --save_path ./decoder_analysis \
    --num-images 20 \
    --vis-stage dec \
    --vis-stage-idx "all" \
    --vis-aggregate mean
```

---

## 关于 "最佳方案" 的讨论

### Q: 为什么之前只能可视化单个 stage/block？
**A:** 这是一个权衡：
- ✅ **优点**: 简单、明确、易于调试
- ❌ **缺点**: 无法对比不同层次、无法看整体趋势

### Q: 现在的多 stage 方案更好吗？
**A:** 取决于你的需求：

| 使用场景 | 推荐方案 |
|---------|---------|
| **调试模型** | 单 stage (`--vis-stage-idx "2"`) |
| **对比分析** | 多 stage 分别保存 (`--vis-stage-idx "0,1,2"`) |
| **论文可视化** | 聚合展示 (`--vis-aggregate weighted`) |
| **理解整体机制** | 所有 stage + 聚合 (`--vis-stage-idx "all" --vis-aggregate mean`) |

### Q: 哪种聚合模式最好？
**A:** 没有绝对的"最好"，根据目标选择：

- **理论分析** → `mean`: 展示平均趋势
- **突出关键区域** → `max`: 保留最强响应
- **平衡不同层次** → `weighted`: 自适应权重
- **详细研究** → `none`: 分别查看每层

### 建议工作流程

```bash
# 1. 先快速检查单个 stage（最快）
python utils/infer.py ... --vis-stage-idx "2"

# 2. 如果需要对比，可视化关键 stages
python utils/infer.py ... --vis-stage-idx "1,2,3"

# 3. 如果需要整体理解，使用聚合
python utils/infer.py ... --vis-stage-idx "all" --vis-aggregate weighted

# 4. 最终论文图，精调参数
python utils/infer.py ... \
    --vis-stage-idx "2" \
    --vis-colormap plasma \
    --attention-alpha 0.7 \
    --vis-gamma 0.8
```

---

## 技术细节

### Attention 聚合算法

不同 stage 的 attention maps 可能有不同的空间分辨率。聚合时会：
1. 以第一个有效 stage 的分辨率为目标
2. 使用双线性插值调整其他 stage 到相同分辨率
3. 按选定模式聚合（mean/max/weighted）

### 自动空间调整示例

```python
# Stage 1: 60x80 tokens, Stage 2: 30x40 tokens
# 自动将 Stage 2 插值到 60x80
# 然后计算聚合
```

### 文件命名规范

```
# 单 stage
enc_stage2_block1/<image>__<type>_<token>_attn.png

# 聚合
aggregated_mean_stage0_stage1_stage2/<image>__<type>_<token>_attn.png

# 分割结果
predictions/<image>_pred.png

# Token 信息
token_info/<image>_tokens.json
```

---

## 性能建议

### 内存优化
可视化所有 stages 会增加内存占用：

```bash
# 如果内存不足，分批处理
python utils/infer.py ... --num-images 5 --image-indices "0,1,2,3,4"
python utils/infer.py ... --num-images 5 --image-indices "5,6,7,8,9"
```

### 速度优化
聚合会增加计算时间：

```bash
# 只聚合，不保存原始 stages（未来可添加此选项）
# 当前版本会同时保存聚合 + 原始
```

---

## 故障排查

### Token 信息为空
**问题**: `token_info/*.json` 中 `num_tokens=0`

**原因**:
- `text_source` 配置不正确
- Per-image labels JSON 路径错误
- Dataset 没有返回 `text_token_meta`

**解决**:
```bash
# 检查配置
grep -r "text_source\|image_labels_json_path" local_configs/

# 验证 JSON 文件存在
ls datasets/NYUDepthv2/out.json
```

### Attention 全是黑色
**问题**: Attention maps 都是纯黑图

**原因**:
- SAM 未启用该 stage
- Text guidance 未开启

**解决**:
```bash
# 检查配置中的 SAM 开关
grep "sam_enc_stages\|sam_dec_stages" local_configs/NYUDepthv2/DFormerv2_S.py

# 确保 enable_text_guidance=True
grep "enable_text_guidance" local_configs/NYUDepthv2/DFormerv2_S.py
```

### 聚合失败
**问题**: 日志显示 "Skipping attention with incompatible shape"

**原因**: 不同 stage 的 token 数不一致（理论上不应该发生）

**解决**: 使用 `--vis-aggregate none` 分别查看每个 stage

---

## 总结

| 功能 | 状态 | 推荐场景 |
|------|------|---------|
| 分割结果保存 | ✅ 完整实现 | 所有场景都建议开启 |
| Token 信息导出 | ✅ 完整实现 | 调试 text guidance |
| 多 stage 可视化 | ✅ 完整实现 | 深入分析、论文撰写 |
| Attention 聚合 | ✅ 完整实现 | 整体理解、突出关键区域 |

**默认推荐配置**（平衡速度和信息量）：
```bash
--vis-stage enc \
--vis-stage-idx "2" \
--vis-aggregate none
```

**深度分析配置**（信息最全，速度较慢）：
```bash
--vis-stage enc \
--vis-stage-idx "all" \
--vis-aggregate weighted
```
