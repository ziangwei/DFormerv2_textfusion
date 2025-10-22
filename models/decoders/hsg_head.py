# hsg_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmseg.ops import resize

from .decode_head import BaseDecodeHead
from .ham_head import Hamburger
from ..blocks.semantic_alignment import SemanticAlignmentModule


class HierarchicalSemanticGuidedHead(BaseDecodeHead):
    """
    HSGHead（Ham 风格 + 逐层可控 SAM）

    路径：
      [每层(可选)SAM] → 上采样到同分辨率 → concat → 1x1 squeeze → Hamburger → 1x1 align → cls_seg

    逐层可控：
      - 通过 `sam_dec_stages`（全局 stage 索引，0/1/2/3）控制每个输入层是否启用 SAM。
      - 我们用 `in_index` 做“全局→本地”的映射，这样 builder 传 [1,2,3] 也能用同一套 0/1/2/3 索引语义。

    兼容性：
      - 保持类名、构造参数、forward 签名与旧版一致（外部调用/配置/训练脚本无需改）。
    """

    def __init__(self,
                 in_channels,                 # e.g., [64, 128, 256, 512] 或 [128, 256, 512]（取决于 builder）
                 in_index,                    # e.g., [0,1,2,3] 或 [1,2,3]
                 input_transform="multiple_select",
                 channels=512,                # ham_channels
                 num_classes=40,
                 norm_cfg=None,
                 # 文本与 SAM 控制参数（沿用你原来的命名）
                 text_dim=512,
                 sam_dec_stages=(0, 1, 2, 3),  # 全局 stage 索引集合
                 sam_use_topk=True,
                 sam_top_m=5,
                 backbone_num_heads=(4, 4, 8, 16),
                 **kwargs):
        super().__init__(in_channels=in_channels,
                         in_index=in_index,
                         input_transform=input_transform,
                         channels=channels,
                         num_classes=num_classes,
                         norm_cfg=norm_cfg,
                         **kwargs)

        self.ham_channels = channels
        self.text_dim = text_dim
        self.backbone_num_heads = list(backbone_num_heads)  # ★ 新增

        # === 逐层 SAM 准备（按全局索引控制） ===
        # 将外部传入的 sam_dec_stages（全局 0/1/2/3）映射到本地输入序列
        global_enable = set(list(sam_dec_stages or []))
        # BaseDecodeHead 会保存 self.in_index；其含的是“全局索引”
        # 构造一个“本地层 i 是否启用”的表
        self.dec_sam_enabled = []
        self.dec_sam_layers = nn.ModuleList()
        for local_i, Cin in enumerate(self.in_channels):
            global_idx = self.in_index[local_i] if isinstance(self.in_index, (list, tuple)) else local_i
            enabled = (global_idx in global_enable)
            self.dec_sam_enabled.append(enabled)
            if enabled:
                self.dec_sam_layers.append(
                    SemanticAlignmentModule(
                        query_dim=Cin,
                        text_dim=text_dim,
                        use_topk=sam_use_topk,
                        top_m=sam_top_m,
                        # 轻注入，避免“前期快、后期疲软”
                        num_heads=self.backbone_num_heads[global_idx],
                        alpha_init=0.05,
                        attn_drop=0.0, proj_drop=0.0, ffn_drop=0.0,
                    )
                )
            else:
                self.dec_sam_layers.append(None)

        # === Ham 风格短路径（与 LightHamHead 一致） ===
        # 多层上采样 → concat 后的 1x1 压到 ham_channels
        self.squeeze = ConvModule(
            sum(self.in_channels), self.ham_channels, 1,
            conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg
        )
        # NMF “汉堡块”
        self.hamburger = Hamburger(
            ham_channels=self.ham_channels, ham_kwargs=dict(),
            norm_cfg=self.norm_cfg, **kwargs
        )
        # 对齐层（回到 BaseDecodeHead 的 self.channels 作为 cls_seg 输入）
        self.align = ConvModule(
            self.ham_channels, self.channels, 1,
            conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg
        )

    # --------- 辅助：安全地对单层特征应用 SAM（BCHW <-> BHWC） ----------
    def _apply_sam_safe(self, x_bchw, sam_layer, text_features):
        if (sam_layer is None) or (text_features is None):
            return x_bchw

        # 若整批文本均无有效 token（全 0），直接旁路，避免 softmax(-inf) → NaN
        try:
            if isinstance(text_features, torch.Tensor) and text_features.numel() > 0:
                # 支持 (B,T,Ct) 或 (T,Ct)；统一成 (B, T, Ct) 再判断
                if text_features.dim() == 2:
                    tf = text_features.unsqueeze(0)  # 1,T,Ct
                elif text_features.dim() == 3:
                    tf = text_features
                else:
                    tf = None
                if tf is not None:
                    # 有效 token：任一 token 的 L1>0 视为有效
                    valid_any = (tf.abs().sum(dim=-1) > 0).any(dim=1)  # (B,)
                    if not bool(valid_any.any()):
                        return x_bchw
        except Exception:
            # 容错：一旦检查失败，不阻塞前向
            pass

        x_nhwc = x_bchw.permute(0, 2, 3, 1).contiguous()
        y_nhwc = sam_layer(x_nhwc, text_features)  # NHWC
        y_bchw = y_nhwc.permute(0, 3, 1, 2).contiguous()
        return y_bchw

    # ----------------------------- 前向 -----------------------------
    def forward(self, inputs, text_features=None):
        """
        inputs: 来自 backbone 的多层特征（list[Tensor]），长度由 builder 的 in_index 决定
        text_features: (B,T,Ct) 或 (T,Ct) 的文本嵌入；为 None 时自动跳过所有 SAM
        """
        feats = self._transform_inputs(inputs)  # list of BxCixHixWi

        # 逐层（可选）SAM，然后统一上采样、拼接
        tgt_hw = feats[0].shape[2:]
        proc_feats = []
        for i, f in enumerate(feats):
            f = self._apply_sam_safe(f, self.dec_sam_layers[i], text_features)
            f = resize(f, size=tgt_hw, mode="bilinear", align_corners=self.align_corners)
            proc_feats.append(f)

        x = torch.cat(proc_feats, dim=1)     # B, sum(Ci), H, W
        x = self.squeeze(x)                  # B, ham_channels, H, W
        x = self.hamburger(x)                # B, ham_channels, H, W
        x = self.align(x)                    # B, channels, H, W
        out = self.cls_seg(x)                # B, num_classes, H, W
        return out
