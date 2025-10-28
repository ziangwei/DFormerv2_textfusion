# hsg_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmseg.ops import resize
from collections.abc import Mapping, Sequence
from .decode_head import BaseDecodeHead
from .ham_head import Hamburger
from ..blocks.semantic_alignment import SemanticAlignmentModule


class HierarchicalSemanticGuidedHead(BaseDecodeHead):
    """
    HSGHead（Ham 风格 + 逐层可控 SAM）

    路径：
      [每层(可选)SAM] → 上采样到同分辨率 → concat → 1x1 squeeze → Hamburger → 1x1 align → cls_seg

    兼容性：
      - 保持类名、构造参数、forward 签名与旧版一致（外部调用/配置/训练脚本无需改）。
    """

    def __init__(self,
                 in_channels,
                 in_index,
                 input_transform="multiple_select",
                 channels=512,
                 num_classes=40,
                 norm_cfg=None,
                 # 文本与 SAM 控制参数（沿用你原来的命名）
                 text_dim=512,
                 sam_dec_stages=(0, 1, 2, 3),
                 sam_use_topk=False,
                 sam_top_m=5,
                 backbone_num_heads=(4, 4, 8, 16),
                 sam_dec_repeats=1,
                 dec_use_ssa: bool = False,   # ★ 仍默认 False：decoder 走 forward（我们在 forward 内加入 Top-p/置信门控/信任域/Token Drop）
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
        self.backbone_num_heads = list(backbone_num_heads)

        # === 逐层 SAM 准备（按全局索引控制） ===
        global_enable = set(list(sam_dec_stages or []))
        self.dec_sam_enabled = []
        self.dec_sam_layers = nn.ModuleList()
        self.dec_use_ssa = bool(dec_use_ssa)

        for local_i, Cin in enumerate(self.in_channels):
            global_idx = self.in_index[local_i] if isinstance(self.in_index, (list, tuple)) else local_i
            enabled = (global_idx in global_enable)
            self.dec_sam_enabled.append(enabled)
            repeat = self._resolve_repeat_count(sam_dec_repeats, global_idx, local_i) if enabled else 0
            repeat = max(int(repeat), 0)
            if enabled and repeat > 0:
                stage_layers = [
                    SemanticAlignmentModule(
                        query_dim=Cin,
                        text_dim=text_dim,
                        use_topk=sam_use_topk,
                        top_m=sam_top_m,
                        num_heads=self.backbone_num_heads[global_idx],
                        alpha_init=0.05,
                        attn_drop=0.0, proj_drop=0.0, ffn_drop=0.0,
                        # ★ 新增机制已在模块内默认开启：topp_p=0.9 / token_keep_prob=0.9 / trust_region_tau=0.5 / use_conf_gate=True
                    )
                    for _ in range(repeat)
                ]
                self.dec_sam_layers.append(nn.ModuleList(stage_layers))
            else:
                self.dec_sam_layers.append(nn.ModuleList())

        # === Ham 风格短路径 ===
        self.squeeze = ConvModule(
            sum(self.in_channels), self.ham_channels, 1,
            conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg
        )
        self.hamburger = Hamburger(
            ham_channels=self.ham_channels, ham_kwargs=dict(),
            norm_cfg=self.norm_cfg, **kwargs
        )
        self.align = ConvModule(
            self.ham_channels, self.channels, 1,
            conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg
        )

    # --------- 辅助：安全地对单层特征应用 SAM（BCHW <-> BHWC） ----------
    def _apply_sam_safe(self, x_bchw, sam_layers, text_features):
        if (text_features is None) or (sam_layers is None) or len(sam_layers) == 0:
            return x_bchw

        for sam_layer in sam_layers:
            x_bchw = self._apply_single_sam(x_bchw, sam_layer, text_features)
        return x_bchw

    def _apply_single_sam(self, x_bchw, sam_layer, text_features):
        if sam_layer is None:
            return x_bchw

        # 若整批文本均无有效 token（全 0），直接旁路，避免 softmax(-inf) → NaN
        try:
            if isinstance(text_features, torch.Tensor) and text_features.numel() > 0:
                if text_features.dim() == 2:
                    tf = text_features.unsqueeze(0)  # 1,T,Ct
                elif text_features.dim() == 3:
                    tf = text_features
                else:
                    tf = None
                if tf is not None:
                    valid_any = (tf.abs().sum(dim=-1) > 0).any(dim=1)  # (B,)
                    if not bool(valid_any.any()):
                        return x_bchw
        except Exception:
            pass

        x_nhwc = x_bchw.permute(0, 2, 3, 1).contiguous()
        # ★ decoder 走 forward（非 SSA）
        y_nhwc = sam_layer.forward_ssa(x_nhwc, text_features) if self.dec_use_ssa else sam_layer(x_nhwc, text_features)
        y_bchw = y_nhwc.permute(0, 3, 1, 2).contiguous()
        return y_bchw

    def _resolve_repeat_count(self, repeat_cfg, global_idx, local_idx):
        if isinstance(repeat_cfg, Mapping):
            if global_idx in repeat_cfg:
                return repeat_cfg[global_idx]
            if local_idx in repeat_cfg:
                return repeat_cfg[local_idx]
        elif isinstance(repeat_cfg, Sequence) and not isinstance(repeat_cfg, (str, bytes)):
            if len(repeat_cfg) == len(self.in_channels):
                return repeat_cfg[local_idx]
            if len(repeat_cfg) > 0:
                return repeat_cfg[min(local_idx, len(repeat_cfg) - 1)]
        return repeat_cfg

    # ----------------------------- 前向 -----------------------------
    def forward(self, inputs, text_features=None):
        """
        inputs: list[Tensor]（来自 backbone 的多层特征）
        text_features: (B,T,Ct) 或 (T,Ct)
        """
        feats = self._transform_inputs(inputs)

        # 逐层（可选）SAM，然后统一上采样、拼接
        tgt_hw = feats[0].shape[2:]
        proc_feats = []
        for i, f in enumerate(feats):
            f = self._apply_sam_safe(f, self.dec_sam_layers[i], text_features)
            f = resize(f, size=tgt_hw, mode="bilinear", align_corners=self.align_corners)
            proc_feats.append(f)

        x = torch.cat(proc_feats, dim=1)
        x = self.squeeze(x)
        x = self.hamburger(x)
        x = self.align(x)
        out = self.cls_seg(x)
        return out
