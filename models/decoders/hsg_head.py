# hsg_head.py
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmseg.ops import resize
from collections.abc import Mapping
from .decode_head import BaseDecodeHead
from .ham_head import Hamburger
from ..blocks.semantic_alignment import SemanticAlignmentModule


class SAMStack(nn.Module):
    """把同一层的若干个 SAM 串起来；无层时等价 Identity。"""
    def __init__(self, sam_layers=None):
        super().__init__()
        self.layers = nn.ModuleList(sam_layers or [])

    def forward(self, f_chw: torch.Tensor, text_features=None) -> torch.Tensor:
        if len(self.layers) == 0:
            return f_chw
        x = f_chw.permute(0, 2, 3, 1).contiguous()   # → NHWC
        for sam in self.layers:
            x = sam(x, text_features)                # decoder forward（Top-K 稀疏 + 置信旁路）
        return x.permute(0, 3, 1, 2).contiguous()    # → NCHW


class HierarchicalSemanticGuidedHead(BaseDecodeHead):
    """
    HSGHead（Ham 流程 + 逐层 SAMStack）
      [每层 SAMStack] → 上采样到同分辨率 → concat → 1x1 squeeze → Hamburger → 1x1 align → cls_seg
    """

    def __init__(self,
                 in_channels,
                 in_index,
                 input_transform="multiple_select",
                 channels=512,
                 num_classes=40,
                 norm_cfg=None,
                 # 文本 / 稀疏注意力参数 —— 与 builder 保持一致的名称
                 text_dim=512,
                 sam_dec_stages=(0, 1, 2, 3),
                 backbone_num_heads=(4, 4, 8, 16),
                 sam_dec_repeats=1,
                 sam_use_topk=True,
                 sam_top_m=5,
                 **kwargs):
        # 注意：这里不要把 sam_* 传给父类，避免 BaseDecodeHead 报“unexpected keyword”
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

        # === 为每个输入特征层构建一个 SAMStack（可能为空） ===
        # builder 里传参：sam_dec_stages、sam_use_topk、sam_top_m，与此一致。:contentReference[oaicite:1]{index=1}
        global_enable = set(list(sam_dec_stages or []))
        self.dec_sam_stacks = nn.ModuleList()

        for local_i, Cin in enumerate(self.in_channels):
            # 将 head 的 local 索引映射回 backbone 的全局 stage 索引
            global_idx = self.in_index[local_i] if isinstance(self.in_index, (list, tuple)) else local_i
            enabled = (global_idx in global_enable)

            repeat = 0
            if enabled:
                repeat = self._resolve_repeat_count(sam_dec_repeats, global_idx, local_i)
                repeat = max(int(repeat), 0)

            layers = []
            for _ in range(repeat):
                layers.append(
                    SemanticAlignmentModule(
                        query_dim=Cin,
                        text_dim=text_dim,
                        num_heads=self.backbone_num_heads[global_idx],
                        alpha_init=0.05,
                        attn_drop=0.0, proj_drop=0.0, ffn_drop=0.0,
                        # 关键：保持与 builder 的 sam_* 对应（内部模块仍用 use_topk/top_m 这两个参数名）
                        use_topk=bool(sam_use_topk),
                        top_m=int(sam_top_m),
                    )
                )
            self.dec_sam_stacks.append(SAMStack(layers))  # 空则为 Identity

        # === Ham 短路径 ===
        self.squeeze = ConvModule(
            sum(self.in_channels), channels, 1,
            conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg
        )
        self.hamburger = Hamburger(
            ham_channels=channels, ham_kwargs=dict(),
            norm_cfg=self.norm_cfg, **kwargs
        )
        self.align = ConvModule(
            channels, self.channels, 1,
            conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg
        )

    def _resolve_repeat_count(self, repeat_cfg, global_idx, local_idx):
        if isinstance(repeat_cfg, Mapping):
            if global_idx in repeat_cfg: return repeat_cfg[global_idx]
            if local_idx in repeat_cfg:  return repeat_cfg[local_idx]
        elif isinstance(repeat_cfg, (list, tuple)):
            if len(repeat_cfg) == len(self.in_channels):
                return repeat_cfg[local_idx]
            if len(repeat_cfg) > 0:
                return repeat_cfg[min(local_idx, len(repeat_cfg) - 1)]
        return int(repeat_cfg)

    # ----------------------------- 前向 -----------------------------
    def forward(self, inputs, text_features=None):
        feats = self._transform_inputs(inputs)

        tgt_hw = feats[0].shape[2:]
        proc_feats = []
        for i, f in enumerate(feats):
            # 一次性通过该层的 SAMStack（内部自处理 NHWC↔NCHW 和多层串联）
            f = self.dec_sam_stacks[i](f, text_features)
            f = resize(f, size=tgt_hw, mode="bilinear", align_corners=self.align_corners)
            proc_feats.append(f)

        x = torch.cat(proc_feats, dim=1)
        x = self.squeeze(x)
        x = self.hamburger(x)
        x = self.align(x)
        return self.cls_seg(x)
