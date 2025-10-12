import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from .decode_head import BaseDecodeHead

try:
    from ..blocks.semantic_alignment import SemanticAlignmentModule
except ImportError:
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
    from models.blocks.semantic_alignment import SemanticAlignmentModule


class PPM(nn.ModuleList):
    def __init__(self, pool_scales, in_channels, channels, conv_cfg, norm_cfg, act_cfg, align_corners):
        super().__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        for ps in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(ps),
                    ConvModule(self.in_channels, self.channels, 1,
                               conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
                )
            )

    def forward(self, x):
        outs = []
        for ppm_conv in self:
            ppm_out = ppm_conv(x)
            outs.append(F.interpolate(ppm_out, size=x.size()[2:], mode='bilinear', align_corners=self.align_corners))
        return outs


class HierarchicalSemanticGuidedHead(BaseDecodeHead):
    """
    UPerNet-like 解码头，在各层加入 SAM，并可通过 sam_dec_stages 控制启用层。
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), text_dim=512, sam_dec_stages=(0, 1, 2, 3), **kwargs):
        in_channels = kwargs.pop("in_channels")
        in_index = kwargs.pop("in_index", None)
        input_transform = kwargs.pop("input_transform", None)

        if isinstance(in_channels, int):
            in_channels = [in_channels]
        if in_index is None:
            in_index = list(range(len(in_channels)))
        elif isinstance(in_index, int):
            in_index = [in_index]

        input_transform = input_transform or ("multiple_select" if len(in_channels) > 1 else None)

        # SAM Top-K 策略（从 builder 传入）
        self.sam_use_topk = bool(kwargs.pop("sam_use_topk", True))
        self.sam_top_m = int(kwargs.pop("sam_top_m", 5))

        super().__init__(in_channels=in_channels, in_index=in_index, input_transform=input_transform, **kwargs)

        self.text_dim = text_dim
        self._sam_dec_enabled = set(int(x) for x in sam_dec_stages) if sam_dec_stages is not None else set([0, 1, 2, 3])

        # PSP on top
        self.psp_module = PPM(pool_scales, self.in_channels[-1], self.channels,
                              conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg,
                              align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels, 3, padding=1,
            conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg
        )

        # FPN
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.decoder_sam_stages = nn.ModuleList()

        for idx, in_ch in enumerate(self.in_channels):
            self.decoder_sam_stages.append(
                SemanticAlignmentModule(
                    query_dim=in_ch,
                    text_dim=text_dim,
                    use_topk=self.sam_use_topk,
                    top_m=self.sam_top_m,
                )
            )
            if idx < len(self.in_channels) - 1:
                self.lateral_convs.append(
                    ConvModule(in_ch, self.channels, 1,
                               conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, inplace=False)
                )
                self.fpn_convs.append(
                    ConvModule(self.channels, self.channels, 3, padding=1,
                               conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, inplace=False)
                )

        # 顶层 FPN conv
        self.fpn_convs.append(
            ConvModule(self.channels, self.channels, 3, padding=1,
                       conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, inplace=False)
        )

        # 冻结未启用的 decoder SAM
        for i, m in enumerate(self.decoder_sam_stages):
            if i not in self._sam_dec_enabled:
                for p in m.parameters():
                    p.requires_grad = False

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels, self.channels, 3, padding=1,
            conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg
        )

    def _apply_sam_if_enabled(self, feature_bchw, stage_idx, text_features):
        if (text_features is None) or (stage_idx not in self._sam_dec_enabled):
            return feature_bchw
        b, c, h, w = feature_bchw.shape
        visual = feature_bchw.permute(0, 2, 3, 1).contiguous()
        guided = self.decoder_sam_stages[stage_idx](visual, text_features)
        return guided.permute(0, 3, 1, 2).contiguous()

    def psp_forward(self, x):
        outs = [x]
        outs.extend(self.psp_module(x))
        outs = torch.cat(outs, dim=1)
        return self.bottleneck(outs)

    def forward(self, inputs, text_features=None):
        if (text_features is not None) and (text_features.dim() == 2):
            text_features = text_features.unsqueeze(0)
        inputs = self._transform_inputs(inputs)

        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            feat_i = self._apply_sam_if_enabled(inputs[i], i, text_features)
            laterals.append(lateral_conv(feat_i))

        top_feat = self._apply_sam_if_enabled(inputs[-1], len(self.in_channels) - 1, text_features)
        laterals.append(self.psp_forward(top_feat))

        used_levels = len(laterals)
        for i in range(used_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            up = F.interpolate(laterals[i], size=prev_shape, mode='bilinear', align_corners=self.align_corners)
            laterals[i - 1] = laterals[i - 1] + up

        fpn_outs = []
        for i in range(used_levels):
            fpn_outs.append(self.fpn_convs[i](laterals[i]))

        fpn_outs.reverse()
        target_size = fpn_outs[0].shape[2:]
        for i in range(1, used_levels):
            fpn_outs[i] = F.interpolate(fpn_outs[i], size=target_size, mode='bilinear', align_corners=self.align_corners)

        fpn_outs = torch.cat(fpn_outs, dim=1)
        output = self.fpn_bottleneck(fpn_outs)
        output = self.cls_seg(output)
        return output
