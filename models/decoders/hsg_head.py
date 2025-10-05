import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

# 导入 decode_head 基类和我们新创建的 SAM 模块
from .decode_head import BaseDecodeHead

try:
    from ..blocks.semantic_alignment import SemanticAlignmentModule
except ImportError:
    import sys
    import os

    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
    from models.blocks.semantic_alignment import SemanticAlignmentModule


class PPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet."""

    def __init__(self, pool_scales, in_channels, channels, conv_cfg, norm_cfg,
                 act_cfg, align_corners):
        super().__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    ConvModule(
                        self.in_channels,
                        self.channels,
                        1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg)))

    def forward(self, x):
        """Forward function."""
        ppm_outs = []
        for ppm_conv in self:
            ppm_out = ppm_conv(x)
            upsampled_ppm_out = F.interpolate(
                ppm_out,
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs


class HierarchicalSemanticGuidedHead(BaseDecodeHead):
    """
    Hierarchical Semantic-Guided Head (HSG-Head).

    This decoder head is based on the UPerNet architecture. Its key innovation is the
    integration of the SemanticAlignmentModule (SAM) at each stage of the Feature
    Pyramid Network (FPN) fusion process. This ensures that textual guidance is applied
    consistently during the feature upsampling and reconstruction phase, creating a
    symmetric, end-to-end guided model.
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), text_dim=512, **kwargs):

        super().__init__(input_transform='multiple_select', **kwargs)
        self.text_dim = text_dim

        # PSP Module
        self.psp_module = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.decoder_sam_stages = nn.ModuleList()

        for idx, in_channels in enumerate(self.in_channels):
            self.decoder_sam_stages.append(
                SemanticAlignmentModule(query_dim=in_channels, text_dim=text_dim)
            )
            if idx < len(self.in_channels) - 1:
                l_conv = ConvModule(
                    in_channels,
                    self.channels,
                    1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    inplace=False)
                fpn_conv = ConvModule(
                    self.channels,
                    self.channels,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    inplace=False)
                self.lateral_convs.append(l_conv)
                self.fpn_convs.append(fpn_conv)

            # FPN conv for the top (psp-enhanced) stage
        self.fpn_convs.append(
            ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False,
            )
        )

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def _apply_sam(self, feature, stage_idx, text_features):
        b, c, h, w = feature.shape
        visual = feature.permute(0, 2, 3, 1).contiguous()
        guided = self.decoder_sam_stages[stage_idx](visual, text_features)
        return guided.permute(0, 3, 1, 2).contiguous()

    def psp_forward(self, x):
        """Forward function of PSP module."""
        psp_outs = [x]
        psp_outs.extend(self.psp_module(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output


    def forward(self, inputs, text_features):

        """Forward function with textual guidance."""
        if text_features is None:
            raise ValueError("text_features must be provided for HierarchicalSemanticGuidedHead")
        if text_features.dim() == 2:
            text_features = text_features.unsqueeze(0)

        inputs = self._transform_inputs(inputs)

        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            guided_features = self._apply_sam(inputs[i], i, text_features)
            laterals.append(lateral_conv(guided_features))

        top_guided = self._apply_sam(inputs[-1], len(self.in_channels) - 1, text_features)
        laterals.append(self.psp_forward(top_guided))

        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        fpn_outs = []
        for i in range(used_backbone_levels):
            fpn_outs.append(self.fpn_convs[i](laterals[i]))

        fpn_outs.reverse()

        for i in range(1, used_backbone_levels):
            fpn_outs[i] = F.interpolate(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)

        fpn_outs = torch.cat(fpn_outs, dim=1)
        output = self.fpn_bottleneck(fpn_outs)
        output = self.cls_seg(output)

        return output