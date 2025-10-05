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

    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from models.blocks.semantic_alignment import SemanticAlignmentModule


class PPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet."""

    def __init__(self, pool_scales, in_channels, channels, conv_cfg, norm_cfg,
                 act_cfg, align_corners):
        super(PPM, self).__init__()
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
    Pyramid Network (FPN) fusion process. This ensures that textual guidance is
    applied consistently during the feature upsampling and reconstruction phase,
    creating a symmetric, end-to-end guided model.
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), text_dim=512, **kwargs):
        super(HierarchicalSemanticGuidedHead, self).__init__(
            input_transform='multiple_select', **kwargs)

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

        for in_channels in self.in_channels[:-1]:  # skip the top layer
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

            # query_dim is in_channels from the encoder for that level
            # We convert to (B, H, W, C) for SAM, so the dim is correct.
            sam_module = SemanticAlignmentModule(query_dim=in_channels, text_dim=text_dim)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            self.decoder_sam_stages.append(sam_module)

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_module(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output


    def forward(self, inputs, text_features):
        """
        The forward function now accepts `text_features` to guide the decoding process.
        """
        # `inputs` is a tuple of features from our modified DFormerv2 encoder.
        # It's already been guided by SAM at each stage.
        inputs = self._transform_inputs(inputs)

        # build laterals
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            # -----------------------------------------------------------------
            # 在FPN的横向连接前，再次使用SAM进行语义对齐
            # 此时的输入inputs[i]已经是被Encoder的SAM引导过的特征
            # 格式为 (B, C, H, W), 需要转换
            # -----------------------------------------------------------------
            visual_features = inputs[i].permute(0, 2, 3, 1).contiguous()
            guided_features = self.decoder_sam_stages[i](visual_features, text_features)
            guided_features = guided_features.permute(0, 3, 1, 2).contiguous()

            laterals.append(lateral_conv(guided_features))

        # build top-down path
        laterals.append(self.psp_forward(inputs))

        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs
        fpn_outs = []
        for i in range(used_backbone_levels):
            fpn_outs.append(self.fpn_convs[i](laterals[i]))

        fpn_outs.reverse()

        # Concatenate all levels
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