import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmseg.ops import resize

from .decode_head import BaseDecodeHead

try:
    from ..blocks.semantic_alignment import SemanticAlignmentModule
except ImportError:
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
    from models.blocks.semantic_alignment import SemanticAlignmentModule

class _MatrixDecomposition2DBase(nn.Module):
    def __init__(self, args=dict()):
        super().__init__()
        self.spatial = args.setdefault("SPATIAL", True)
        self.S = args.setdefault("MD_S", 1)
        self.D = args.setdefault("MD_D", 512)
        self.R = args.setdefault("MD_R", 64)
        self.train_steps = args.setdefault("TRAIN_STEPS", 6)
        self.eval_steps = args.setdefault("EVAL_STEPS", 7)
        self.inv_t = args.setdefault("INV_T", 100)
        self.eta = args.setdefault("ETA", 0.9)
        self.rand_init = args.setdefault("RAND_INIT", True)

    def _build_bases(self, B, S, D, R, cuda=False):
        raise NotImplementedError

    def local_step(self, x, bases, coef):
        raise NotImplementedError

    def local_inference(self, x, bases):
        coef = torch.bmm(x.transpose(1, 2), bases)
        coef = F.softmax(self.inv_t * coef, dim=-1)
        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef)
        return bases, coef

    def compute_coef(self, x, bases, coef):
        raise NotImplementedError

    def forward(self, x, return_bases=False):
        B, C, H, W = x.shape
        if self.spatial:
            D = C // self.S; N = H * W; x = x.view(B * self.S, D, N)
        else:
            D = H * W; N = C // self.S; x = x.view(B * self.S, N, D).transpose(1, 2)

        if self.rand_init:
            bases = self._build_bases(B, self.S, D, self.R, cuda=x.is_cuda)
        else:
            if not hasattr(self, "bases"):
                self.register_buffer("bases", self._build_bases(1, self.S, D, self.R, cuda=x.is_cuda))
            bases = self.bases.repeat(B, 1, 1)

        bases, coef = self.local_inference(x, bases)
        coef = self.compute_coef(x, bases, coef)
        x = torch.bmm(bases, coef.transpose(1, 2))
        if self.spatial: x = x.view(B, C, H, W)
        else:            x = x.transpose(1, 2).view(B, C, H, W)
        return x

class NMF2D(_MatrixDecomposition2DBase):
    def __init__(self, args=dict()):
        super().__init__(args); self.inv_t = 1
    def _build_bases(self, B, S, D, R, cuda=False):
        bases = torch.rand((B * S, D, R))
        if cuda: bases = bases.cuda()
        bases = F.normalize(bases, dim=1); return bases
    def local_step(self, x, bases, coef):
        numerator = torch.bmm(x.transpose(1, 2), bases)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        coef = coef * numerator / (denominator + 1e-6)
        numerator = torch.bmm(x, coef)
        denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))
        bases = bases * numerator / (denominator + 1e-6)
        return bases, coef
    def compute_coef(self, x, bases, coef):
        numerator = torch.bmm(x.transpose(1, 2), bases)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        coef = coef * numerator / (denominator + 1e-6)
        return coef

class Hamburger(nn.Module):
    def __init__(self, ham_channels=512, ham_kwargs=dict(), norm_cfg=None, **kwargs):
        super().__init__()
        self.ham_in  = ConvModule(ham_channels, ham_channels, 1, norm_cfg=None, act_cfg=None)
        self.ham     = NMF2D(ham_kwargs)
        self.ham_out = ConvModule(ham_channels, ham_channels, 1, norm_cfg=norm_cfg, act_cfg=None)
    def forward(self, x):
        y = F.relu(self.ham_in(x), inplace=True)
        y = self.ham(y)
        y = self.ham_out(y)
        return F.relu(x + y, inplace=True)  # 残差

class LightHamSAMHead(BaseDecodeHead):
    """
    HamHead 最小改动：squeeze → [SAM]*1 → Hamburger → align → cls_seg
    - 仅在 squeeze 之后单点引入 SAM（NHWC 对齐），不改动其余路径
    - 文本可选；无文本时等价于原 LightHamHead
    """
    def __init__(
        self,
        ham_channels=512,
        ham_kwargs=dict(),
        text_dim=512,
        enable_sam=True,
        sam_use_topk=True,
        sam_top_m=5,
        sam_decoder_use_cosine: bool = True,
        sam_decoder_learnable_temp: bool = True,
        sam_decoder_logit_init: float = 1 / 0.07,
        **kwargs,
    ):
        super().__init__(input_transform="multiple_select", **kwargs)
        self.ham_channels = ham_channels

        # 拼多尺度后压缩到 ham_channels（与原 HamHead 相同）
        self.squeeze = ConvModule(
            sum(self.in_channels),
            self.ham_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )

        # 单点 SAM（可关）
        self.enable_sam = enable_sam
        if self.enable_sam:
            self.sam = SemanticAlignmentModule(
                query_dim=self.ham_channels,
                text_dim=text_dim,
                use_topk=sam_use_topk,
                top_m=sam_top_m,
                # 轻注入，避免过强扰动
                alpha_init=0.05,
                attn_drop=0.0,
                proj_drop=0.0,
                ffn_drop=0.0,
                decoder_use_cosine=sam_decoder_use_cosine,
                decoder_learnable_temp=sam_decoder_learnable_temp,
                decoder_logit_scale_init=sam_decoder_logit_init,
            )

        # 原汉堡块与对齐层保持不变
        self.hamburger = Hamburger(ham_channels, ham_kwargs, **kwargs)
        self.align = ConvModule(
            self.ham_channels, self.channels, 1,
            conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg
        )

    def _apply_sam_if_needed(self, x_bchw, text_features):
        if (not self.enable_sam) or (text_features is None):
            return x_bchw
        b, c, h, w = x_bchw.shape
        x_nhwc = x_bchw.permute(0, 2, 3, 1).contiguous()
        y_nhwc = self.sam(x_nhwc, text_features)  # NHWC
        return y_nhwc.permute(0, 3, 1, 2).contiguous()

    def forward(self, inputs, text_features=None):
        # 与 HamHead 相同的多尺度对齐与拼接
        feats = self._transform_inputs(inputs)
        feats = [resize(f, size=feats[0].shape[2:], mode="bilinear", align_corners=self.align_corners) for f in feats]
        x = torch.cat(feats, dim=1)          # B, sum(Ci), H, W
        x = self.squeeze(x)                  # B, ham_channels, H, W

        # 仅此处插入 SAM；无文本则跳过 → 与原 HamHead 等价
        x = self._apply_sam_if_needed(x, text_features)

        x = self.hamburger(x)                # B, ham_channels, H, W
        x = self.align(x)                    # B, channels, H, W
        out = self.cls_seg(x)                # B, num_classes, H, W
        return out