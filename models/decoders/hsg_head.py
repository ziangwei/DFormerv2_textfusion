# hsg_head.py
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmseg.ops import resize
from collections.abc import Mapping
from .decode_head import BaseDecodeHead
from .ham_head import Hamburger
from ..blocks.semantic_alignment import SemanticAlignmentModule
import torch.nn.functional as F  # 🔧 新增：用于 F.interpolate 上采样geo_mask
import math                       # 🔧 新增：用于 math.sqrt 计算特征图尺寸


def sobel_xy(depth: torch.Tensor):
    # depth: [B,1,H,W]
    kx = depth.new_tensor([[[[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]]]])
    ky = depth.new_tensor([[[[-1.,-2.,-1.],[0.,0.,0.],[1.,2.,1.]]]])
    dx = F.conv2d(depth, kx, padding=1)
    dy = F.conv2d(depth, ky, padding=1)
    return dx, dy


class SAMStack(nn.Module):
    """把同一层的若干个 SAM 串起来；无层时等价 Identity。"""
    def __init__(self, sam_layers=None):
        super().__init__()
        self.layers = nn.ModuleList(sam_layers or [])

    def forward(self, f_chw, text_features=None,
                geo_mask=None,  # 🔧 新增
                return_attn=False):  # 🔧 新增
        if len(self.layers) == 0:
            if return_attn:
                return f_chw, None
            return f_chw

        x = f_chw.permute(0, 2, 3, 1).contiguous()
        attn_list = [] if return_attn else None

        for sam in self.layers:
            if return_attn:
                x, attn_i = sam(x, text_features, geo_mask, return_attn=True)
                attn_list.append(attn_i)
            else:
                x = sam(x, text_features, geo_mask, return_attn=False)

        out = x.permute(0, 3, 1, 2).contiguous()
        if return_attn:
            return out, attn_list
        return out


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

        self.dgn = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, 1)  # [:,0:1] -> gate_logit, [:,1:2] -> beta_logit
        )

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
    def forward(self, inputs, text_features=None,
                geo_priors=None,  # 🔧 新增：来自encoder的几何先验列表
                return_attn=False):  # 🔧 新增：是否返回attention maps
        feats = self._transform_inputs(inputs)

        attn_maps = [] if return_attn else None  # 🔧 收集attention

        tgt_hw = feats[0].shape[2:]
        proc_feats = []
        for i, f in enumerate(feats):
            # 🔧 处理 geo_mask
            geo_mask_i = None
            global_idx = self.in_index[i] if isinstance(self.in_index, (list, tuple)) else i
            if geo_priors is not None:
                try:
                    geo_raw = geo_priors[global_idx]  # 优先按全局 stage 取
                    if isinstance(geo_raw, torch.Tensor) and geo_raw.dim() == 4 and geo_raw.shape[1] == 1:
                        depth_i = F.interpolate(geo_raw, size=f.shape[-2:], mode='bilinear',
                                                align_corners=self.align_corners).to(f.dtype)
                        # 深度 + 梯度 -> DGN（可学习 gate/beta）
                        dx, dy = sobel_xy(depth_i)
                        d_in = torch.cat([depth_i, dx.abs(), dy.abs()], dim=1)  # [B,3,Hs,Ws]
                        d_out = self.dgn(d_in)  # [B,2,Hs,Ws]
                        gate = torch.sigmoid(d_out[:, 0:1])  # [B,1,Hs,Ws] ∈ (0,1)
                        beta = F.softplus(d_out[:, 1:2]) + 1.0  # [B,1,Hs,Ws] > 0
                        geo_mask_i = {"depth": depth_i, "gate": gate, "beta": beta}  # 传 dict
                    else:
                        geo_mask_i = geo_raw  # 兼容 [B,N]/[B,N,N]
                except Exception:
                    geo_mask_i = None
            else:
                geo_mask_i = None

            # 🔧 传给 SAMStack
            if return_attn:
                f, attn_i = self.dec_sam_stacks[i](f, text_features, geo_mask_i, return_attn=True)
                attn_maps.append(attn_i)
            else:
                f = self.dec_sam_stacks[i](f, text_features, geo_mask_i)

            f = resize(f, size=tgt_hw, mode="bilinear", align_corners=self.align_corners)
            proc_feats.append(f)

        x = torch.cat(proc_feats, dim=1)
        x = self.squeeze(x)
        x = self.hamburger(x)
        x = self.align(x)
        out = self.cls_seg(x)

        # 🔧 可选返回attention maps
        if return_attn:
            return out, attn_maps
        return out
