import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math
from timm.models.layers import DropPath, trunc_normal_
from typing import Tuple
from mmengine.runner.checkpoint import load_state_dict
from collections import OrderedDict
from ..blocks.semantic_alignment import SemanticAlignmentModule


class _NoOpSAM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward_ssa(self, x, text_features=None, geo_mask=None, return_attn=False):
        """接受所有参数但什么都不做"""
        if return_attn:
            return x, None
        return x


class _NoOpStageSAM(nn.Module):
    """export-only 路径下需要的 stage 级 SAM 占位（调用 forward(x, text)）"""
    def __init__(self):
        super().__init__()

    def forward(self, x, text_features=None):
        return x


class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class PatchEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, 3, 2, 1),
            nn.SyncBatchNorm(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim // 2, 3, 1, 1),
            nn.SyncBatchNorm(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, 3, 2, 1),
            nn.SyncBatchNorm(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.SyncBatchNorm(embed_dim),
        )

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        return x


class DWConv2d(nn.Module):
    def __init__(self, dim, kernel_size, stride, padding):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        return x


class PatchMerging(nn.Module):
    def __init__(self, dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv2d(dim, out_dim, 3, 2, 1)
        self.norm = nn.SyncBatchNorm(out_dim)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.reduction(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)
        return x


def angle_transform(x, sin, cos):
    x1 = x[:, :, :, :, ::2]
    x2 = x[:, :, :, :, 1::2]
    return (x * cos) + (torch.stack([-x2, x1], dim=-1).flatten(-2) * sin)


class GeoPriorGen(nn.Module):
    def __init__(self, embed_dim, num_heads, initial_value, heads_range):
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.weight = nn.Parameter(torch.ones(2, 1, 1, 1), requires_grad=True)
        decay = torch.log(
            1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads)
        )
        self.register_buffer("angle", angle)
        self.register_buffer("decay", decay)

    def generate_depth_decay(self, H: int, W: int, depth_grid):
        B, _, H, W = depth_grid.shape
        grid_d = depth_grid.reshape(B, H * W, 1)
        mask_d = grid_d[:, :, None, :] - grid_d[:, None, :, :]
        mask_d = (mask_d.abs()).sum(dim=-1)
        mask_d = mask_d.unsqueeze(1) * self.decay[None, :, None, None]
        return mask_d

    def generate_pos_decay(self, H: int, W: int):
        index_h = torch.arange(H).to(self.decay)
        index_w = torch.arange(W).to(self.decay)
        grid = torch.meshgrid([index_h, index_w])
        grid = torch.stack(grid, dim=-1).reshape(H * W, 2)
        mask = grid[:, None, :] - grid[None, :, :]
        mask = (mask.abs()).sum(dim=-1)
        mask = mask * self.decay[:, None, None]
        return mask

    def generate_1d_depth_decay(self, H, W, depth_grid):
        mask = depth_grid[:, :, :, :, None] - depth_grid[:, :, :, None, :]
        mask = mask.abs()
        mask = mask * self.decay[:, None, None, None]
        assert mask.shape[2:] == (W, H, H)
        return mask

    def generate_1d_decay(self, l: int):
        index = torch.arange(l).to(self.decay)
        mask = index[:, None] - index[None, :]
        mask = mask.abs()
        mask = mask * self.decay[:, None, None]
        return mask

    def forward(self, HW_tuple: Tuple[int], depth_map, split_or_not=False):
        depth_map = F.interpolate(depth_map, size=HW_tuple, mode="bilinear", align_corners=False)
        if split_or_not:
            index = torch.arange(HW_tuple[0] * HW_tuple[1]).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :]).reshape(HW_tuple[0], HW_tuple[1], -1)
            cos = torch.cos(index[:, None] * self.angle[None, :]).reshape(HW_tuple[0], HW_tuple[1], -1)
            mask_d_h = self.generate_1d_depth_decay(HW_tuple[0], HW_tuple[1], depth_map.transpose(-2, -1))
            mask_d_w = self.generate_1d_depth_decay(HW_tuple[1], HW_tuple[0], depth_map)
            mask_h = self.generate_1d_decay(HW_tuple[0])
            mask_w = self.generate_1d_decay(HW_tuple[1])
            mask_h = self.weight[0] * mask_h.unsqueeze(0).unsqueeze(2) + self.weight[1] * mask_d_h
            mask_w = self.weight[0] * mask_w.unsqueeze(0).unsqueeze(2) + self.weight[1] * mask_d_w
            geo_prior = ((sin, cos), (mask_h, mask_w))
        else:
            index = torch.arange(HW_tuple[0] * HW_tuple[1]).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :]).reshape(HW_tuple[0], HW_tuple[1], -1)
            cos = torch.cos(index[:, None] * self.angle[None, :]).reshape(HW_tuple[0], HW_tuple[1], -1)
            mask = self.generate_pos_decay(HW_tuple[0], HW_tuple[1])
            mask_d = self.generate_depth_decay(HW_tuple[0], HW_tuple[1], depth_map)
            mask = self.weight[0] * mask + self.weight[1] * mask_d
            geo_prior = ((sin, cos), mask)
        return geo_prior


class Decomposed_GSA(nn.Module):
    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim**-0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)
        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim, bias=True)
        self.reset_parameters()

    def forward(self, x: torch.Tensor, rel_pos, split_or_not=False):
        bsz, h, w, _ = x.size()
        (sin, cos), (mask_h, mask_w) = rel_pos
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)
        k = k * self.scaling
        q = q.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)
        k = k.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)
        qr = angle_transform(q, sin, cos)
        kr = angle_transform(k, sin, cos)
        qr_w = qr.transpose(1, 2)
        kr_w = kr.transpose(1, 2)
        v = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 1, 3, 2, 4)
        qk_mat_w = qr_w @ kr_w.transpose(-1, -2)
        qk_mat_w = qk_mat_w + mask_w.transpose(1, 2)
        qk_mat_w = torch.softmax(qk_mat_w, -1)
        v = torch.matmul(qk_mat_w, v)
        qr_h = qr.permute(0, 3, 1, 2, 4)
        kr_h = kr.permute(0, 3, 1, 2, 4)
        v = v.permute(0, 3, 2, 1, 4)
        qk_mat_h = qr_h @ kr_h.transpose(-1, -2)
        qk_mat_h = qk_mat_h + mask_h.transpose(1, 2)
        qk_mat_h = torch.softmax(qk_mat_h, -1)
        output = torch.matmul(qk_mat_h, v)
        output = output.permute(0, 3, 1, 2, 4).flatten(-2, -1)
        output = output + lepe
        output = self.out_proj(output)
        return output

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2**-2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2**-2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2**-2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)


class Full_GSA(nn.Module):
    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim**-0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)
        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim, bias=True)
        self.reset_parameters()

    def forward(self, x: torch.Tensor, rel_pos, split_or_not=False):
        bsz, h, w, _ = x.size()
        (sin, cos), mask = rel_pos
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)
        k = k * self.scaling
        q = q.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4)
        k = k.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4)
        qr = angle_transform(q, sin, cos)
        kr = angle_transform(k, sin, cos)
        qr = qr.flatten(2, 3)
        kr = kr.flatten(2, 3)
        vr = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4).flatten(2, 3)
        qk_mat = qr @ kr.transpose(-1, -2)
        qk_mat = qk_mat + mask
        qk_mat = torch.softmax(qk_mat, -1)
        output = torch.matmul(qk_mat, vr).transpose(1, 2).reshape(bsz, h, w, -1)
        output = output + lepe
        output = self.out_proj(output)
        return output

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2**-2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2**-2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2**-2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)


class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, ffn_dim, activation_fn=F.gelu, dropout=0.0, activation_dropout=0.0,
                 layernorm_eps=1e-6, subln=False, subconv=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.activation_fn = activation_fn
        self.activation_dropout_module = torch.nn.Dropout(activation_dropout)
        self.dropout_module = torch.nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, self.embed_dim)
        self.ffn_layernorm = nn.LayerNorm(ffn_dim, eps=layernorm_eps) if subln else None
        self.dwconv = DWConv2d(ffn_dim, 3, 1, 1) if subconv else None

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        if self.ffn_layernorm is not None:
            self.ffn_layernorm.reset_parameters()

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.activation_dropout_module(x)
        residual = x
        if self.dwconv is not None:
            x = self.dwconv(x)
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = x + residual
        x = self.fc2(x)
        x = self.dropout_module(x)
        return x


class RGBD_Block(nn.Module):
    def __init__(self, split_or_not: str, embed_dim: int, num_heads: int, ffn_dim: int,
                 drop_path=0.0, layerscale=False, layer_init_values=1e-5,
                 init_value=2, heads_range=4):
        super().__init__()
        self.layerscale = layerscale
        self.embed_dim = embed_dim
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=1e-6)
        if split_or_not:
            self.Attention = Decomposed_GSA(embed_dim, num_heads)
        else:
            self.Attention = Full_GSA(embed_dim, num_heads)
        self.drop_path = DropPath(drop_path)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim)
        self.cnn_pos_encode = DWConv2d(embed_dim, 3, 1, 1)
        self.Geo = GeoPriorGen(embed_dim, num_heads, init_value, heads_range)
        if layerscale:
            self.gamma_1 = nn.Parameter(layer_init_values * torch.ones(1, 1, 1, embed_dim), requires_grad=True)
            self.gamma_2 = nn.Parameter(layer_init_values * torch.ones(1, 1, 1, embed_dim), requires_grad=True)

    def forward(self, x: torch.Tensor, x_e: torch.Tensor, split_or_not=False,
                sam_b: nn.Module = None, text_features: torch.Tensor = None, superpower: bool = False):
        x = x + self.cnn_pos_encode(x)
        b, h, w, d = x.size()
        geo_prior = self.Geo((h, w), x_e, split_or_not=split_or_not)

        # 自注意力
        out = self.Attention(self.layer_norm1(x), geo_prior, split_or_not)

        # ★ superpower=SSA-lite：在 GSA 之后、FFN 之前做一次轻量 SAM
        if superpower and (sam_b is not None) and (text_features is not None):
            # 🔧 不要从 geo_prior 提取，直接用特征图本身
            b, h, w, d = out.size()  # out 是 GSA 输出的特征

            # 方案A：传入当前分辨率的深度图
            depth_resized = F.interpolate(x_e, size=(h, w), mode='bilinear', align_corners=False)
            geo_mask = depth_resized  # [B, 1, H, W]

            out = sam_b.forward_ssa(out, text_features, geo_mask)

        # 残差1
        if self.layerscale:
            x = x + self.drop_path(self.gamma_1 * out)
        else:
            x = x + self.drop_path(out)

        # FFN（保持不变）
        if self.layerscale:
            x = x + self.drop_path(self.gamma_2 * self.ffn(self.layer_norm2(x)))
        else:
            x = x + self.drop_path(self.ffn(self.layer_norm2(x)))

        return x


class BasicLayer(nn.Module):
    def __init__(self, embed_dim, out_dim, depth, num_heads, init_value: float, heads_range: float,
                 ffn_dim=96.0, drop_path=0.0, norm_layer=nn.LayerNorm, split_or_not=False,
                 downsample: PatchMerging = None, use_checkpoint=False, layerscale=False, layer_init_values=1e-5):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.split_or_not = split_or_not
        self.blocks = nn.ModuleList(
            [
                RGBD_Block(
                    split_or_not, embed_dim, num_heads, ffn_dim,
                    drop_path[i] if isinstance(drop_path, list) else drop_path,
                    layerscale, layer_init_values, init_value=init_value, heads_range=heads_range,
                )
                for i in range(depth)
            ]
        )
        self.downsample = PatchMerging(dim=embed_dim, out_dim=out_dim, norm_layer=norm_layer) if downsample is not None else None

    def forward(self, x, x_e, text_features=None, sam_module=None, sam_blocks=None, superpower=False):
        for b_idx, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                # 注意：把本 block 对应的 sam_b/text_features/superpower 一并传给 RGBD_Block.forward
                x = checkpoint.checkpoint(blk, x=x, x_e=x_e, split_or_not=self.split_or_not,
                                          sam_b=(sam_blocks[b_idx] if (superpower and sam_blocks is not None and b_idx < len(sam_blocks)) else None),
                                          text_features=text_features,
                                          superpower=superpower)
            else:
                x = blk(x, x_e, split_or_not=self.split_or_not,
                        sam_b=(sam_blocks[b_idx] if (superpower and sam_blocks is not None and b_idx < len(sam_blocks)) else None),
                        text_features=text_features,
                        superpower=superpower)

        # 非 superpower（旧导出或 pre_down）时，如需在 stage 末做一次 SAM，仍走旧分支
        if (not superpower) and (sam_module is not None) and (text_features is not None):
            x = sam_module(x, text_features)

        if self.downsample is not None:
            x_down = self.downsample(x)
            return x, x_down
        else:
            return x, x


class dformerv2(nn.Module):
    def __init__(
        self,
        out_indices=(0, 1, 2, 3),
        embed_dims=[64, 128, 256, 512],
        depths=[3, 4, 18, 4],
        num_heads=[4, 4, 8, 16],
        init_values=[2, 2, 2, 2],
        heads_ranges=[4, 4, 6, 6],
        mlp_ratios=[4, 4, 3, 3],
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        use_checkpoint=False,
        norm_cfg=None,
        layerscales=[False, False, False, False],
        layer_init_values=1e-6,
        norm_eval=True,
        text_dim=512,
        sam_enc_stages=(0, 1, 2, 3),
        sam_use_topk=True,
        sam_top_m=5,
        superpower: bool = False,
        sam_enc_gamma_scale: float = 1.0,
    ):
        super().__init__()
        self.out_indices = out_indices
        self.num_layers = len(depths)
        self.patch_norm = patch_norm
        self.norm_eval = norm_eval
        self.num_heads = num_heads  # ★ 新增：保存各 stage 的头数

        # 哪些 encoder stage 启用 SAM
        self._sam_enc_enabled = set(int(x) for x in sam_enc_stages) if sam_enc_stages is not None else set([0, 1, 2, 3])

        self.patch_embed = PatchEmbed(in_chans=3, embed_dim=embed_dims[0], norm_layer=norm_layer if self.patch_norm else None)
        self.superpower = bool(superpower)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                embed_dim=embed_dims[i_layer],
                out_dim=embed_dims[i_layer + 1] if (i_layer < self.num_layers - 1) else None,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                init_value=init_values[i_layer],
                heads_range=heads_ranges[i_layer],
                ffn_dim=int(mlp_ratios[i_layer] * embed_dims[i_layer]),
                drop_path=dpr[sum(depths[:i_layer]): sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                split_or_not=(i_layer != 3),
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                layerscale=layerscales[i_layer],
                layer_init_values=layer_init_values,
            )
            self.layers.append(layer)

        # ★ 只有在 superpower=False（export-only）时才需要 stage 级 SAM
        self.encoder_sam_stages = nn.ModuleList()
        if not self.superpower:
            for i in range(self.num_layers):
                if i in self._sam_enc_enabled:
                    self.encoder_sam_stages.append(
                        SemanticAlignmentModule(
                            query_dim=embed_dims[i],
                            text_dim=text_dim,
                            use_topk=sam_use_topk,
                            top_m=sam_top_m,
                            num_heads=self.num_heads[i],
                            gamma_scale=sam_enc_gamma_scale,
                        )
                    )
                else:
                    self.encoder_sam_stages.append(_NoOpStageSAM())
        # superpower=True 时 forward 根本不会用到 encoder_sam_stages，因此无需构建

        # ★ 每两个 Block 放一个 SAM（偶数位），并强制包含最后一个 Block；其余用 NoOp
        self.encoder_sam_blocks = nn.ModuleList()
        for i in range(self.num_layers):
            depth_i = depths[i]
            if self.superpower and (i in self._sam_enc_enabled):
                # 选取集合：偶数位 {0,2,4,...}；并确保包含最后一层 depth_i-1
                keyset = set(range(0, depth_i, 2))
                if depth_i > 0:
                    keyset.add(depth_i - 1)

                stage_ml = nn.ModuleList()
                gamma_sched = torch.linspace(0.9, 0.5, steps=depth_i)  # 轻微递减，可保留你原来的也行
                for b in range(depth_i):
                    if b in keyset:
                        mod = SemanticAlignmentModule(
                            query_dim=embed_dims[i],
                            text_dim=text_dim,
                            use_topk=sam_use_topk,
                            top_m=sam_top_m,
                            num_heads=self.num_heads[i],
                            gamma_scale=sam_enc_gamma_scale,
                        )
                        with torch.no_grad():
                            mod.gamma.copy_(gamma_sched[b].to(mod.gamma))
                        stage_ml.append(mod)
                    else:
                        stage_ml.append(_NoOpSAM())
                self.encoder_sam_blocks.append(stage_ml)
            else:
                self.encoder_sam_blocks.append(nn.ModuleList([_NoOpSAM() for _ in range(depth_i)]))

        self.extra_norms = nn.ModuleList()
        for i in range(len(embed_dims) - 1):
            self.extra_norms.append(nn.LayerNorm(embed_dims[i + 1]))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            try:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            except:
                pass

    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            _state_dict = torch.load(pretrained)
            if "model" in _state_dict.keys():
                _state_dict = _state_dict["model"]
            if "state_dict" in _state_dict.keys():
                _state_dict = _state_dict["state_dict"]
            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                if k.startswith("backbone."):
                    state_dict[k[9:]] = v
                else:
                    state_dict[k] = v
            print("load " + pretrained)
            load_state_dict(self, state_dict, strict=False)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError("pretrained must be a str or None")

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def forward(self, x, x_e, text_features=None, export_geo_priors=False):
        # rgb
        x = self.patch_embed(x)
        # depth
        x_e = x_e[:, 0, :, :].unsqueeze(1)

        outs = []
        geo_priors = [] if export_geo_priors else None
        use_text_guidance = text_features is not None
        if use_text_guidance and text_features.dim() == 2:
            text_features = text_features.unsqueeze(0)
        if use_text_guidance:
            text_features = text_features.to(device=x.device, dtype=x.dtype)

        for i in range(self.num_layers):
            if export_geo_priors:
                with torch.no_grad():
                    H, W = x.shape[1], x.shape[2]

                    # 🔧 简化：只导出深度图，让 SAM 自己计算亲和度
                    depth_resized = F.interpolate(x_e, size=(H, W), mode='bilinear', align_corners=False)
                    geo_priors.append(depth_resized)  # [B, 1, H, W]

            if self.superpower:
                # 逐 block：仅当该 stage 启用时传入对应 ModuleList；否则传空
                sam_blocks = self.encoder_sam_blocks[i] if (
                            use_text_guidance and (i in self._sam_enc_enabled)) else None
                x_out, x = self.layers[i](x, x_e,
                                          text_features=text_features,
                                          sam_module=None,
                                          sam_blocks=sam_blocks,
                                          superpower=True)
            else:
                # === export-only：先跑完本 stage，再只对 x_out 做 SAM（不改 x / 不进主干） ===
                # 1) 进入 BasicLayer 时不传 sam_module，避免在 layer 内注入
                x_out, x = self.layers[i](
                    x, x_e,
                    text_features=None,
                    sam_module=None,
                    sam_blocks=None,
                    superpower=False,
                )

                # 2) 仅当启用并有文本时，对 x_out 做一次 SAM
                if use_text_guidance and (i in self._sam_enc_enabled):
                    sam_module = self.encoder_sam_stages[i]
                    x_out = sam_module(x_out, text_features)

            if i in self.out_indices:
                if i != 0:
                    norm_layer = self.extra_norms[i - 1]
                    x_out = norm_layer(x_out)
                out = x_out.permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        if export_geo_priors:
            return tuple(outs), geo_priors  # 🔧 返回 (features, geo_masks)
        return tuple(outs)

    def train(self, mode=True):
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()


def DFormerv2_S(pretrained=False, **kwargs):
    return dformerv2(
        embed_dims=[64, 128, 256, 512],
        depths=[3, 4, 18, 4],
        num_heads=[4, 4, 8, 16],
        heads_ranges=[4, 4, 6, 6],
        **kwargs,
    )


def DFormerv2_B(pretrained=False, **kwargs):
    return dformerv2(
        embed_dims=[80, 160, 320, 512],
        depths=[4, 8, 25, 8],
        num_heads=[5, 5, 10, 16],
        heads_ranges=[5, 5, 6, 6],
        layerscales=[False, False, True, True],
        layer_init_values=1e-6,
        **kwargs,
    )


def DFormerv2_L(pretrained=False, **kwargs):
    return dformerv2(
        embed_dims=[112, 224, 448, 640],
        depths=[4, 8, 25, 8],
        num_heads=[7, 7, 14, 20],
        heads_ranges=[6, 6, 6, 6],
        layerscales=[False, False, True, True],
        layer_init_values=1e-6,
        **kwargs,
    )
