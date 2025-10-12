import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticAlignmentModule(nn.Module):
    """
    文本引导的语义对齐模块（SAM v2.1）
    - 支持可选 Top-K（像素级），默认对 imglabels 可关闭
    - 对 padding 的全零 token 做 mask
    - q/k/text 全部 L2 归一化
    - 残差缩放 alpha（可学习，初始 0.1），温度 logit_scale clamp
    """

    def __init__(
        self,
        query_dim: int,          # C_visual
        text_dim: int,           # C_text
        top_m: int = 5,
        use_topk: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        ffn_drop: float = 0.0,
        add_residual: bool = True,
        gate_channels: bool = False,  # False: scalar gate; True: channel gate
        alpha_init: float = 0.1,      # 残差缩放初值
        clamp_logit: float = 2.0      # 温度上下界的绝对值：exp(clamp(...))
    ):
        super().__init__()
        self.top_m = top_m
        self.use_topk = use_topk
        self.add_residual = add_residual
        self.gate_channels = gate_channels
        self.clamp_logit = float(clamp_logit)

        # 环境变量可一键关闭 Top-K（对所有端口），不设置则不影响
        self._env_use_topk = (os.environ.get("SAM_USE_TOPK", "1") != "0")

        # 预归一化
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        # 线性投影
        self.q_proj = nn.Linear(query_dim, text_dim)
        self.v_proj = nn.Linear(text_dim, query_dim)

        # 门控
        if gate_channels:
            self.gate = nn.Sequential(
                nn.Linear(query_dim, query_dim),
                nn.Sigmoid()
            )
        else:
            self.gate = nn.Sequential(
                nn.Linear(query_dim, 1),
                nn.Sigmoid()
            )

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # 温度（对数域）与维度
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
        self.d_k = float(text_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(query_dim, query_dim * 4),
            nn.GELU(),
            nn.Dropout(ffn_drop),
            nn.Linear(query_dim * 4, query_dim),
            nn.Dropout(ffn_drop),
        )

        # 残差缩放
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float))

        # 初始化
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
        if self.v_proj.bias is not None:
            nn.init.zeros_(self.v_proj.bias)

    @staticmethod
    def _make_text_pad_mask(text_feats: torch.Tensor, eps: float = 1e-6):
        """返回 (B,T) 的bool掩码：True 表示该 token 是 padding（整行≈0）。"""
        with torch.no_grad():
            norm = text_feats.float().abs().sum(dim=-1)  # (B,T)
            mask = norm <= eps
        return mask

    @staticmethod
    def _ensure_batched_text(text_features: torch.Tensor, B: int):
        if text_features.dim() == 2:  # (T,Ct)
            text_features = text_features.unsqueeze(0)  # (1,T,Ct)
        if text_features.size(0) != B:
            text_features = text_features.expand(B, -1, -1).contiguous()
        return text_features

    def forward(self, visual_features: torch.Tensor, text_features: torch.Tensor):
        """
        visual_features: (B,H,W,Cv) NHWC
        text_features:   (B,T,Ct) 或 (T,Ct)
        """
        B, H, W, Cv = visual_features.shape

        # 预归一化 + q
        x = self.norm1(visual_features).view(B, H * W, Cv)  # (B,N,Cv)
        q = self.q_proj(x)                                  # (B,N,Ct)
        q = F.normalize(q, dim=-1, eps=1e-6)

        # 文本处理（广播 + 归一化 + padding mask）
        text_features = self._ensure_batched_text(text_features, B)    # (B,T,Ct)
        k = F.normalize(text_features, dim=-1, eps=1e-6)               # (B,T,Ct)
        v = self.v_proj(text_features)                                  # (B,T,Cv)
        pad_mask = self._make_text_pad_mask(text_features)              # (B,T)

        # 相似度 logits
        scale = torch.clamp(self.logit_scale, min=-self.clamp_logit, max=self.clamp_logit).exp() / math.sqrt(self.d_k)
        sim = torch.einsum('bnc,btc->bnt', q, k) * scale  # (B,N,T)

        # 是否执行 Top-K（像素级）
        effective_topk = (self.use_topk and self._env_use_topk and
                          (self.top_m is not None))
        if effective_topk and (self.top_m < sim.size(-1)):
            # 取每个像素的 Top-K 文本
            top_vals, top_idx = sim.topk(self.top_m, dim=-1)  # (B,N,M)
            if pad_mask.any():
                pad_mask_top = torch.gather(pad_mask.unsqueeze(1).expand(B, H * W, -1), 2, top_idx)
                top_vals = top_vals.masked_fill(pad_mask_top, float("-inf"))
            attn = F.softmax(top_vals, dim=-1)
            attn = self.attn_drop(attn)  # (B,N,M)

            # 从 v 里 gather 出对应 (B,N,M,Cv)
            v_exp = v.unsqueeze(1).expand(B, H * W, v.size(1), v.size(2))
            v_sel = torch.gather(v_exp, 2, top_idx.unsqueeze(-1).expand(B, H * W, self.top_m, v.size(2)))
            aligned = (attn.unsqueeze(-1) * v_sel).sum(dim=2)  # (B,N,Cv)
        else:
            # 全量 softmax，但先屏蔽 padding token
            if pad_mask.any():
                sim = sim.masked_fill(pad_mask.unsqueeze(1), float("-inf"))
            attn = F.softmax(sim, dim=-1)             # (B,N,T)
            attn = self.attn_drop(attn)
            aligned = torch.einsum('bnt,btc->bnc', attn, v)  # (B,N,Cv)

        aligned = self.proj_drop(aligned)

        # 门控 + 残差
        gate = self.gate(x)  # (B,N,1) 或 (B,N,Cv)
        aligned = aligned * gate

        y = x + self.alpha * aligned if self.add_residual else aligned
        y = self.norm2(y)
        y = y + self.ffn(y)  # Pre-LN 结构

        return y.view(B, H, W, Cv)
