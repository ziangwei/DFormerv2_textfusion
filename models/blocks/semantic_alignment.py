import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticAlignmentModule(nn.Module):
    """
    多头版 SAM（与各 stage 的 num_heads 对齐；在 Cv=query_dim 空间做多头）
    - Decoder 侧 forward: 多头注意力 (+可选Top-K) + 残差门控(alpha) + FFN (Pre-LN)
    - Encoder/Superpower 侧 forward_ssa: 多头 SSA-lite（无Top-K/无FFN/无额外LN，单标量 gamma）
    """

    def __init__(
        self,
        query_dim: int,            # Cv（当前 stage 的通道数）
        text_dim: int,             # Ct（通常 512）
        top_m: int = 5,
        use_topk: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        ffn_drop: float = 0.0,
        add_residual: bool = True,
        gate_channels: bool = False,
        alpha_init: float = 0.1,
        clamp_logit: float = 2.0,
        num_heads: int = 1,        # 由调用方传：该 stage 的 num_heads
        gamma_scale: float = 1.0,  # SSA-lite 的额外缩放，默认为 1
    ):
        super().__init__()
        self.top_m = top_m
        self.use_topk = use_topk
        self.add_residual = add_residual
        self.gate_channels = gate_channels
        self.clamp_logit = float(clamp_logit)

        # === 多头配置：在 Cv 空间按头数切分 ===
        self.num_heads = int(num_heads)
        assert query_dim % self.num_heads == 0, \
            f"query_dim({query_dim}) must be divisible by num_heads({self.num_heads})"
        self.head_dim = query_dim // self.num_heads  # Dh
        self.d_k = float(self.head_dim)              # 缩放用

        # === 预归一化（decoder 侧 forward 用；SSA-lite 不用） ===
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        # === 线性投影：统一映射到 Cv 空间做注意力 ===
        self.q_proj  = nn.Linear(query_dim, query_dim)   # (B,N,Cv)->(B,N,Cv)
        self.k_proj  = nn.Linear(text_dim,  query_dim)   # (B,T,Ct)->(B,T,Cv)
        self.v_proj  = nn.Linear(text_dim,  query_dim)   # (B,T,Ct)->(B,T,Cv)

        # 多头拼接后已是 Cv，按需再做投影；为保持接口一致提供 out_proj（可为 Identity）
        self.out_proj = nn.Identity()

        # === 门控（decoder 侧使用；encoder 侧用 gamma） ===
        if gate_channels:
            self.gate = nn.Sequential(nn.Linear(query_dim, query_dim), nn.Sigmoid())
        else:
            self.gate = nn.Sequential(nn.Linear(query_dim, 1), nn.Sigmoid())

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # 温度参数（对数域）
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

        # FFN（decoder 侧 forward 使用；SSA-lite 不用）
        self.ffn = nn.Sequential(
            nn.Linear(query_dim, query_dim * 4),
            nn.GELU(),
            nn.Dropout(ffn_drop),
            nn.Linear(query_dim * 4, query_dim),
            nn.Dropout(ffn_drop),
        )

        # 残差缩放参数
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float))  # decoder
        self.gamma = nn.Parameter(torch.tensor(0.5, dtype=torch.float))       # encoder/superpower (SSA-lite)
        self.register_buffer("gamma_scale", torch.tensor(float(gamma_scale), dtype=torch.float))

        # 初始化
        nn.init.xavier_uniform_(self.q_proj.weight);  nn.init.zeros_(self.q_proj.bias)
        nn.init.xavier_uniform_(self.k_proj.weight);  nn.init.zeros_(self.k_proj.bias)
        nn.init.xavier_uniform_(self.v_proj.weight);  nn.init.zeros_(self.v_proj.bias)

    # ---------- 工具函数 ----------
    @staticmethod
    def _ensure_batched_text(text_features: torch.Tensor, B: int) -> torch.Tensor:
        """
        接受 (B,T,Ct) / (T,Ct) / (B,Ct) → 统一为 (B,T,Ct)
        """
        if text_features.dim() == 3:
            return text_features
        if text_features.dim() == 2:  # (T,Ct) 共享 token
            return text_features.unsqueeze(0).expand(B, -1, -1)
        if text_features.dim() == 1:  # (Ct,)
            return text_features.view(1, 1, -1).expand(B, 1, -1)
        raise ValueError(f"Unsupported text tensor shape: {text_features.shape}")

    @staticmethod
    def _make_text_pad_mask(text_feats: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        将全零 token 视作 padding：返回 (B,T) 的 bool 掩码
        """
        return (text_feats.float().abs().sum(dim=-1) <= eps)

    # ---------- Decoder 侧：多头 +（可选）TopK + FFN ----------
    def forward(self, visual_features: torch.Tensor, text_features: torch.Tensor):
        """
        visual_features: (B,H,W,Cv)  —— NHWC
        text_features:   (B,T,Ct) / (T,Ct) / (B,Ct)
        """
        B, H, W, Cv = visual_features.shape

        # Pre-LN + Q
        x = self.norm1(visual_features).view(B, H * W, Cv)     # (B,N,Cv)
        q_full = self.q_proj(x)                                # (B,N,Cv)

        # 文本 → K/V（到 Cv 空间）
        text_b = self._ensure_batched_text(text_features, B)   # (B,T,Ct)
        k_full = self.k_proj(text_b)                           # (B,T,Cv)
        v_full = self.v_proj(text_b)                           # (B,T,Cv)
        pad_mask = self._make_text_pad_mask(text_b)            # (B,T)

        # 多头拆分
        Hh, Dh = self.num_heads, self.head_dim
        q = F.normalize(q_full, dim=-1, eps=1e-6).view(B, -1, Hh, Dh)  # (B,N,H,Dh)
        k = F.normalize(k_full, dim=-1, eps=1e-6).view(B, -1, Hh, Dh)  # (B,T,H,Dh)
        v = v_full.view(B, -1, Hh, Dh)                                  # (B,T,H,Dh)

        # 注意力
        scale = torch.clamp(self.logit_scale, min=-self.clamp_logit, max=self.clamp_logit).exp() / math.sqrt(self.d_k)
        sim = torch.einsum('bnhd,bthd->bnht', q, k) * scale             # (B,N,H,T)
        if pad_mask.any():
            sim = sim.masked_fill(pad_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        # Top-K（对 T 维；行为与原实现一致）
        if self.use_topk and (self.top_m is not None) and (self.top_m < sim.size(-1)):
            topv, topi = torch.topk(sim, k=self.top_m, dim=-1)          # (B,N,H,M)
            mask = torch.zeros_like(sim).scatter_(-1, topi, 1.0)
            sim = sim.masked_fill(mask.eq(0), float('-inf'))

        attn = F.softmax(sim, dim=-1)
        attn = self.attn_drop(attn)

        # 聚合回 Cv（多头拼接）
        aligned_h = torch.einsum('bnht,bthd->bnhd', attn, v)             # (B,N,H,Dh)
        aligned = self.proj_drop(aligned_h.reshape(B, -1, Hh * Dh))      # (B,N,Cv)
        aligned = self.out_proj(aligned)                                  # Identity

        # 残差 + 门控 + FFN（Pre-LN）
        gate = self.gate(x)                                              # (B,N,1) 或 (B,N,Cv)
        y = (x + self.alpha * gate * aligned) if self.add_residual else (self.alpha * aligned)
        y = self.norm2(y)
        y = y + self.ffn(y)
        return y.view(B, H, W, Cv)

    # ---------- Encoder/Superpower 侧：多头 SSA-lite（无Top-K/无FFN/无额外LN） ----------
    def forward_ssa(self, visual_features, text_features):
        B, H, W, Cv = visual_features.shape
        x = visual_features.view(B, H * W, Cv)
        q_full = self.q_proj(x)

        # (1) 统一成 (B,T,Ct)
        text_b = self._ensure_batched_text(text_features, B)  # [B, T, Ct]
        pad_mask = self._make_text_pad_mask(text_b)  # [B, T], True=padding

        # (2) 若存在全 pad 的样本，直接旁路
        if bool(pad_mask.all(dim=1).any()):
            return visual_features

        # (3) 动态裁短到本 batch 的最大有效长度
        valid_len = (~pad_mask).sum(dim=1)  # [B]
        T_active = int(valid_len.max().item())
        if T_active < text_b.size(1):
            text_b = text_b[:, :T_active, :]
            pad_mask = pad_mask[:, :T_active]

        # (4) 线性投影（此时 T 仅为 T_active）
        k_full = self.k_proj(text_b)  # [B, T_active, Cv]
        v_full = self.v_proj(text_b)  # [B, T_active, Cv]

        Hh, Dh = self.num_heads, self.head_dim
        q = q_full.view(B, -1, Hh, Dh)
        k = k_full.view(B, -1, Hh, Dh)
        v = v_full.view(B, -1, Hh, Dh)

        # (5) 标准缩放 + mask 再 softmax
        sim = torch.einsum('bnhd,bthd->bnht', q, k) / math.sqrt(self.d_k)
        sim = sim.masked_fill(pad_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn = torch.softmax(sim, dim=-1)
        aligned_h = torch.einsum('bnht,bthd->bnhd', attn, v)
        aligned = aligned_h.reshape(B, -1, Hh * Dh)

        y = x + (self.gamma * self.gamma_scale) * aligned
        return y.view(B, H, W, Cv)