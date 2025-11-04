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
        decoder_use_cosine: bool = True,
        decoder_learnable_temp: bool = True,
        decoder_logit_scale_init: float = 1 / 0.07,
        encoder_use_cosine: bool = False,
        encoder_learnable_temp: bool = False,
        encoder_logit_scale_init: float = 1.0,
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
        self.register_buffer("_inv_sqrt_dk", torch.tensor(1.0 / math.sqrt(self.d_k), dtype=torch.float))

        # === 预归一化（decoder 侧 forward 用；SSA-lite 不用） ===
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        # === 线性投影：统一映射到 Cv 空间做注意力 ===
        self.q_proj  = nn.Linear(query_dim, query_dim)   # (B,N,Cv)->(B,N,Cv)
        self.k_proj  = nn.Linear(text_dim,  query_dim)   # (B,T,Ct)->(B,T,Cv)
        self.v_proj  = nn.Linear(text_dim,  query_dim)   # (B,T,Ct)->(B,T,Cv)

        # 多头聚合后再做一次线性投影，保持与旧 SSA 逻辑一致
        self.out_proj = nn.Linear(query_dim, query_dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # 温度与注意力样式配置
        self.decoder_use_cosine = bool(decoder_use_cosine)
        self.decoder_learnable_temp = bool(decoder_learnable_temp)
        decoder_init = math.log(max(decoder_logit_scale_init, 1e-6))
        self.decoder_logit_scale = nn.Parameter(
            torch.tensor(decoder_init, dtype=torch.float),
            requires_grad=self.decoder_learnable_temp,
        )

        self.encoder_use_cosine = bool(encoder_use_cosine)
        self.encoder_learnable_temp = bool(encoder_learnable_temp)
        encoder_init = math.log(max(encoder_logit_scale_init, 1e-6))
        self.encoder_logit_scale = nn.Parameter(
            torch.tensor(encoder_init, dtype=torch.float),
            requires_grad=self.encoder_learnable_temp,
        )

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

        self.save_attention = False  # 新增：控制是否保存attention
        self.last_attention_map = None  # 新增：保存最后一次的attention
        self.last_spatial_shape = None

        # 初始化
        nn.init.xavier_uniform_(self.q_proj.weight);  nn.init.zeros_(self.q_proj.bias)
        nn.init.xavier_uniform_(self.k_proj.weight);  nn.init.zeros_(self.k_proj.bias)
        nn.init.xavier_uniform_(self.v_proj.weight);  nn.init.zeros_(self.v_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight); nn.init.zeros_(self.out_proj.bias)

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

    def _decoder_scale(self) -> torch.Tensor:
        scale_log = self.decoder_logit_scale
        if self.decoder_learnable_temp:
            scale_log = torch.clamp(scale_log, min=-self.clamp_logit, max=self.clamp_logit)
        return torch.exp(scale_log) * self._inv_sqrt_dk

    def _encoder_scale(self) -> torch.Tensor:
        scale_log = self.encoder_logit_scale
        if self.encoder_learnable_temp:
            scale_log = torch.clamp(scale_log, min=-self.clamp_logit, max=self.clamp_logit)
        return torch.exp(scale_log) * self._inv_sqrt_dk

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


        # ===== [FIX] 检查是否所有文本都是padding，避免 softmax(-inf) 产生 NaN =====
        # 统计每个样本的有效token数量
        valid_len = (~pad_mask).sum(dim=1)  # (B,)
        T_active = int(valid_len.max().item()) if valid_len.numel() > 0 else 0

        # 如果整个batch都没有有效文本，直接跳过SAM，返回原始特征
        if T_active == 0:
            # 保持原始结构：仍然过FFN以保持数据流一致性
            y = self.norm2(x)
            y = y + self.ffn(y)
            return y.view(B, H, W, Cv)

        # 检查每个样本是否全为padding
        all_pad_per_sample = pad_mask.all(dim=1)  # (B,) bool tensor

        # 多头拆分
        Hh, Dh = self.num_heads, self.head_dim
        q = q_full.view(B, -1, Hh, Dh)
        k = k_full.view(B, -1, Hh, Dh)
        if self.decoder_use_cosine:
            q = F.normalize(q, dim=-1, eps=1e-6)
            k = F.normalize(k, dim=-1, eps=1e-6)
        v = v_full.view(B, -1, Hh, Dh)                                  # (B,T,H,Dh)

        # 注意力
        scale = self._decoder_scale()
        sim = torch.einsum('bnhd,bthd->bnht', q, k) * scale             # (B,N,H,T)
        if pad_mask.any():
            sim = sim.masked_fill(pad_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        # Top-K（对 T 维；行为与原实现一致）
        if self.use_topk and (self.top_m is not None) and (self.top_m < sim.size(-1)):
            topv, topi = torch.topk(sim, k=self.top_m, dim=-1)          # (B,N,H,M)
            mask = torch.zeros_like(sim).scatter_(-1, topi, 1.0)
            sim = sim.masked_fill(mask.eq(0), float('-inf'))

        attn = F.softmax(sim, dim=-1)
        # ===== [FIX] 将全padding样本的attention置为0，避免NaN传播 =====
        if all_pad_per_sample.any():
            # 对于全padding的样本，将其attention全部置为0
            # attn shape: (B,N,H,T)
            attn = attn.masked_fill(
                all_pad_per_sample.view(B, 1, 1, 1).expand(-1, attn.size(1), attn.size(2), -1),
                0.0
            )
        attn = self.attn_drop(attn)

        if self.save_attention:
            self.last_attention_map = attn.detach()  # (B,N,H,T)
            self.last_spatial_shape = (H, W)
        # 聚合回 Cv（多头拼接）
        aligned_h = torch.einsum('bnht,bthd->bnhd', attn, v)             # (B,N,H,Dh)
        aligned = self.proj_drop(aligned_h.reshape(B, -1, Hh * Dh))      # (B,N,Cv)
        aligned = self.out_proj(aligned)                                  # 线性映射（与 SSA 对齐）

        # 残差 + 门控 + FFN（Pre-LN）                                          # (B,N,1) 或 (B,N,Cv)
        y = (x + self.alpha * aligned) if self.add_residual else (self.alpha * aligned)
        y = self.norm2(y)
        y = y + self.ffn(y)
        return y.view(B, H, W, Cv)

    # ---------- Encoder/Superpower 侧：多头 SSA-lite（无Top-K/无FFN/无额外LN） ----------
    def forward_ssa(self, visual_features, text_features):
        """Encoder/Superpower: 轻量版多头注意力

        Args:
            visual_features: (B, H, W, Cv) 的视觉特征
            text_features:   支持 (B,T,Ct) / (T,Ct) / (B,Ct)
        """
        B, H, W, Cv = visual_features.shape
        x = visual_features.view(B, H * W, Cv)
        q_full = self.q_proj(x)

        # (1) 统一成 (B,T,Ct)
        text_b = self._ensure_batched_text(text_features, B).to(visual_features.dtype)  # [B, T, Ct]
        pad_mask = self._make_text_pad_mask(text_b)  # [B, T], True=padding

        # (2) 动态裁短到本 batch 的最大有效长度（仅统计非 padding token）
        valid_len = (~pad_mask).sum(dim=1)  # [B]
        T_active = int(valid_len.max().item()) if valid_len.numel() > 0 else 0

        # 若整个 batch 都没有有效文本，引导信息为空，直接旁路
        if T_active == 0:
            return visual_features

        if T_active < text_b.size(1):
            text_b = text_b[:, :T_active, :]
            pad_mask = pad_mask[:, :T_active]

        # (4) 线性投影（此时 T 仅为 T_active）
        k_full = self.k_proj(text_b)  # [B, T_active, Cv]
        v_full = self.v_proj(text_b)  # [B, T_active, Cv]

        Hh, Dh = self.num_heads, self.head_dim
        q = q_full.view(B, -1, Hh, Dh)
        k = k_full.view(B, -1, Hh, Dh)
        if self.encoder_use_cosine:
            q = F.normalize(q, dim=-1, eps=1e-6)
            k = F.normalize(k, dim=-1, eps=1e-6)
        q = q.permute(0, 2, 1, 3)  # [B, Hh, N, Dh]
        k = k.permute(0, 2, 1, 3)  # [B, Hh, T, Dh]
        v = v_full.view(B, -1, Hh, Dh).permute(0, 2, 1, 3)  # [B, Hh, T, Dh]

        all_pad = pad_mask.all(dim=1)  # [B]
        active_idx = (~all_pad).nonzero(as_tuple=False).squeeze(1)

        # 先构造输出张量，默认直接拷贝输入（对应没有文本的样本）
        y = x.clone()

        if active_idx.numel() > 0:
            q_act = q.index_select(0, active_idx)
            k_act = k.index_select(0, active_idx)
            v_act = v.index_select(0, active_idx)
            pad_act = pad_mask.index_select(0, active_idx)

            # (5) 标准缩放 + mask 再 softmax（与旧 SSA 逻辑保持一致）
            attn_logits = torch.matmul(q_act, k_act.transpose(-2, -1)) * self._encoder_scale()
            attn_logits = attn_logits.masked_fill(pad_act.unsqueeze(1).unsqueeze(2), float('-inf'))
            attn = torch.softmax(attn_logits, dim=-1)

            if self.save_attention:
                # attn: [B_active, Hh, N, T]
                # 需要恢复到完整 batch 大小
                full_attn = torch.zeros(B, Hh, q.size(2), attn.size(-1),
                                        device=attn.device, dtype=attn.dtype)
                full_attn.index_copy_(0, active_idx, attn)

                # 转换为 (B, N, H, T) 格式，与 decoder 保持一致
                self.last_attention_map = full_attn.permute(0, 2, 1, 3).detach()
                self.last_spatial_shape = (H, W)

            aligned = torch.matmul(attn, v_act)  # [B_active, Hh, N, Dh]
            aligned = aligned.permute(0, 2, 1, 3).reshape(active_idx.numel(), -1, Hh * Dh)
            aligned = self.out_proj(aligned)

            y_active = x.index_select(0, active_idx) + (self.gamma * self.gamma_scale) * aligned
            y.index_copy_(0, active_idx, y_active)

        y = y.view(B, H, W, Cv)
        return y