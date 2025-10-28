import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticAlignmentModule(nn.Module):
    """
    多头版 SAM（与各 stage 的 num_heads 对齐；在 Cv=query_dim 空间做多头）
    - Decoder 侧 forward: 多头注意力 +（Top-p / Top-K）+ 置信门控 + 信任域 + FFN (Pre-LN)
    - Encoder/Superpower 侧 forward_ssa: 多头 SSA-lite（同样加入 Top-p / 置信门控 / 信任域；无FFN/无额外LN，单标量 gamma）
    """

    def __init__(
        self,
        query_dim: int,            # Cv
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
        gamma_scale: float = 1.0,  # SSA-lite 的额外缩放
        ssa_fusion_ratio: float = 1.5,
        ssa_fusion_drop: float = 0.0,

        # ★ 新增（默认即开启，无需在外部传参）：
        topp_p: float = 0.9,              # Top-p（0<p<1生效）
        token_keep_prob: float = 0.9,     # 训练期 Token Drop 的保留率（推理期无效）
        trust_region_tau: float = 0.5,    # 信任域强度系数（相对 x 的 RMS）
        use_conf_gate: bool = True,       # 置信门控开关
    ):
        super().__init__()
        self.top_m = top_m
        self.use_topk = use_topk
        self.add_residual = add_residual
        self.gate_channels = gate_channels
        self.clamp_logit = float(clamp_logit)

        # === 新增的稳健性开关/参数（默认开启） ===
        self.topp_p = float(topp_p)
        self.token_keep_prob = float(token_keep_prob)
        self.trust_region_tau = float(trust_region_tau)
        self.use_conf_gate = bool(use_conf_gate)

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

        # 多头聚合后再做一次线性投影
        self.out_proj = nn.Linear(query_dim, query_dim)

        # --- Null absorb token (for SSA-lite) ---
        self.enable_null = False
        self.null_text = nn.Parameter(torch.zeros(text_dim))  # [Ct]
        nn.init.normal_(self.null_text, mean=0.0, std=1e-3)

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
        self.gamma = nn.Parameter(torch.tensor(0.5, dtype=torch.float))        # encoder/superpower (SSA-lite)
        self.register_buffer("gamma_scale", torch.tensor(float(gamma_scale), dtype=torch.float))
        self.register_buffer("ssa_scale", torch.tensor(self.head_dim ** -0.5, dtype=torch.float))

        if ssa_fusion_ratio is not None and ssa_fusion_ratio > 0.0:
            mid = int(round(query_dim * float(ssa_fusion_ratio)))
            self.ssa_fusion = nn.Sequential(
                nn.Linear(query_dim, mid),
                nn.GELU(),
                nn.Dropout(ssa_fusion_drop),
                nn.Linear(mid, query_dim),
            )
        else:
            self.ssa_fusion = None

        # 初始化
        nn.init.xavier_uniform_(self.q_proj.weight);  nn.init.zeros_(self.q_proj.bias)
        nn.init.xavier_uniform_(self.k_proj.weight);  nn.init.zeros_(self.k_proj.bias)
        nn.init.xavier_uniform_(self.v_proj.weight);  nn.init.zeros_(self.v_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight); nn.init.zeros_(self.out_proj.bias)

    # ---------- 工具函数 ----------
    @staticmethod
    def _ensure_batched_text(text_features: torch.Tensor, B: int) -> torch.Tensor:
        """接受 (B,T,Ct) / (T,Ct) / (B,Ct) → 统一为 (B,T,Ct)"""
        if text_features.dim() == 3:
            return text_features
        if text_features.dim() == 2:  # (T,Ct) 共享 token
            return text_features.unsqueeze(0).expand(B, -1, -1)
        if text_features.dim() == 1:  # (Ct,)
            return text_features.view(1, 1, -1).expand(B, 1, -1)
        raise ValueError(f"Unsupported text tensor shape: {text_features.shape}")

    @staticmethod
    def _make_text_pad_mask(text_feats: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """将全零 token 视作 padding：返回 (B,T) 的 bool 掩码"""
        return (text_feats.float().abs().sum(dim=-1) <= eps)

    def _token_dropout(self, text_b: torch.Tensor, keep_prob: float, keep_last_if_null: bool):
        """训练期随机丢弃一部分文本 token，推理期不启用；保证至少保留一个 token。"""
        if (not self.training) or (keep_prob >= 1.0):
            return text_b
        B, T, Ct = text_b.shape
        keep = (torch.rand(B, T, 1, device=text_b.device) < keep_prob).float()
        # 保底：保留第一个 token；若末尾有 Null，则确保 Null 不丢
        keep[:, :1, :] = 1.0
        if keep_last_if_null:
            keep[:, -1:, :] = 1.0
        return text_b * keep

    @staticmethod
    def _nucleus_on_attn(attn: torch.Tensor, p: float):
        """对概率分布 attn（最后一维为 token）做 Top-p 稀疏化并重归一化。"""
        if (p is None) or (p <= 0.0) or (p >= 1.0):
            return attn
        prob, idx = torch.sort(attn, dim=-1, descending=True)
        csum = prob.cumsum(-1)
        keep = (csum <= p).to(attn.dtype)
        keep[..., 0:1] = 1.0  # 至少保留一个
        sparse = torch.zeros_like(attn).scatter(-1, idx, keep * prob)
        return sparse / sparse.sum(-1, keepdim=True).clamp_min(1e-6)

    @staticmethod
    def _conf_from_attn(attn: torch.Tensor):
        """
        给定注意力 attn：
        - forward:  attn.shape = [B, N, H, T]
        - forward_ssa: attn.shape = [B, H, N, T]
        返回像素级置信度 conf，形状均规约为 [B, N, 1]
        """
        if attn.dim() != 4:
            raise ValueError("attn must be 4D")
        if attn.shape[2] > 1:  # 有多头
            # 统一到 [B, H, N, T]
            if attn.shape[1] != attn.shape[2]:  # forward: [B,N,H,T]
                attn_hn = attn.permute(0, 2, 1, 3)  # [B,H,N,T]
            else:
                attn_hn = attn
        else:
            attn_hn = attn.permute(0, 2, 1, 3)  # 退化为单头也统一为 [B,H,N,T]

        maxp, _ = attn_hn.max(dim=-1, keepdim=True)  # [B,H,N,1]
        T_eff = (attn_hn > 0).float().sum(dim=-1, keepdim=True).clamp_min(1.0)
        denom = (T_eff.clamp_min(2.0)).log()
        entropy = -(attn_hn.clamp_min(1e-8) * attn_hn.clamp_min(1e-8).log()).sum(-1, keepdim=True) / denom

        conf_h = (maxp * (1.0 - entropy)).clamp(0, 1)  # [B,H,N,1]
        conf_bn = conf_h.mean(dim=1, keepdim=False)  # [B,N,1]
        return conf_bn

    @staticmethod
    def _trust_region(aligned: torch.Tensor, x: torch.Tensor, tau: float):
        """信任域：限制 aligned 的能量不超过 τ · rms(x)（逐位置）。"""
        if tau <= 0:
            return aligned
        x_rms = (x.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-6)     # [B,N,1]
        thr = tau * x_rms
        a_norm = (aligned.pow(2).sum(dim=-1, keepdim=True)).sqrt()       # [B,N,1]
        scale = (thr / a_norm.clamp_min(1e-6)).clamp(max=1.0)
        return aligned * scale

    # ---------- Decoder 侧：多头 + Top-p/Top-K + FFN ----------
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
        # 训练期 Token Drop（decoder forward 也启用）
        text_b = self._token_dropout(text_b, self.token_keep_prob, keep_last_if_null=False)
        k_full = self.k_proj(text_b)                           # (B,T,Cv)
        v_full = self.v_proj(text_b)                           # (B,T,Cv)
        pad_mask = self._make_text_pad_mask(text_b)            # (B,T)

        # 多头拆分
        Hh, Dh = self.num_heads, self.head_dim
        q = F.normalize(q_full, dim=-1, eps=1e-6).view(B, -1, Hh, Dh)    # (B,N,H,Dh)
        k = F.normalize(k_full, dim=-1, eps=1e-6).view(B, -1, Hh, Dh)    # (B,T,H,Dh)
        v = v_full.view(B, -1, Hh, Dh)                                    # (B,T,H,Dh)

        # 注意力（logits）
        scale = torch.clamp(self.logit_scale, min=-self.clamp_logit, max=self.clamp_logit).exp() / math.sqrt(self.d_k)
        sim = torch.einsum('bnhd,bthd->bnht', q, k) * scale               # (B,N,H,T)
        if pad_mask.any():
            sim = sim.masked_fill(pad_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        # 先按旧逻辑可选 Top-K（兼容原实现）
        if self.use_topk and (self.top_m is not None) and (self.top_m < sim.size(-1)):
            topv, topi = torch.topk(sim, k=self.top_m, dim=-1)            # (B,N,H,M)
            mask = torch.zeros_like(sim).scatter_(-1, topi, 1.0)
            sim = sim.masked_fill(mask.eq(0), float('-inf'))

        # softmax 后做 Top-p 稀疏化（默认启用）
        attn = F.softmax(sim, dim=-1)
        attn = self._nucleus_on_attn(attn, self.topp_p)
        attn = self.attn_drop(attn)                                       # (B,N,H,T)

        # 置信门控
        conf_bn1 = self._conf_from_attn(attn) if self.use_conf_gate else None

        # 聚合回 Cv（多头拼接）
        aligned_h = torch.einsum('bnht,bthd->bnhd', attn, v)              # (B,N,H,Dh)
        aligned = self.proj_drop(aligned_h.reshape(B, -1, Hh * Dh))       # (B,N,Cv)
        aligned = self.out_proj(aligned)

        # 信任域限制
        aligned = self._trust_region(aligned, x, self.trust_region_tau)

        # 残差 + 门控 + FFN（Pre-LN）
        gate = self.gate(x)                                               # (B,N,1) or (B,N,Cv)
        if conf_bn1 is not None:
            gate = gate * conf_bn1
        y = (x + self.alpha * gate * aligned) if self.add_residual else (self.alpha * aligned)
        y_norm = self.norm2(y)
        y = y + self.ffn(y_norm)
        return y.view(B, H, W, Cv)

    # ---------- Encoder/Superpower 侧：多头 SSA-lite ----------
    def forward_ssa(self, visual_features, text_features):
        """
        Encoder/Superpower: 轻量版多头注意力 + Top-p + 置信门控 + 信任域
        Args:
            visual_features: (B, H, W, Cv)
            text_features:   (B,T,Ct) / (T,Ct) / (B,Ct)
        """
        B, H, W, Cv = visual_features.shape
        x = visual_features.view(B, H * W, Cv)
        q_full = self.q_proj(x)

        # 统一成 (B,T,Ct)
        text_b = self._ensure_batched_text(text_features, B).to(visual_features.dtype)  # [B, T, Ct]
        pad_mask = self._make_text_pad_mask(text_b)  # [B, T], True=padding

        # 追加 Null（若启用），并在训练期做 Token Drop（保留 Null）
        if getattr(self, "enable_null", False):
            null_tok = self.null_text.to(text_b.dtype).view(1, 1, -1).expand(B, 1, -1)  # [B,1,Ct]
            text_b = torch.cat([text_b, null_tok], dim=1)  # [B,T+1,Ct]
            pad_mask = F.pad(pad_mask, (0, 1), value=False)
            text_b = self._token_dropout(text_b, self.token_keep_prob, keep_last_if_null=True)
        else:
            text_b = self._token_dropout(text_b, self.token_keep_prob, keep_last_if_null=False)

        k_full = self.k_proj(text_b)  # [B, T_active, Cv]
        v_full = self.v_proj(text_b)  # [B, T_active, Cv]

        Hh, Dh = self.num_heads, self.head_dim
        q = q_full.view(B, -1, Hh, Dh).permute(0, 2, 1, 3)  # [B, Hh, N, Dh]
        k = k_full.view(B, -1, Hh, Dh).permute(0, 2, 1, 3)  # [B, Hh, T, Dh]
        v = v_full.view(B, -1, Hh, Dh).permute(0, 2, 1, 3)  # [B, Hh, T, Dh]

        # Null 的 value 置零（不贡献特征）
        if getattr(self, "enable_null", False):
            v[:, :, -1, :] = 0.0

        all_pad = pad_mask.all(dim=1)  # [B]
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.ssa_scale  # [B, Hh, N, T]
        if pad_mask.any():
            attn_logits = attn_logits.masked_fill(
                pad_mask.unsqueeze(1).unsqueeze(2),  # [B,1,1,T]
                float('-inf')
            )

        # softmax → Top-p 稀疏化（默认启用）
        attn = torch.softmax(attn_logits, dim=-1)  # [B,Hh,N,T]
        if all_pad.any():
            attn[all_pad] = 0
        attn = self._nucleus_on_attn(attn, self.topp_p)

        # 置信门控
        conf_bn1 = None
        if self.use_conf_gate:
            # attn: [B, Hh, N, T]
            maxp, _ = attn.max(dim=-1, keepdim=True)  # [B,Hh,N,1]
            # 用“有效 token 数”做熵归一化的分母；当只剩1个token时，取 log(2) 防止除零
            T_eff = (attn > 0).float().sum(dim=-1, keepdim=True).clamp_min(2.0)
            entropy = -(attn.clamp_min(1e-8) * attn.clamp_min(1e-8).log()).sum(-1, keepdim=True) / T_eff.log()
            conf_h = (maxp * (1.0 - entropy)).clamp(0, 1)  # [B,Hh,N,1]
            conf_bn1 = conf_h.mean(dim=1, keepdim=False)  # [B,N,1]

        aligned = torch.matmul(attn, v)  # [B, Hh, N, Dh]
        aligned = aligned.permute(0, 2, 1, 3).reshape(B, -1, Hh * Dh)  # [B, N, Cv]
        aligned = self.out_proj(aligned)

        if self.ssa_fusion is not None:
            aligned = self.ssa_fusion(aligned)

        if all_pad.any():
            aligned = aligned * (~all_pad).view(B, 1, 1).float()

        # 信任域限制
        aligned = self._trust_region(aligned, x, self.trust_region_tau)

        # 残差连接（乘上置信门控）
        if conf_bn1 is not None:
            aligned = conf_bn1 * aligned
        y = x + (self.gamma * self.gamma_scale) * aligned
        y = y.view(B, H, W, Cv)
        return y
