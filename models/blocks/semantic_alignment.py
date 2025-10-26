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

        # 多头聚合后再做一次线性投影，保持与旧 SSA 逻辑一致
        self.out_proj = nn.Linear(query_dim, query_dim)

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
        self.register_buffer("ssa_scale", torch.tensor(self.head_dim ** -0.5, dtype=torch.float))

        self.ssr_top_m_cap = 8  # 每图最多保留的标签数（剪枝上限）
        self.pixel_topk = 2  # 每像素在标签维上的Top-k（1或2）
        self.keep_mass = 0.90  # 动态保留：覆盖到90%“token分布质量”
        self.enable_null = True  # 是否拼接一个可学习的null文本token
        self.gamma_entropy = True  # 是否用熵调节注入强度
        self.null_text = nn.Parameter(torch.zeros(1, 1, self.k_proj.in_features))

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
        aligned = self.out_proj(aligned)                                  # 线性映射（与 SSA 对齐）

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

        # 统一文本维度，并（可选）拼接null token
        text_b = self._ensure_batched_text(text_features, B).to(visual_features.dtype)  # [B,T,Ct]
        if self.enable_null:
            null = self.null_text.to(text_b.dtype).expand(B, -1, -1)  # [B,1,Ct]
            text_b = torch.cat([text_b, null], dim=1)  # [B,T+1,Ct]
        pad_mask = self._make_text_pad_mask(text_b)  # [B,T(+1)]

        # 投影到Cv并拆头
        k_full = self.k_proj(text_b);
        v_full = self.v_proj(text_b)  # [B,T',Cv]
        Hh, Dh = self.num_heads, self.head_dim
        q = q_full.view(B, -1, Hh, Dh).permute(0, 2, 1, 3)  # [B,Hh,N,Dh]
        k = k_full.view(B, -1, Hh, Dh).permute(0, 2, 1, 3)  # [B,Hh,T',Dh]
        v = v_full.view(B, -1, Hh, Dh).permute(0, 2, 1, 3)  # [B,Hh,T',Dh]

        # 1) 自评分（基于第一趟 logits，不softmax）
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.ssa_scale  # [B,Hh,N,T']
        if pad_mask.any():
            attn_logits = attn_logits.masked_fill(pad_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        token_score = attn_logits.amax(dim=2).mean(dim=1)  # [B,T']  max_N then mean_heads
        # 动态“保质量”Top-M：以softmax(token_score)为质量分布，累计到keep_mass为止，且<=cap
        prob = torch.softmax(token_score, dim=-1)
        sorted_prob, sorted_idx = prob.sort(dim=-1, descending=True)
        cum = torch.cumsum(sorted_prob, dim=-1)
        # 选到第一个使 cum>=keep_mass 的位置，再与上限cap取min
        mass_M = (cum >= self.keep_mass).float().argmax(dim=-1) + 1  # [B]
        M_cap = min(self.ssr_top_m_cap, prob.size(-1))
        M_sel = torch.clamp(mass_M, min=1, max=M_cap)  # [B]
        # 逐batch拿各自Top-M索引（按概率排序），然后在T'维gather
        gather_idx = []
        for b in range(B):
            m = int(M_sel[b].item())
            gather_idx.append(sorted_idx[b, :m])
        maxM = max(int(m.item()) for m in M_sel)
        # pad到相同长度便于堆叠
        gather_idx = [torch.cat([idx, idx.new_full((maxM - idx.numel(),), idx[0].item())]) for idx in gather_idx]
        gather_idx = torch.stack(gather_idx, dim=0)  # [B,maxM]

        # 重新索引K/V（保留Top-M），并重算logits
        k2 = k.gather(dim=2, index=gather_idx.view(B, 1, maxM, 1).expand(B, Hh, maxM, Dh))
        v2 = v.gather(dim=2, index=gather_idx.view(B, 1, maxM, 1).expand(B, Hh, maxM, Dh))

        logits2 = torch.matmul(q, k2.transpose(-2, -1)) * self.ssa_scale  # [B,Hh,N,maxM]

        # 2) 像素级Top-k（在标签维稀疏）
        if (self.pixel_topk is not None) and (self.pixel_topk >= 1) and (self.pixel_topk < maxM):
            topv, topi = torch.topk(logits2, k=self.pixel_topk, dim=-1)  # [B,Hh,N,k]
            mask = torch.zeros_like(logits2).scatter_(-1, topi, 1.0)
            logits2 = logits2.masked_fill(mask.eq(0), float('-inf'))

        attn = torch.softmax(logits2, dim=-1)  # [B,Hh,N,maxM]

        # --- 熵感知 γ（不确定时少注入）---
        if self.gamma_entropy:
            eps = 1e-6
            p = torch.clamp(attn, min=eps)  # [B,Hh,N,M]
            entropy = -(p * torch.log(p)).sum(dim=-1) / math.log(attn.size(-1) + 1e-12)  # [B,Hh,N]
            concentration = 1.0 - entropy  # [B,Hh,N]
            c = concentration.mean(dim=1, keepdim=True)  # [B,1,N]
        else:
            c = None

        # 聚合
        aligned = torch.matmul(attn, v2)  # [B,Hh,N,Dh] 或 v
        aligned = aligned.permute(0, 2, 1, 3).reshape(B, -1, Hh * Dh)  # [B,N,Cv]
        aligned = self.out_proj(aligned)

        if c is not None:
            aligned = aligned * c.permute(0, 2, 1)  # [B,N,1] → 广播到 Cv

        y = x + (self.gamma * self.gamma_scale) * aligned

        _B, _H, _W, _Cv = visual_features.shape
        return y.view(_B, _H, _W, _Cv)