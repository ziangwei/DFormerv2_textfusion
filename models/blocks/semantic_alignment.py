# semantic_alignment.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticAlignmentModule(nn.Module):
    """
    多头语义对齐（SAM）
    - Decoder: forward（Pre-LN + MHA + FFN）
      稀疏注意力：Top-K（默认 K=3）→ Softmax → Dropout → 归一化
    - Encoder: forward_ssa（轻量注意力 + 残差 γ）
      稀疏注意力：Top-K（默认 K=3）→ Softmax → Dropout → 归一化
    - 置信旁路：conf = max_prob × (1 − norm-entropy)
        若 conf < conf_thresh，则该像素位置“不注入增益”（旁路）。
    """

    def __init__(
        self,
        query_dim: int,
        text_dim: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        ffn_drop: float = 0.0,
        add_residual: bool = True,
        gate_channels: bool = False,
        alpha_init: float = 0.1,
        clamp_logit: float = 2.0,
        num_heads: int = 1,
        gamma_scale: float = 1.0,
        # —— 保持对接：由外部传入 sam_use_topk/sam_top_m 后在调用处映射到下两项 ——
        use_topk: bool = True,
        top_m: int = 3,
        # 内部固定阈值，避免新增外部参数名
        conf_thresh: float = 0.35,
    ):
        super().__init__()
        self.add_residual = bool(add_residual)
        self.gate_channels = bool(gate_channels)
        self.clamp_logit = float(clamp_logit)
        self.use_topk = bool(use_topk)
        self.top_m = int(top_m) if top_m is not None else None
        self.conf_thresh = float(conf_thresh)

        # 多头
        self.num_heads = int(num_heads)
        assert query_dim % self.num_heads == 0
        self.head_dim = query_dim // self.num_heads
        self.d_k = float(self.head_dim)

        # 预归一化（decoder）
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        # 投影
        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(text_dim,  query_dim)
        self.v_proj = nn.Linear(text_dim,  query_dim)
        self.out_proj = nn.Linear(query_dim, query_dim)

        # 门控
        if gate_channels:
            self.gate = nn.Sequential(nn.Linear(query_dim, query_dim), nn.Sigmoid())
        else:
            self.gate = nn.Sequential(nn.Linear(query_dim, 1), nn.Sigmoid())

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # 温度（对数域）
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

        # FFN（decoder）
        self.ffn = nn.Sequential(
            nn.Linear(query_dim, query_dim * 4),
            nn.GELU(),
            nn.Dropout(ffn_drop),
            nn.Linear(query_dim * 4, query_dim),
            nn.Dropout(ffn_drop),
        )

        # 残差缩放
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float))  # decoder
        self.gamma = nn.Parameter(torch.tensor(0.5, dtype=torch.float))        # encoder
        self.register_buffer("gamma_scale", torch.tensor(float(gamma_scale), dtype=torch.float))
        self.register_buffer("ssa_scale", torch.tensor(self.head_dim ** -0.5, dtype=torch.float))

        # init
        for m in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    # ---------- utils ----------
    @staticmethod
    def _ensure_batched_text(text_features: torch.Tensor, B: int) -> torch.Tensor:
        if text_features is None:
            raise ValueError("text_features is None")
        if text_features.dim() == 3:   # (B,T,Ct)
            return text_features
        if text_features.dim() == 2:
            return text_features.unsqueeze(0).expand(B, -1, -1)  \
                if text_features.size(0) != B else text_features.unsqueeze(1)
        if text_features.dim() == 1:   # (Ct,)
            return text_features.view(1, 1, -1).expand(B, 1, -1)
        raise ValueError(f"Unsupported text_features shape: {text_features.shape}")

    @staticmethod
    def _make_text_pad_mask(text_feats: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return (text_feats.float().abs().sum(dim=-1) <= eps)

    @staticmethod
    def _conf_from_attn(attn: torch.Tensor, heads_first: bool):
        # 返回 [B,N,1]
        if attn.dim() != 4:
            raise ValueError("attn must be 4D")
        attn_hn = attn if heads_first else attn.permute(0, 2, 1, 3)  # → [B,H,N,T]
        maxp, _ = attn_hn.max(dim=-1, keepdim=True)                  # [B,H,N,1]
        T_eff = (attn_hn > 0).float().sum(dim=-1, keepdim=True).clamp_min(2.0)
        entropy = -(attn_hn.clamp_min(1e-8) * attn_hn.clamp_min(1e-8).log()).sum(-1, keepdim=True) / T_eff.log()
        entropy = entropy.clamp_min(0.0)
        conf_h = (maxp * (1.0 - entropy)).clamp(0, 1)                # [B,H,N,1]
        return conf_h.mean(dim=1, keepdim=False)                      # [B,N,1]

    # ---------- Decoder ----------
    def forward(self, visual_features: torch.Tensor, text_features: torch.Tensor):
        B, H, W, Cv = visual_features.shape
        x = self.norm1(visual_features).view(B, H * W, Cv)               # (B,N,Cv)
        q_full = self.q_proj(x)

        # 文本
        text_b = self._ensure_batched_text(text_features, B)             # (B,T,Ct)
        pad_mask = self._make_text_pad_mask(text_b)                      # (B,T)
        k_full = self.k_proj(text_b)                                     # (B,T,Cv)
        v_full = self.v_proj(text_b)                                     # (B,T,Cv)

        # 多头
        Hh, Dh = self.num_heads, self.head_dim
        q = F.normalize(q_full, dim=-1, eps=1e-6).view(B, -1, Hh, Dh)    # (B,N,H,Dh)
        k = F.normalize(k_full, dim=-1, eps=1e-6).view(B, -1, Hh, Dh)    # (B,T,H,Dh)
        v = v_full.view(B, -1, Hh, Dh)                                    # (B,T,H,Dh)

        # logits
        scale = torch.clamp(self.logit_scale, min=-self.clamp_logit, max=self.clamp_logit).exp() / math.sqrt(self.d_k)
        sim = torch.einsum('bnhd,bthd->bnht', q, k) * scale               # (B,N,H,T)
        if pad_mask.any():
            sim = sim.masked_fill(pad_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        # 稀疏注意力：Top-K（预 softmax；仅当 K < T）
        if self.use_topk and (self.top_m is not None) and (self.top_m < sim.size(-1)):
            topv, topi = torch.topk(sim, k=self.top_m, dim=-1)
            mask = torch.zeros_like(sim).scatter_(-1, topi, 1.0)
            sim = sim.masked_fill(mask.eq(0), float('-inf'))

        # softmax → Dropout → 归一化
        attn = F.softmax(sim, dim=-1)
        attn = self.attn_drop(attn)
        attn = attn / attn.sum(-1, keepdim=True).clamp_min(1e-6)

        # all-pad 兜底
        all_pad = pad_mask.all(dim=1)
        if all_pad.any():
            attn[all_pad] = 0

        # 置信度（像素级）→ 旁路掩码
        conf_bn1 = self._conf_from_attn(attn, heads_first=False)         # (B,N,1)
        bypass = (conf_bn1 >= self.conf_thresh).to(attn.dtype)           # (B,N,1)

        # 聚合
        aligned_h = torch.einsum('bnht,bthd->bnhd', attn, v)             # (B,N,H,Dh)
        aligned = self.proj_drop(aligned_h.reshape(B, -1, Hh * Dh))      # (B,N,Cv)
        aligned = self.out_proj(aligned)

        # 旁路：低置信像素不注入增益
        aligned = aligned * bypass

        # 残差 + FFN
        gate = self.gate(x)                                              # (B,N,1) or (B,N,Cv)
        y = (x + self.alpha * gate * aligned) if self.add_residual else (self.alpha * aligned)
        y = self.norm2(y)
        y = y + self.ffn(y)
        return y.view(B, H, W, Cv)

    # ---------- Encoder ----------
    def forward_ssa(self, visual_features: torch.Tensor, text_features: torch.Tensor):
        B, H, W, Cv = visual_features.shape
        x = visual_features.view(B, H * W, Cv)
        q_full = self.q_proj(x)

        text_b = self._ensure_batched_text(text_features, B).to(visual_features.dtype)
        pad_mask = self._make_text_pad_mask(text_b)
        k_full = self.k_proj(text_b)
        v_full = self.v_proj(text_b)

        Hh, Dh = self.num_heads, self.head_dim
        q = q_full.view(B, -1, Hh, Dh).permute(0, 2, 1, 3)     # (B,H,N,Dh)
        k = k_full.view(B, -1, Hh, Dh).permute(0, 2, 1, 3)     # (B,H,T,Dh)
        v = v_full.view(B, -1, Hh, Dh).permute(0, 2, 1, 3)     # (B,H,T,Dh)

        # Top-K 稀疏注意力（与 decoder 一致）
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.ssa_scale   # (B,H,N,T)
        if pad_mask.any():
            attn_logits = attn_logits.masked_fill(pad_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        if self.use_topk and (self.top_m is not None) and (self.top_m < attn_logits.size(-1)):
            topv, topi = torch.topk(attn_logits, k=self.top_m, dim=-1)
            mask = torch.zeros_like(attn_logits).scatter_(-1, topi, 1.0)
            attn_logits = attn_logits.masked_fill(mask.eq(0), float('-inf'))

        attn = torch.softmax(attn_logits, dim=-1)                           # (B,H,N,T)
        attn = self.attn_drop(attn)
        attn = attn / attn.sum(-1, keepdim=True).clamp_min(1e-6)

        all_pad = pad_mask.all(dim=1)
        if all_pad.any():
            attn[all_pad] = 0

        conf_bn1 = self._conf_from_attn(attn, heads_first=True)             # (B,N,1)
        bypass = (conf_bn1 >= self.conf_thresh).to(attn.dtype)              # (B,N,1)

        aligned = torch.matmul(attn, v)                                      # (B,H,N,Dh)
        aligned = aligned.permute(0, 2, 1, 3).reshape(B, -1, Hh * Dh)        # (B,N,Cv)
        aligned = self.out_proj(aligned)

        # 旁路
        aligned = aligned * bypass

        y = x + (self.gamma * self.gamma_scale) * aligned
        return y.view(B, H, W, Cv)
