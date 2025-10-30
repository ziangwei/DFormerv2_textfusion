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
        gate_channels: bool = True,
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

        self.use_geo_consistency = True
        self.geo_mlp = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )

        # 残差缩放
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float))  # decoder
        self.gamma = nn.Parameter(torch.tensor(0.5, dtype=torch.float))        # encoder
        self.register_buffer("gamma_scale", torch.tensor(float(gamma_scale), dtype=torch.float))
        self.register_buffer("ssa_scale", torch.tensor(self.head_dim ** -0.5, dtype=torch.float))

        # init
        for m in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

        try:
            from torch.nn import LazyLinear
            self.text_scale = LazyLinear(2)  # 输出 [Δalpha, Δbeta]
        except Exception:
            self.text_scale = None  # 老版本 PyTorch 兜底

        # --- 几何融合与温度的基线 ---
        self.alpha_base = 0.25  # 几何融合基线（attn 与 attn_geo 的混合系数基线）
        self.beta_base = 5.0  # pairwise 深度亲和的基线温度
        self.max_pairwise_n = getattr(self, "max_pairwise_n", 1600)  # 小图阈值，防 OOM


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
    def forward(self, visual_features: torch.Tensor, text_features: torch.Tensor,
                geo_mask=None,  # 支持 [B, N] 或 [B, N, N]
                return_attn=False):
        B, H, W, Cv = visual_features.shape
        N = H * W
        x = self.norm1(visual_features).view(B, N, Cv)
        q_full = self.q_proj(x)

        # 文本
        text_b = self._ensure_batched_text(text_features, B)
        pad_mask = self._make_text_pad_mask(text_b)
        k_full = self.k_proj(text_b)
        v_full = self.v_proj(text_b)

        # 多头
        Hh, Dh = self.num_heads, self.head_dim
        q = F.normalize(q_full, dim=-1, eps=1e-6).view(B, -1, Hh, Dh)
        k = F.normalize(k_full, dim=-1, eps=1e-6).view(B, -1, Hh, Dh)
        v = v_full.view(B, -1, Hh, Dh)

        # logits
        scale = torch.clamp(self.logit_scale, min=-self.clamp_logit, max=self.clamp_logit).exp() / math.sqrt(self.d_k)
        sim = torch.einsum('bnhd,bthd->bnht', q, k) * scale  # (B, N, H, T)

        geo_gate_flat = None
        geo_context = None
        if isinstance(geo_mask, dict) and {"depth", "gate", "beta"} <= set(geo_mask.keys()):
            depth = geo_mask["depth"]  # [B,1,H,W]
            gate_map = geo_mask["gate"].clamp(0.3, 1.0)  # [B,1,H,W]
            beta_m = geo_mask["beta"].clamp_min(0.0)
            geo_context = (depth, gate_map, beta_m)
            geo_gate_flat = gate_map.flatten(1).unsqueeze(-1).to(sim.dtype)  # [B,N,1]
            gate_bias = torch.log(gate_map.flatten(1).unsqueeze(-1).unsqueeze(-1).clamp_min(1e-6).to(sim.dtype))
            sim = sim + gate_bias
        elif isinstance(geo_mask, torch.Tensor) and geo_mask.dim() in {2, 4}:
            if geo_mask.dim() == 4:
                gate_map = geo_mask.clamp(0.3, 1.0)
                geo_gate_flat = gate_map.flatten(1).unsqueeze(-1).to(sim.dtype)
                gate_bias = torch.log(
                    gate_map.flatten(1).unsqueeze(-1).unsqueeze(-1).clamp_min(1e-6).to(sim.dtype)
                )
            else:
                gate_map = geo_mask.clamp(0.3, 1.0)
                geo_gate_flat = gate_map.unsqueeze(-1).to(sim.dtype)
                gate_bias = torch.log(
                    gate_map.unsqueeze(-1).unsqueeze(-1).clamp_min(1e-6).to(sim.dtype)
                )
            sim = sim + gate_bias

        if pad_mask.any():
            sim = sim.masked_fill(pad_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        # Top-K 稀疏注意力
        if self.use_topk and (self.top_m is not None) and (self.top_m < sim.size(-1)):
            topv, topi = torch.topk(sim, k=self.top_m, dim=-1)
            mask = torch.zeros_like(sim).scatter_(-1, topi, 1.0)
            sim = sim.masked_fill(mask.eq(0), float('-inf'))

        # softmax → Dropout → 归一化
        attn = F.softmax(sim, dim=-1)
        attn = self.attn_drop(attn)
        attn = attn / attn.sum(-1, keepdim=True).clamp_min(1e-6)

        # === attn: [B, Hh, N, T] 先标准化到 [B, N, Hh, T] ===
        attn_nhT = attn.permute(0, 2, 1, 3).contiguous()  # [B,N,Hh,T]

        # --- 文本条件调制：从 text_features 得到 Δalpha/Δbeta ---
        if text_features is not None:
            if text_features.dim() == 2:  # (T,Ct) -> (B,T,Ct)
                B = attn_nhT.size(0)
                text_features = text_features.unsqueeze(0).expand(B, -1, -1)
            t_pool = text_features.mean(dim=1)  # [B,Ct]
            if self.text_scale is not None:
                delta = self.text_scale(t_pool)  # [B,2]
                delta_alpha = torch.tanh(delta[:, 0:1]) * 0.10  # +/-0.1
                delta_beta = F.softplus(delta[:, 1:2]) * 0.50  # +[0,0.5]
            else:
                delta_alpha = attn_nhT.new_zeros(attn_nhT.size(0), 1)
                delta_beta = attn_nhT.new_zeros(attn_nhT.size(0), 1)
        else:
            delta_alpha = attn_nhT.new_zeros(attn_nhT.size(0), 1)
            delta_beta = attn_nhT.new_zeros(attn_nhT.size(0), 1)

        alpha_txt = (self.alpha_base + delta_alpha).clamp(0.05, 0.40)  # [B,1]
        beta_txt = self.beta_base + delta_beta  # [B,1]

        # --- 几何分支 ---
        if geo_context is not None:
            depth, gate_map, beta_m = geo_context

            B, _, Hs, Ws = depth.shape
            N_cur = Hs * Ws
            if (Hs, Ws) != (int(N_cur ** 0.5), int(N_cur ** 0.5)):
                pass

            # 小图才做 pairwise（用文本调制 beta 与 DGN beta 融合）
            if N_cur <= self.max_pairwise_n:
                dvec = depth.reshape(B, -1).unsqueeze(-1)  # [B,N,1]
                ddiff = (dvec - dvec.transpose(1, 2)).abs()  # [B,N,N]
                beta_flat = beta_m.reshape(B, -1)
                beta_pair = 0.5 * (beta_flat.unsqueeze(-1) + beta_flat.unsqueeze(-2))  # [B,N,N]
                beta_eff = (beta_txt.view(B, 1, 1) + beta_pair).clamp(min=1.0)
                geo_w = torch.exp(-beta_eff * ddiff).to(attn_nhT.dtype)
                geo_w = geo_w / (geo_w.sum(dim=-1, keepdim=True) + 1e-6)

                Bn, Nn, Hh, Tt = attn_nhT.shape
                attn_flat = attn_nhT.reshape(Bn, Nn, Hh * Tt)  # [B,N,H*T]
                attn_geo = torch.bmm(geo_w, attn_flat).view(Bn, Nn, Hh, Tt)

                a = alpha_txt.view(B, 1, 1, 1)  # [B,1,1,1]
                attn_nhT = (1 - a) * attn_nhT + a * attn_geo
                # 融合后再次盖 pad + 归一化
                if 'pad_mask' in locals() and pad_mask is not None:
                    attn_nhT = attn_nhT.masked_fill(pad_mask.unsqueeze(1).unsqueeze(2), 0.0)
                attn_nhT = attn_nhT / (attn_nhT.sum(dim=-1, keepdim=True) + 1e-6)

        # 回到 [B,Hh,N,T]
        attn = attn_nhT.permute(0, 2, 1, 3).contiguous()

        # all-pad 兜底
        all_pad = pad_mask.all(dim=1)
        if all_pad.any():
            attn[all_pad] = 0

        # 置信度 → 旁路掩码
        conf_bn1 = self._conf_from_attn(attn, heads_first=False)
        conf_scale = self.conf_thresh + (1.0 - self.conf_thresh) * conf_bn1.clamp(0.0, 1.0)

        # 聚合
        aligned_h = torch.einsum('bnht,bthd->bnhd', attn, v)
        aligned = self.proj_drop(aligned_h.reshape(B, -1, Hh * Dh))
        aligned = self.out_proj(aligned)

        # 置信软门控
        aligned = aligned * conf_scale.to(aligned.dtype)

        # 残差 + FFN
        gate = self.gate(x)
        if geo_gate_flat is not None:
            gate = gate * geo_gate_flat.to(gate.dtype)
        y = (x + self.alpha * gate * aligned) if self.add_residual else (self.alpha * aligned)
        y = self.norm2(y)
        y = y + self.ffn(y)

        if return_attn:
            attn_vis = attn.mean(dim=2)
            return y.view(B, H, W, Cv), attn_vis

        return y.view(B, H, W, Cv)

    # ---------- Encoder ----------
    # semantic_alignment.py Line ~153
    def forward_ssa(self, visual_features, text_features,
                    geo_mask=None,  # 现在可以是 [B, N] 或 [B, N, N]
                    return_attn=False):
        B, H, W, Cv = visual_features.shape
        x = visual_features.view(B, H * W, Cv)
        q_full = self.q_proj(x)

        text_b = self._ensure_batched_text(text_features, B).to(visual_features.dtype)
        pad_mask = self._make_text_pad_mask(text_b)
        k_full = self.k_proj(text_b)
        v_full = self.v_proj(text_b)

        Hh, Dh = self.num_heads, self.head_dim
        q = q_full.view(B, -1, Hh, Dh).permute(0, 2, 1, 3)  # (B, H, N, Dh)
        k = k_full.view(B, -1, Hh, Dh).permute(0, 2, 1, 3)  # (B, H, T, Dh)
        v = v_full.view(B, -1, Hh, Dh).permute(0, 2, 1, 3)  # (B, H, T, Dh)

        # Top-K 稀疏注意力
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.ssa_scale  # (B, H, N, T)

        geo_gate_flat = None
        if isinstance(geo_mask, dict) and "gate" in geo_mask:
            gate_map = geo_mask["gate"].clamp(0.4, 1.0)
            geo_gate_flat = gate_map.flatten(1).unsqueeze(-1).to(attn_logits.dtype)
            gate_bias = torch.log(
                gate_map.flatten(1).unsqueeze(1).unsqueeze(-1).clamp_min(1e-6).to(attn_logits.dtype)
            )
            attn_logits = attn_logits + gate_bias
        elif isinstance(geo_mask, torch.Tensor) and geo_mask.dim() == 2 and geo_mask.size(1) == q.size(2):
            gate_map = geo_mask.clamp(0.4, 1.0)
            geo_gate_flat = gate_map.unsqueeze(-1).to(attn_logits.dtype)
            gate_bias = torch.log(gate_map.unsqueeze(1).unsqueeze(-1).clamp_min(1e-6).to(attn_logits.dtype))
            attn_logits = attn_logits + gate_bias


        if pad_mask.any():
            attn_logits = attn_logits.masked_fill(pad_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        if self.use_topk and (self.top_m is not None) and (self.top_m < attn_logits.size(-1)):
            topv, topi = torch.topk(attn_logits, k=self.top_m, dim=-1)
            mask = torch.zeros_like(attn_logits).scatter_(-1, topi, 1.0)
            attn_logits = attn_logits.masked_fill(mask.eq(0), float('-inf'))

        attn = torch.softmax(attn_logits, dim=-1)
        attn = self.attn_drop(attn)
        attn = attn / attn.sum(-1, keepdim=True).clamp_min(1e-6)

        # attn: [B, Hh, N, T] -> [B,N,Hh,T]
        attn_nhT = attn.permute(0, 2, 1, 3).contiguous()

        # 文本调制 alpha（只影响注入强度，不做 pairwise）
        if text_features is not None:
            if text_features.dim() == 2:
                B = attn_nhT.size(0)
                text_features = text_features.unsqueeze(0).expand(B, -1, -1)
            t_pool = text_features.mean(dim=1)
            if self.text_scale is not None:
                delta = self.text_scale(t_pool)
                delta_alpha = torch.tanh(delta[:, 0:1]) * 0.10
            else:
                delta_alpha = attn_nhT.new_zeros(attn_nhT.size(0), 1)
        else:
            delta_alpha = attn_nhT.new_zeros(attn_nhT.size(0), 1)

        alpha_txt = (self.alpha_base + delta_alpha).clamp(0.05, 0.35)  # Encoder更保守

        # 回到 [B,Hh,N,T] 继续后续 Vz 聚合；将 alpha_txt 用作残差注入强度（乘在你的增量上）
        attn = attn_nhT.permute(0, 2, 1, 3).contiguous()
        # 后续：delta_feat = (attn @ Vz) * alpha_txt[...,0,0,0]  或在你的残差处乘 alpha_txt

        all_pad = pad_mask.all(dim=1)
        if all_pad.any():
            attn[all_pad] = 0

        conf_bn1 = self._conf_from_attn(attn, heads_first=True)
        conf_scale = self.conf_thresh + (1.0 - self.conf_thresh) * conf_bn1.clamp(0.0, 1.0)

        aligned = torch.matmul(attn, v)  # (B, H, N, Dh)
        aligned = aligned.permute(0, 2, 1, 3).reshape(B, -1, Hh * Dh)
        aligned = self.out_proj(aligned)

        aligned = aligned * conf_scale.to(aligned.dtype)
        if geo_gate_flat is not None:
            aligned = aligned * geo_gate_flat.to(aligned.dtype)

        y = x + (self.gamma * self.gamma_scale) * aligned

        if return_attn:
            attn_vis = attn.mean(dim=1)
            return y.view(B, H, W, Cv), attn_vis
        return y.view(B, H, W, Cv)
