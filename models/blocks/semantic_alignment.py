import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SemanticAlignmentModule(nn.Module):
    """
    SAM v2: 可学习缩放 + 真·TopK聚合 + 门控 + Dropout + 统一 Pre-LN
    形状: visual (B,H,W,Cv), text (B,K,Ct) 或 (K,Ct)
    """
    def __init__(
        self,
        query_dim,              # C_visual
        text_dim,               # C_text
        top_m=5,
        use_topk=True,
        attn_drop=0.0,
        proj_drop=0.0,
        ffn_drop=0.0,
        add_residual=True,
        gate_channels=False     # False: 标量门控；True: channel 门控
    ):
        super().__init__()
        self.top_m = top_m
        self.use_topk = use_topk
        self.add_residual = add_residual
        self.gate_channels = gate_channels

        # 预归一化
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        # Q 从视觉到文本维度（用作 d_k）
        self.q_proj = nn.Linear(query_dim, text_dim, bias=False)
        # V 从文本到视觉维度
        self.v_proj = nn.Linear(text_dim, query_dim, bias=False)

        # 可学习缩放（配合 1/sqrt(d_k)）
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1/0.07)))  # ≈ 1/0.07
        self.d_k = float(text_dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # 门控：标量或通道门
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

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(query_dim, query_dim * 4),
            nn.GELU(),
            nn.Dropout(ffn_drop),
            nn.Linear(query_dim * 4, query_dim),
            nn.Dropout(ffn_drop),
        )

        # 简单初始化
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)

    def _ensure_batched_text(self, text_features, B):
        if text_features.dim() == 2:                 # (K,Ct)
            text_features = text_features.unsqueeze(0)  # (1,K,Ct)
        if text_features.size(0) != B:               # broadcast
            text_features = text_features.expand(B, -1, -1).contiguous()
        return text_features

    def forward(self, visual_features, text_features):
        # NHWC -> BN C
        B, H, W, Cv = visual_features.shape
        x = self.norm1(visual_features).view(B, H * W, Cv)    # Pre-LN
        q = F.normalize(self.q_proj(x), dim=-1)               # (B,N,Ct)

        text_features = self._ensure_batched_text(text_features, B)
        k = F.normalize(text_features, dim=-1)                # (B,K,Ct)
        v = self.v_proj(text_features)                        # (B,K,Cv)

        # sim: (B,N,K)
        scale = self.logit_scale.exp() / math.sqrt(self.d_k)
        sim = torch.einsum('bnc,bkc->bnk', q, k) * scale

        if self.use_topk and self.top_m is not None and self.top_m < sim.size(-1):
            # 只算 Top-K 的注意力和值聚合
            top_vals, top_idx = sim.topk(self.top_m, dim=-1)                    # (B,N,M)
            attn = F.softmax(top_vals, dim=-1)
            attn = self.attn_drop(attn)

            # 从 v 里按 (B,N,M) 的 idx 取对应行：(B,N,M,Cv)
            v_exp = v.unsqueeze(1).expand(B, H * W, v.size(1), v.size(2))
            v_sel = torch.gather(v_exp, 2, top_idx.unsqueeze(-1).expand(B, H * W, self.top_m, v.size(2)))
            aligned = (attn.unsqueeze(-1) * v_sel).sum(dim=2)                   # (B,N,Cv)
        else:
            attn = F.softmax(sim, dim=-1)
            attn = self.attn_drop(attn)
            aligned = torch.einsum('bnk,bkc->bnc', attn, v)                     # (B,N,Cv)

        aligned = self.proj_drop(aligned)

        # 门控 + 残差
        if self.gate_channels:
            gate = self.gate(x)                         # (B,N,Cv)
        else:
            gate = self.gate(x)                         # (B,N,1)
        if not self.gate_channels:
            aligned = aligned * gate
        else:
            aligned = aligned * gate

        y = x + aligned if self.add_residual else aligned
        y = self.norm2(y)
        y = y + self.ffn(y)                             # Pre-LN 结构

        return y.view(B, H, W, Cv)
