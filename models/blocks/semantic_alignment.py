import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticAlignmentModule(nn.Module):
    """
    Semantic Alignment Module (SAM)

    This module aligns visual features with textual features using a sparse cross-attention mechanism.
    It takes visual features as queries and textual features as keys/values. The core idea is to
    use a Top-M selection to sparsify the attention, allowing each visual token to focus on the
    most relevant text embeddings. This is inspired by the TextGuidedEnhancer but generalized
    for use at any stage of the network.
    """
    def __init__(self, query_dim, text_dim, top_m=5, temp=0.07, add_residual=True):
        super().__init__()
        self.top_m = top_m
        self.temp = temp
        self.add_residual = add_residual

        # Projection layer for visual features (Query) to match text feature dimension
        self.query_proj = nn.Linear(query_dim, text_dim)

        # Projection layer for text features (Value) to match visual feature dimension
        self.value_proj = nn.Linear(text_dim, query_dim)

        # Feed-forward network to further refine the fused features
        self.ffn = nn.Sequential(
            nn.Linear(query_dim, query_dim * 4),
            nn.GELU(),
            nn.Linear(query_dim * 4, query_dim)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

    def forward(self, visual_features, text_features):
        """
        Args:
            visual_features (torch.Tensor): Visual features, shape (B, H, W, C_visual)
            text_features (torch.Tensor): Text features, shape (B, K, C_text), where K is the number of text tokens.

        Returns:
            torch.Tensor: Text-aligned visual features, shape (B, H, W, C_visual)
        """
        # Store original features for the residual connection
        residual = visual_features
        visual_features = self.norm1(visual_features)

        B, H, W, C_visual = visual_features.shape
        vis_flat = visual_features.view(B, H * W, C_visual)

        # 1. Project visual features to create the Query
        query = F.normalize(self.query_proj(vis_flat), dim=-1)  # (B, H*W, C_text)

        # 2. Use text features as Key and project them for the Value
        key = F.normalize(text_features, dim=-1)  # (B, K, C_text)
        value = self.value_proj(text_features)    # (B, K, C_visual)

        # 3. Compute similarity matrix (attention scores)
        sim_matrix = torch.einsum('bnc,bkc->bnk', query, key)  # (B, H*W, K)

        # 4. Apply Top-M sparsification
        if self.top_m is not None and self.top_m < sim_matrix.size(-1):
            top_v, _ = sim_matrix.topk(self.top_m, dim=-1)
            # Create a mask to keep only the top-m scores
            mask = torch.full_like(sim_matrix, float('-inf'))
            mask.scatter_(-1, sim_matrix.topk(self.top_m, dim=-1)[1], top_v)
            sim_matrix = mask

        # 5. Compute attention weights and aggregate values
        attn_weights = F.softmax(sim_matrix / self.temp, dim=-1)
        aligned_features = torch.einsum('bnk,bkc->bnc', attn_weights, value)  # (B, H*W, C_visual)

        # 6. Apply residual connection and FFN
        if self.add_residual:
            fused_features = residual.view(B, H * W, -1) + aligned_features
        else:
            fused_features = aligned_features

        fused_features = self.norm2(fused_features)
        fused_features = fused_features + self.ffn(fused_features)

        return fused_features.view(B, H, W, C_visual)