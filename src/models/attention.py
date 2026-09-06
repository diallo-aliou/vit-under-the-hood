"""Multi-Head Self-Attention (MHSA) module for Vision Transformer (ViT)."""

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    r"""Multi-Head Self-Attention (MHSA) mechanism built from scratch.

    Computes scaled dot-product attention across multiple representation heads in parallel.
    Uses a fused linear projection for Query, Key, and Value to maximize hardware efficiency.

    Args:
        embed_dim: Total dimension of the model tokens (:math:`D`). Default: 192.
        num_heads: Number of parallel attention heads (:math:`h`). Default: 3.
        qkv_bias: If True, adds a learnable bias to Q, K, V projections. Default: True.
        attn_drop: Dropout probability applied to the attention weight matrix. Default: 0.0.
        proj_drop: Dropout probability applied after the output projection. Default: 0.0.

    Shape:
        - Input: :math:`(B, N, D)` where :math:`B` is batch size, :math:`N` is sequence length,
          and :math:`D` is `embed_dim`.
        - Output: :math:`(B, N, D)` or tuple of :math:`((B, N, D), (B, h, N, N))` if
          `return_attention=True`.
    """

    def __init__(
        self,
        embed_dim: int = 192,
        num_heads: int = 3,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        # Scaling factor: 1 / sqrt(d_k)
        self.scale = self.head_dim**-0.5

        # Fused linear projection for Q, K, V in a single GEMM operation: (D -> 3 * D)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(p=attn_drop)

        # Output projection recombining all heads: (D -> D)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(p=proj_drop)

        # Cache for the most recent attention weights (used for Rollout and introspection)
        self.last_attn_weights: torch.Tensor | None = None

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize linear layer weights with truncated normal distribution."""
        nn.init.trunc_normal_(self.qkv.weight, std=0.02)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
        head_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        r"""Compute multi-head self-attention for the input sequence.

        Args:
            x: Input token sequence of shape :math:`(B, N, D)`.
            return_attention: If True, also returns the attention map :math:`(B, h, N, N)`.
            head_mask: Optional mask tensor of shape :math:`(h,)` or :math:`(1, h, 1, 1)`
                to ablate specific attention heads (Phase 7 experiments).

        Returns:
            Output tensor of shape :math:`(B, N, D)`, or tuple `(output, attention_weights)`.
        """
        B, N, D = x.shape

        # 1. Project input to Q, K, V simultaneously: (B, N, 3 * D)
        qkv = self.qkv(x)

        # 2. Reshape to split into heads and separate Q, K, V:
        # (B, N, 3 * D) -> (B, N, 3, h, d_k) -> (3, B, h, N, d_k)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)  # Each: (B, h, N, d_k)

        # 3. Scaled Dot-Product Attention:
        # (B, h, N, d_k) @ (B, h, d_k, N) -> (B, h, N, N)
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn_scores.softmax(dim=-1)

        # 4. Optional head ablation masking (Phase 7):
        if head_mask is not None:
            # Reshape head_mask to broadcast over (B, h, N, N) if provided as (h,)
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            attn = attn * head_mask

        # 5. Cache raw attention weights for rollout / hooks
        self.last_attn_weights = attn.detach()

        # 6. Apply dropout on attention probabilities
        attn_dropped = self.attn_drop(attn)

        # 7. Aggregate values and recombine heads:
        # (B, h, N, N) @ (B, h, N, d_k) -> (B, h, N, d_k) -> (B, N, h, d_k) -> (B, N, D)
        out = (attn_dropped @ v).transpose(1, 2).reshape(B, N, D)

        # 8. Output projection and dropout
        out = self.proj(out)
        out = self.proj_drop(out)

        if return_attention:
            return out, attn
        return out


if __name__ == "__main__":
    # Sanity check with STL-10 token dimensions (144 patches + 1 [CLS] = 145 tokens, D=192)
    dummy_input = torch.randn(2, 145, 192)
    mha = MultiHeadAttention(embed_dim=192, num_heads=3)

    output, weights = mha(dummy_input, return_attention=True)
    print(f"Input shape:             {dummy_input.shape}")
    print(f"Output shape:            {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print(f"Row sum verification:    {weights[0, 0, 0].sum().item():.4f} (expected: 1.0000)")
