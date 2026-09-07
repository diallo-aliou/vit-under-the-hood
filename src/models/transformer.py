"""Transformer Encoder modules for Vision Transformer (ViT).

Implements the multi-layer perceptron (MLP), the Pre-LayerNorm Transformer
encoder block with residual connections, and the full Transformer encoder stack.
"""

import torch
import torch.nn as nn

from src.models.attention import MultiHeadAttention


class MLP(nn.Module):
    r"""Feed-Forward Network (MLP) block for Transformer encoder.

    Applies two linear transformations with a GELU non-linearity and dropout in between.

    Args:
        in_features: Input token dimension (:math:`D`).
        hidden_features: Expanded inner hidden dimension. Default: `4 * in_features`.
        out_features: Output token dimension. Default: `in_features`.
        drop: Dropout probability. Default: 0.0.

    Shape:
        - Input: :math:`(B, N, D)`
        - Output: :math:`(B, N, D)`
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        out_features = out_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p=drop)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize linear layer weights with truncated normal distribution."""
        nn.init.trunc_normal_(self.fc1.weight, std=0.02)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        nn.init.trunc_normal_(self.fc2.weight, std=0.02)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, N, D) -> (B, N, 4D)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        # (B, N, 4D) -> (B, N, D)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerEncoderBlock(nn.Module):
    r"""Transformer Encoder Block with Pre-LayerNorm and residual connections.

    Follows the Pre-LN design:
        .. math::
            \hat{x} = x + \text{Attention}(\text{LN}_1(x)) \\
            x_{\text{out}} = \hat{x} + \text{MLP}(\text{LN}_2(\hat{x}))

    Args:
        embed_dim: Embedding dimension (:math:`D`). Default: 192.
        num_heads: Number of attention heads (:math:`h`). Default: 3.
        mlp_ratio: Ratio of MLP hidden dimension to embedding dimension. Default: 4.0.
        qkv_bias: If True, adds learnable bias to Q, K, V projections. Default: True.
        drop: Dropout probability for MLP and projection layers. Default: 0.0.
        attn_drop: Dropout probability for attention weights. Default: 0.0.

    Shape:
        - Input: :math:`(B, N, D)`
        - Output: :math:`(B, N, D)` or tuple `((B, N, D), (B, h, N, N))` if
          `return_attention=True`.
    """

    def __init__(
        self,
        embed_dim: int = 192,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            drop=drop,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
        head_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        r"""Forward pass through the Pre-LN Transformer encoder block.

        Args:
            x: Input token sequence :math:`(B, N, D)`.
            return_attention: If True, also returns attention weights :math:`(B, h, N, N)`.
            head_mask: Optional mask tensor to ablate specific attention heads.

        Returns:
            Processed tokens :math:`(B, N, D)` or tuple `(tokens, attention_weights)`.
        """
        # Pre-LN sublayer 1: Multi-Head Self-Attention with residual
        norm_x = self.norm1(x)
        if return_attention:
            attn_out, attn_weights = self.attn(
                norm_x,
                return_attention=True,
                head_mask=head_mask,
            )
        else:
            attn_out = self.attn(
                norm_x,
                return_attention=False,
                head_mask=head_mask,
            )
            attn_weights = None

        x = x + attn_out

        # Pre-LN sublayer 2: Feed-Forward MLP with residual
        x = x + self.mlp(self.norm2(x))

        if return_attention:
            return x, attn_weights
        return x


class TransformerEncoder(nn.Module):
    r"""Stack of Transformer Encoder Blocks.

    Args:
        depth: Number of encoder blocks (:math:`L`). Default: 8.
        embed_dim: Embedding dimension (:math:`D`). Default: 192.
        num_heads: Number of attention heads (:math:`h`). Default: 3.
        mlp_ratio: Ratio of MLP hidden dimension to embedding dimension. Default: 4.0.
        qkv_bias: If True, adds learnable bias to Q, K, V projections. Default: True.
        drop: Dropout probability for MLP and projection layers. Default: 0.0.
        attn_drop: Dropout probability for attention weights. Default: 0.0.

    Shape:
        - Input: :math:`(B, N, D)`
        - Output: :math:`(B, N, D)` or tuple `((B, N, D), list[Tensor])` if
          `return_all_attentions=True`.
    """

    def __init__(
        self,
        depth: int = 8,
        embed_dim: int = 192,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        return_all_attentions: bool = False,
        head_masks: list[torch.Tensor | None] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        r"""Forward pass through all stacked encoder blocks.

        Args:
            x: Input sequence :math:`(B, N, D)`.
            return_all_attentions: If True, collects and returns attention maps from all layers.
            head_masks: Optional list of head mask tensors, one per encoder block.

        Returns:
            Output sequence :math:`(B, N, D)` or tuple `(output, all_attentions)`.
        """
        all_attentions: list[torch.Tensor] = []

        for idx, layer in enumerate(self.layers):
            mask = head_masks[idx] if head_masks is not None else None
            if return_all_attentions:
                x, attn = layer(x, return_attention=True, head_mask=mask)
                all_attentions.append(attn)
            else:
                x = layer(x, return_attention=False, head_mask=mask)

        if return_all_attentions:
            return x, all_attentions
        return x


if __name__ == "__main__":
    # Quick sanity check with STL-10 dimensions (B=2, N=145, D=192, L=8)
    dummy_x = torch.randn(2, 145, 192)
    encoder = TransformerEncoder(depth=8, embed_dim=192, num_heads=3)

    out, attns = encoder(dummy_x, return_all_attentions=True)
    print(f"Input shape:                {dummy_x.shape}")
    print(f"Encoder output shape:       {out.shape}")
    print(f"Number of collected layers: {len(attns)}")
    print(f"Layer 0 attention shape:    {attns[0].shape}")
