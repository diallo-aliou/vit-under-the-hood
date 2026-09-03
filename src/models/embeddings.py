"""Patch and positional embedding modules for Vision Transformer (ViT)."""

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    r"""Splits a 2D image into non-overlapping patches and projects to embedding dimension.

    Args:
        in_channels: Number of input image channels. Default: 3.
        patch_size: Square patch spatial resolution (P). Default: 8.
        embed_dim: Token embedding dimension (D). Default: 192.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, N, D)` where :math:`N = \frac{H \cdot W}{P^2}`
    """

    def __init__(
        self,
        in_channels: int = 3,
        patch_size: int = 8,
        embed_dim: int = 192,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.proj(x)        # (B, D, H/P, W/P)
        x = x.flatten(2)        # (B, D, N)
        x = x.transpose(1, 2)   # (B, N, D)
        return x


class ViTEmbedding(nn.Module):
    r"""Constructs token sequences for the Vision Transformer encoder.

    Extracts patch tokens, prepends a learnable [CLS] classification token,
    adds 1D learnable position embeddings, and applies regularizing dropout.

    Args:
        image_size: Input image spatial resolution (square). Default: 96.
        patch_size: Square patch spatial resolution. Default: 8.
        in_channels: Number of input image channels. Default: 3.
        embed_dim: Token embedding dimension (D). Default: 192.
        dropout: Dropout probability for embedded tokens. Default: 0.1.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, N + 1, D)` where :math:`N = (image\_size / patch\_size)^2`
    """

    def __init__(
        self,
        image_size: int = 96,
        patch_size: int = 8,
        in_channels: int = 3,
        embed_dim: int = 192,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert image_size % patch_size == 0, (
            f"image_size ({image_size}) must be divisible by patch_size ({patch_size})"
        )

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        self.patch_embed = PatchEmbedding(
            in_channels=in_channels,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )

        # Learnable classification token: (1, 1, D)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Learnable 1D position embeddings: (1, N + 1, D)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        self.dropout = nn.Dropout(p=dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        # Truncated normal distribution with std=0.02 matches the original ViT paper
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # Extract patch tokens
        x = self.patch_embed(x)                         # (B, N, D)

        # Broadcast [CLS] token across the batch and prepend
        cls_tokens = self.cls_token.expand(B, -1, -1)   # (B, 1, D)
        x = torch.cat((cls_tokens, x), dim=1)           # (B, N + 1, D)

        # Inject spatial position information and apply dropout
        x = x + self.pos_embed                          # (B, N + 1, D)
        x = self.dropout(x)
        return x
