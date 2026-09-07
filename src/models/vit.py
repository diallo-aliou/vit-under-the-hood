"""Vision Transformer (ViT) architecture implementation."""

import torch
import torch.nn as nn

from src.models.embeddings import ViTEmbedding
from src.models.transformer import TransformerEncoder


class VisionTransformer(nn.Module):
    r"""Vision Transformer (ViT) for image classification.

    Built from scratch following Dosovitskiy et al. (2020) with a Pre-LayerNorm
    Transformer encoder backbone and configurable depth/embedding dimensions.

    Args:
        image_size: Input square image resolution (H=W). Default: 96 (STL-10).
        patch_size: Square patch spatial resolution (P). Default: 8.
        in_channels: Number of image input channels. Default: 3.
        num_classes: Number of target classification classes. Default: 10.
        embed_dim: Token embedding dimension (D). Default: 192.
        depth: Number of Transformer encoder blocks (L). Default: 8.
        num_heads: Number of attention heads (h). Default: 3.
        mlp_ratio: Ratio of MLP hidden dimension to embedding dimension. Default: 4.0.
        qkv_bias: If True, adds learnable bias to Q, K, V projections. Default: True.
        drop_rate: Dropout probability for token embeddings, MLP, and projections. Default: 0.0.
        attn_drop_rate: Dropout probability for attention weights. Default: 0.0.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, \text{num\_classes})` or tuple `((B, \text{num\_classes}), list[Tensor])`
          if `return_all_attentions=True`.
    """

    def __init__(
        self,
        image_size: int = 96,
        patch_size: int = 8,
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 192,
        depth: int = 8,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth

        # 1. Patch extraction, [CLS] token prepending, positional encoding
        self.embedding = ViTEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            dropout=drop_rate,
        )

        # 2. Stack of Pre-LN Transformer encoder blocks
        self.encoder = TransformerEncoder(
            depth=depth,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
        )

        # 3. Final normalization layer before prediction
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # 4. Classification head operating on the [CLS] representation
        if num_classes > 0:
            self.head: nn.Module = nn.Linear(embed_dim, num_classes)
            self._init_head()
        else:
            self.head = nn.Identity()

    def _init_head(self) -> None:
        """Initialize classification head weights with truncated normal distribution."""
        if isinstance(self.head, nn.Linear):
            nn.init.trunc_normal_(self.head.weight, std=0.02)
            if self.head.bias is not None:
                nn.init.zeros_(self.head.bias)

    def forward_features(
        self,
        x: torch.Tensor,
        return_all_attentions: bool = False,
        head_masks: list[torch.Tensor | None] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        r"""Extract normalized token representations from input images.

        Args:
            x: Input images of shape :math:`(B, C, H, W)`.
            return_all_attentions: If True, returns collected attention weights.
            head_masks: Optional list of head masks for ablation experiments.

        Returns:
            Normalized token sequence :math:`(B, N+1, D)` or tuple `(tokens, all_attentions)`.
        """
        # (B, C, H, W) -> (B, N+1, D)
        x = self.embedding(x)

        # Process tokens through all encoder layers
        if return_all_attentions:
            x, all_attentions = self.encoder(
                x,
                return_all_attentions=True,
                head_masks=head_masks,
            )
        else:
            x = self.encoder(
                x,
                return_all_attentions=False,
                head_masks=head_masks,
            )
            all_attentions = None

        # Apply final LayerNorm
        x = self.norm(x)

        if return_all_attentions:
            return x, all_attentions
        return x

    def forward(
        self,
        x: torch.Tensor,
        return_all_attentions: bool = False,
        head_masks: list[torch.Tensor | None] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        r"""Forward pass computing class logits for input images.

        Args:
            x: Input images of shape :math:`(B, C, H, W)`.
            return_all_attentions: If True, returns class logits and all attention maps.
            head_masks: Optional head masks for Phase 7 ablation experiments.

        Returns:
            Class logits :math:`(B, \text{num\_classes})` or tuple `(logits, all_attentions)`.
        """
        if return_all_attentions:
            x, all_attentions = self.forward_features(
                x,
                return_all_attentions=True,
                head_masks=head_masks,
            )
        else:
            x = self.forward_features(
                x,
                return_all_attentions=False,
                head_masks=head_masks,
            )
            all_attentions = None

        # Extract the [CLS] classification token (index 0)
        cls_token = x[:, 0]  # (B, D)

        # Map to class logits: (B, D) -> (B, num_classes)
        logits = self.head(cls_token)

        if return_all_attentions:
            return logits, all_attentions
        return logits

    def get_num_params(self) -> tuple[int, int]:
        """Return (total_parameters, trainable_parameters)."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def vit_tiny(num_classes: int = 10, **kwargs) -> VisionTransformer:
    r"""Factory for lightweight ViT-Tiny preset adapted for STL-10 (96x96, ~5.8M params)."""
    return VisionTransformer(
        image_size=96,
        patch_size=8,
        in_channels=3,
        num_classes=num_classes,
        embed_dim=192,
        depth=8,
        num_heads=3,
        mlp_ratio=4.0,
        **kwargs,
    )


def vit_small(num_classes: int = 10, **kwargs) -> VisionTransformer:
    r"""Factory for ViT-Small preset (~22M params)."""
    return VisionTransformer(
        image_size=96,
        patch_size=8,
        in_channels=3,
        num_classes=num_classes,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        **kwargs,
    )


if __name__ == "__main__":
    # Sanity check with ViT-Tiny on a batch of 2 STL-10 images (96x96)
    dummy_imgs = torch.randn(2, 3, 96, 96)
    model = vit_tiny(num_classes=10)

    total_p, trainable_p = model.get_num_params()
    logits, all_attns = model(dummy_imgs, return_all_attentions=True)

    print(f"Model parameters:          {total_p:,} (trainable: {trainable_p:,})")
    print(f"Input batch shape:         {dummy_imgs.shape}")
    print(f"Output logits shape:       {logits.shape}")
    print(f"Number of attention layers: {len(all_attns)}")
    print(f"Attention map shape:       {all_attns[0].shape}")
