"""Unit tests for patch and positional embeddings."""

import pytest
import torch

from src.models.embeddings import PatchEmbedding, ViTEmbedding


def test_patch_embedding_shape() -> None:
    """Verify patch projection output dimensions match (B, N, D)."""
    batch_size = 4
    in_channels = 3
    height = width = 96
    patch_size = 8
    embed_dim = 192

    num_patches = (height // patch_size) * (width // patch_size)  # 12 * 12 = 144

    model = PatchEmbedding(
        in_channels=in_channels,
        patch_size=patch_size,
        embed_dim=embed_dim,
    )
    x = torch.randn(batch_size, in_channels, height, width)
    out = model(x)

    assert out.shape == (batch_size, num_patches, embed_dim)


def test_vit_embedding_shape() -> None:
    """Verify ViTEmbedding output shape includes [CLS] token and pos embeddings."""
    batch_size = 2
    image_size = 96
    patch_size = 8
    embed_dim = 192

    num_patches = (image_size // patch_size) ** 2  # 144
    expected_seq_len = num_patches + 1             # 145 (including [CLS])

    model = ViTEmbedding(
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
    )
    x = torch.randn(batch_size, 3, image_size, image_size)
    out = model(x)

    assert out.shape == (batch_size, expected_seq_len, embed_dim)


def test_vit_embedding_gradients() -> None:
    """Verify learnable parameters require gradients."""
    model = ViTEmbedding(image_size=96, patch_size=8, embed_dim=192)

    assert model.cls_token.requires_grad
    assert model.pos_embed.requires_grad


def test_indivisible_image_and_patch_size() -> None:
    """Verify that an AssertionError is raised if image_size % patch_size != 0."""
    with pytest.raises(AssertionError):
        ViTEmbedding(image_size=95, patch_size=8)
