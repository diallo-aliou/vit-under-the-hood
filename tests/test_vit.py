"""Unit and integration tests for Transformer Encoder and full Vision Transformer."""

import torch
import torch.nn as nn

from src.models.transformer import MLP, TransformerEncoder, TransformerEncoderBlock
from src.models.vit import VisionTransformer, vit_small, vit_tiny


def test_mlp_shape_and_forward() -> None:
    """Verify MLP block preserves sequence shape (B, N, D)."""
    batch_size = 2
    seq_len = 50
    embed_dim = 64
    mlp = MLP(in_features=embed_dim, hidden_features=embed_dim * 4)

    x = torch.randn(batch_size, seq_len, embed_dim)
    out = mlp(x)

    assert out.shape == (batch_size, seq_len, embed_dim)


def test_encoder_block_shape_and_attention() -> None:
    """Verify TransformerEncoderBlock output shape and optional attention return."""
    batch_size = 2
    seq_len = 65
    embed_dim = 96
    num_heads = 3

    block = TransformerEncoderBlock(embed_dim=embed_dim, num_heads=num_heads)
    x = torch.randn(batch_size, seq_len, embed_dim)

    # Standard forward
    out = block(x)
    assert out.shape == (batch_size, seq_len, embed_dim)

    # Forward with return_attention
    out, attn = block(x, return_attention=True)
    assert out.shape == (batch_size, seq_len, embed_dim)
    assert attn.shape == (batch_size, num_heads, seq_len, seq_len)


def test_transformer_encoder_all_attentions() -> None:
    """Verify TransformerEncoder stacks layers and collects all attention maps."""
    batch_size = 2
    seq_len = 32
    embed_dim = 64
    depth = 4
    num_heads = 2

    encoder = TransformerEncoder(
        depth=depth,
        embed_dim=embed_dim,
        num_heads=num_heads,
    )
    x = torch.randn(batch_size, seq_len, embed_dim)
    out, attns = encoder(x, return_all_attentions=True)

    assert out.shape == (batch_size, seq_len, embed_dim)
    assert len(attns) == depth
    for attn in attns:
        assert attn.shape == (batch_size, num_heads, seq_len, seq_len)


def test_vit_tiny_forward_shape() -> None:
    """Verify end-to-end forward pass of vit_tiny on STL-10 dimensions (96x96 -> 10)."""
    batch_size = 2
    model = vit_tiny(num_classes=10)
    x = torch.randn(batch_size, 3, 96, 96)

    logits = model(x)
    assert logits.shape == (batch_size, 10)


def test_vit_parameter_count() -> None:
    """Verify vit_tiny and vit_small parameter counts match expected budgets."""
    tiny_model = vit_tiny(num_classes=10)
    total_tiny, trainable_tiny = tiny_model.get_num_params()

    # vit_tiny (D=192, L=8, H=3, MLP=4x) should be ~3.6M parameters
    assert 3_000_000 < total_tiny < 4_500_000
    assert total_tiny == trainable_tiny

    small_model = vit_small(num_classes=10)
    total_small, _ = small_model.get_num_params()
    # vit_small (D=384, L=12, H=6, MLP=4x) should be ~22M parameters
    assert 20_000_000 < total_small < 25_000_000


def test_vit_feature_extractor_mode() -> None:
    """Verify that num_classes=0 acts as a feature extractor returning [CLS] tokens."""
    batch_size = 2
    model = VisionTransformer(image_size=96, patch_size=8, num_classes=0)
    x = torch.randn(batch_size, 3, 96, 96)

    features = model(x)
    assert features.shape == (batch_size, 192)


def test_vit_return_all_attentions() -> None:
    """Verify model forward pass returns all layer attention maps for Rollout."""
    batch_size = 2
    model = vit_tiny(num_classes=10)
    x = torch.randn(batch_size, 3, 96, 96)

    logits, all_attns = model(x, return_all_attentions=True)
    assert logits.shape == (batch_size, 10)
    assert len(all_attns) == model.depth
    # 96 / 8 = 12 -> 12 * 12 = 144 patches + 1 [CLS] = 145 tokens
    expected_attn_shape = (batch_size, 3, 145, 145)
    for layer_attn in all_attns:
        assert layer_attn.shape == expected_attn_shape


def test_vit_end_to_end_gradient_flow() -> None:
    """Verify full backpropagation from CrossEntropyLoss to input and patch embeddings."""
    model = vit_tiny(num_classes=10)
    x = torch.randn(2, 3, 96, 96, requires_grad=True)
    targets = torch.tensor([0, 5])

    logits = model(x)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, targets)
    loss.backward()

    # Gradients must reach input, patch embedding, cls token, pos embed, and head
    assert x.grad is not None
    assert model.embedding.patch_embed.proj.weight.grad is not None
    assert model.embedding.cls_token.grad is not None
    assert model.embedding.pos_embed.grad is not None
    assert model.encoder.layers[0].attn.qkv.weight.grad is not None
    assert isinstance(model.head, nn.Linear)
    assert model.head.weight.grad is not None

    # Gradients must be finite (no NaN or Inf)
    assert not torch.isnan(model.head.weight.grad).any()
    assert not torch.isnan(model.embedding.patch_embed.proj.weight.grad).any()
