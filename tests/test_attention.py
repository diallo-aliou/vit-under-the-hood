"""Unit tests for Multi-Head Self-Attention module."""

import pytest
import torch

from src.models.attention import MultiHeadAttention


def test_attention_output_shape() -> None:
    """Verify that MultiHeadAttention preserves input sequence shape (B, N, D)."""
    batch_size = 4
    seq_len = 145  # 12*12 patches + 1 [CLS]
    embed_dim = 192
    num_heads = 3

    model = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
    x = torch.randn(batch_size, seq_len, embed_dim)
    out = model(x)

    assert out.shape == (batch_size, seq_len, embed_dim)


def test_invalid_head_configuration() -> None:
    """Verify AssertionError when embed_dim is not divisible by num_heads."""
    with pytest.raises(AssertionError):
        MultiHeadAttention(embed_dim=192, num_heads=5)


def test_attention_weights_shape_and_caching() -> None:
    """Verify return_attention flag and last_attn_weights caching property."""
    batch_size = 2
    seq_len = 65
    embed_dim = 128
    num_heads = 4

    model = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
    x = torch.randn(batch_size, seq_len, embed_dim)

    # 1. Test return_attention=True
    out, weights = model(x, return_attention=True)
    expected_weights_shape = (batch_size, num_heads, seq_len, seq_len)

    assert out.shape == (batch_size, seq_len, embed_dim)
    assert weights.shape == expected_weights_shape

    # 2. Test cached last_attn_weights
    assert model.last_attn_weights is not None
    assert model.last_attn_weights.shape == expected_weights_shape
    assert torch.allclose(weights, model.last_attn_weights)


def test_softmax_stochasticity() -> None:
    """Verify attention probability rows sum to 1.0 (valid probability distribution)."""
    batch_size = 2
    seq_len = 50
    embed_dim = 64
    num_heads = 2

    model = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
    x = torch.randn(batch_size, seq_len, embed_dim)
    _, weights = model(x, return_attention=True)

    # Sum along target token dimension (last axis)
    row_sums = weights.sum(dim=-1)
    expected_ones = torch.ones_like(row_sums)

    assert torch.allclose(row_sums, expected_ones, atol=1e-6)


def test_head_masking_ablation() -> None:
    """Verify head_mask zeroes out target attention heads for Phase 7 ablation."""
    batch_size = 2
    seq_len = 20
    embed_dim = 64
    num_heads = 4

    model = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
    x = torch.randn(batch_size, seq_len, embed_dim)

    # Disable head index 1 and 3 (mask = [1, 0, 1, 0])
    head_mask = torch.tensor([1.0, 0.0, 1.0, 0.0])
    _, weights = model(x, return_attention=True, head_mask=head_mask)

    # Active heads should have non-zero attention
    assert not torch.allclose(weights[:, 0], torch.zeros_like(weights[:, 0]))
    assert not torch.allclose(weights[:, 2], torch.zeros_like(weights[:, 2]))

    # Ablated heads must be strictly zeroed out
    assert torch.allclose(weights[:, 1], torch.zeros_like(weights[:, 1]))
    assert torch.allclose(weights[:, 3], torch.zeros_like(weights[:, 3]))


def test_attention_gradient_flow() -> None:
    """Verify backpropagation produces valid gradients for all learnable parameters."""
    embed_dim = 96
    num_heads = 3
    model = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)

    x = torch.randn(2, 32, embed_dim, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()

    assert model.qkv.weight.grad is not None
    assert not torch.isnan(model.qkv.weight.grad).any()
    assert model.proj.weight.grad is not None
    assert not torch.isnan(model.proj.weight.grad).any()
    assert x.grad is not None


def test_eval_mode_determinism() -> None:
    """Verify eval mode disables dropout and ensures reproducible outputs."""
    embed_dim = 64
    num_heads = 2
    model = MultiHeadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        attn_drop=0.5,
        proj_drop=0.5,
    )
    model.eval()

    x = torch.randn(2, 20, embed_dim)
    out1 = model(x)
    out2 = model(x)

    assert torch.allclose(out1, out2)
