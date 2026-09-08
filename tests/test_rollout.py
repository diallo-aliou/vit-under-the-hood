"""Unit tests for Attention Rollout algorithm and visualization."""

from pathlib import Path

import pytest
import torch

from src.utils.rollout import (
    compute_attention_rollout,
    extract_cls_rollout,
    plot_rollout_comparison,
)
from src.utils.visualize import create_sample_image


def _generate_mock_attentions(
    num_layers: int = 8,
    batch_size: int = 1,
    num_heads: int = 3,
    num_tokens: int = 145,
) -> list[torch.Tensor]:
    """Helper to generate row-normalized attention matrices."""
    layers = []
    for _ in range(num_layers):
        raw = torch.randn(batch_size, num_heads, num_tokens, num_tokens)
        attn = torch.softmax(raw, dim=-1)
        layers.append(attn)
    return layers


def test_compute_attention_rollout_shapes_and_stochasticity() -> None:
    """Verify rollout matrix dimensions and row stochasticity (sum = 1.0)."""
    attns = _generate_mock_attentions(num_layers=8, batch_size=2, num_heads=3, num_tokens=145)
    rollout = compute_attention_rollout(attns, head_fusion="mean")

    assert rollout.shape == (2, 145, 145)

    # Invariant: Every row of the rollout matrix must sum to 1.0
    row_sums = rollout.sum(dim=-1)
    expected_ones = torch.ones_like(row_sums)
    assert torch.allclose(row_sums, expected_ones, atol=1e-4)


def test_compute_attention_rollout_head_fusions() -> None:
    """Verify rollout computation with different head fusion methods."""
    attns = _generate_mock_attentions(num_layers=4, batch_size=1, num_heads=3, num_tokens=17)

    for fusion in ["mean", "max", "min"]:
        r = compute_attention_rollout(attns, head_fusion=fusion)
        assert r.shape == (1, 17, 17)
        assert torch.allclose(r.sum(dim=-1), torch.ones_like(r.sum(dim=-1)), atol=1e-4)

    with pytest.raises(ValueError, match="Unknown head fusion"):
        compute_attention_rollout(attns, head_fusion="invalid")


def test_compute_attention_rollout_discard_ratio() -> None:
    """Verify threshold pruning with discard_ratio."""
    attns = _generate_mock_attentions(num_layers=3, batch_size=1, num_heads=2, num_tokens=17)
    r_no_discard = compute_attention_rollout(attns, discard_ratio=0.0)
    r_discarded = compute_attention_rollout(attns, discard_ratio=0.5)

    assert r_no_discard.shape == r_discarded.shape
    assert torch.allclose(r_discarded.sum(dim=-1), torch.ones_like(r_discarded.sum(dim=-1)), atol=1e-4)


def test_extract_cls_rollout_shapes_and_bounds() -> None:
    """Verify [CLS] attention extraction with and without upsampling."""
    # 145 tokens for image_size=96 and patch_size=8 (12*12 + 1 = 145)
    attns = _generate_mock_attentions(num_layers=4, batch_size=1, num_heads=3, num_tokens=145)
    rollout = compute_attention_rollout(attns)

    # 1. Upsampled heatmap
    heatmap_up = extract_cls_rollout(rollout, patch_size=8, image_size=96, upsample=True)
    assert heatmap_up.shape == (96, 96)
    assert heatmap_up.min().item() >= 0.0
    assert heatmap_up.max().item() <= 1.0 + 1e-6

    # 2. Grid heatmap without upsampling
    heatmap_raw = extract_cls_rollout(rollout, patch_size=8, image_size=96, upsample=False)
    assert heatmap_raw.shape == (12, 12)

    # 3. Invalid token dimension error
    with pytest.raises(ValueError, match="does not match expected tokens"):
        extract_cls_rollout(rollout[:, :50, :50], patch_size=8, image_size=96)


def test_plot_rollout_comparison(tmp_path: Path) -> None:
    """Verify side-by-side rollout plot generation to disk."""
    img = create_sample_image(image_size=96)
    raw_attn = torch.softmax(torch.randn(1, 3, 145, 145), dim=-1)
    attns = _generate_mock_attentions(num_layers=4, batch_size=1, num_heads=3, num_tokens=145)
    rollout = compute_attention_rollout(attns)

    out_file = str(tmp_path / "rollout_compare.png")
    plot_rollout_comparison(img, raw_attn, rollout, patch_size=8, save_path=out_file)

    assert Path(out_file).exists()
    assert Path(out_file).stat().st_size > 0
