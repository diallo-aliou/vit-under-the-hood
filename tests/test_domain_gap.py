"""Unit tests for Domain Gap and Shape Bias experiments."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.experiments.domain_gap import (
    create_sobel_kernels,
    evaluate_domain_shift,
    plot_photo_vs_sketch_comparison,
    to_edge_sketch,
)
from src.models.vit import VisionTransformer


@pytest.fixture
def miniature_vit() -> VisionTransformer:
    """Fixture providing a tiny VisionTransformer."""
    torch.manual_seed(42)
    return VisionTransformer(
        image_size=32,
        patch_size=8,
        in_channels=3,
        num_classes=4,
        embed_dim=32,
        depth=2,
        num_heads=2,
    )


@pytest.fixture
def dummy_dataloader() -> DataLoader:
    """Fixture providing a synthetic image DataLoader."""
    torch.manual_seed(42)
    images = torch.rand(8, 3, 32, 32)
    targets = torch.randint(0, 4, (8,))
    dataset = TensorDataset(images, targets)
    return DataLoader(dataset, batch_size=4)


class TestDomainGap:
    """Test suite for edge sketch transformation and domain shift evaluation."""

    def test_sobel_kernels(self) -> None:
        """Verify dimensions and orthogonality of Sobel kernels."""
        kx, ky = create_sobel_kernels()
        assert kx.shape == (1, 1, 3, 3)
        assert ky.shape == (1, 1, 3, 3)
        assert kx.sum().item() == pytest.approx(0.0)
        assert ky.sum().item() == pytest.approx(0.0)

    def test_to_edge_sketch_shapes_and_bounds(self) -> None:
        """Verify edge sketch returns clean 3-channel bounded images."""
        batch = torch.rand(3, 3, 32, 32)
        sketches = to_edge_sketch(batch, invert=True)

        assert sketches.shape == (3, 3, 32, 32)
        assert sketches.min().item() >= 0.0
        assert sketches.max().item() <= 1.0

        # Test single image 3D tensor
        single = torch.rand(3, 32, 32)
        single_sketch = to_edge_sketch(single, invert=False)
        assert single_sketch.shape == (3, 32, 32)
        assert single_sketch.min().item() >= 0.0
        assert single_sketch.max().item() <= 1.0

    def test_evaluate_domain_shift(
        self,
        miniature_vit: VisionTransformer,
        dummy_dataloader: DataLoader,
    ) -> None:
        """Verify domain shift evaluation computes valid metrics."""
        results = evaluate_domain_shift(miniature_vit, dummy_dataloader, device="cpu")

        assert "photo_acc" in results
        assert "sketch_acc" in results
        assert "shape_retention_index" in results

        assert 0.0 <= results["photo_acc"] <= 100.0
        assert 0.0 <= results["sketch_acc"] <= 100.0
        assert results["shape_retention_index"] >= 0.0

    def test_plot_photo_vs_sketch_comparison(self, tmp_path) -> None:
        """Verify photo vs sketch visual comparison renders without error."""
        photo = torch.rand(3, 32, 32)
        sketch = torch.rand(3, 32, 32)
        attn_p = torch.rand(32, 32)
        attn_s = torch.rand(32, 32)

        save_file = tmp_path / "domain_comp.png"
        plot_photo_vs_sketch_comparison(
            photo=photo,
            sketch=sketch,
            attn_photo=attn_p,
            attn_sketch=attn_s,
            label="Dog",
            save_path=str(save_file),
        )

        assert save_file.exists()
        assert save_file.stat().st_size > 0
