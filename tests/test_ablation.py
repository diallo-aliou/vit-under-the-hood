"""Unit tests for Attention Head Specialization and Ablation experiments."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.experiments.ablation import (
    compute_mean_attention_distance,
    compute_patch_distance_matrix,
    evaluate_single_head_ablation,
    plot_ablation_matrix,
    plot_attention_distance_heatmap,
    plot_pruning_retention_curve,
    prune_heads_cumulative,
)
from src.models.vit import VisionTransformer


@pytest.fixture
def miniature_vit() -> VisionTransformer:
    """Fixture providing a tiny VisionTransformer for rapid deterministic testing."""
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
    """Fixture providing a synthetic DataLoader."""
    torch.manual_seed(42)
    images = torch.randn(8, 3, 32, 32)
    targets = torch.randint(0, 4, (8,))
    dataset = TensorDataset(images, targets)
    return DataLoader(dataset, batch_size=4)


class TestAblation:
    """Test suite for head distance, single-head ablation, and greedy pruning."""

    def test_patch_distance_matrix(self) -> None:
        """Verify geometric properties of patch distance matrix."""
        dist = compute_patch_distance_matrix(image_size=32, patch_size=8)
        # 32 / 8 = 4 patches per dim -> 16 total patches
        assert dist.shape == (16, 16)
        # Distance to self is zero
        assert torch.allclose(torch.diag(dist), torch.zeros(16))
        # Symmetry
        assert torch.allclose(dist, dist.T)
        # Max distance bounded by diagonal of 32x32 image (approx 45.25 pixels)
        max_dist = dist.max().item()
        assert 0 < max_dist <= 32 * np.sqrt(2)

    def test_mean_attention_distance(self, miniature_vit: VisionTransformer) -> None:
        """Verify mean attention distance returns valid spatial measurements."""
        images = torch.randn(2, 3, 32, 32)
        distances = compute_mean_attention_distance(miniature_vit, images)

        assert distances.shape == (2, 2)  # (depth=2, num_heads=2)
        assert (distances > 0).all()
        assert (distances <= 32 * np.sqrt(2)).all()

    def test_single_head_ablation(
        self,
        miniature_vit: VisionTransformer,
        dummy_dataloader: DataLoader,
    ) -> None:
        """Verify single-head ablation returns correct drop matrix dimensions."""
        baseline_acc, drops = evaluate_single_head_ablation(
            miniature_vit,
            dummy_dataloader,
            device="cpu",
        )

        assert isinstance(baseline_acc, float)
        assert 0.0 <= baseline_acc <= 100.0
        assert drops.shape == (2, 2)
        assert isinstance(drops, np.ndarray)

    def test_prune_heads_cumulative(
        self,
        miniature_vit: VisionTransformer,
        dummy_dataloader: DataLoader,
    ) -> None:
        """Verify cumulative greedy pruning over defined ratios."""
        mock_drops = np.array([[0.5, 2.0], [-0.5, 1.0]], dtype=np.float32)
        ratios = [0.0, 0.25, 0.5, 1.0]

        results = prune_heads_cumulative(
            miniature_vit,
            dummy_dataloader,
            drop_matrix=mock_drops,
            prune_ratios=ratios,
            device="cpu",
        )

        assert len(results) == len(ratios)
        for r in ratios:
            assert r in results
            assert 0.0 <= results[r] <= 100.0

    def test_ablation_plots(self, tmp_path) -> None:
        """Verify ablation visualizations render without error."""
        dists = torch.tensor([[15.2, 28.4], [18.1, 35.0]])
        drops = np.array([[0.5, -0.2], [1.8, 0.1]])
        prune_results = {0.0: 60.0, 0.25: 58.5, 0.5: 52.0}

        path_dists = tmp_path / "dists.png"
        path_drops = tmp_path / "drops.png"
        path_prune = tmp_path / "prune.png"

        plot_attention_distance_heatmap(dists, save_path=str(path_dists))
        plot_ablation_matrix(drops, save_path=str(path_drops))
        plot_pruning_retention_curve(prune_results, baseline_acc=60.0, save_path=str(path_prune))

        assert path_dists.exists() and path_dists.stat().st_size > 0
        assert path_drops.exists() and path_drops.stat().st_size > 0
        assert path_prune.exists() and path_prune.stat().st_size > 0
