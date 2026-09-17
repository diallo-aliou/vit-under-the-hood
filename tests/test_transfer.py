"""Unit tests for Transfer Learning and Pretrained ViT utilities."""

import pytest
import torch

from src.experiments.transfer import (
    build_transfer_vit,
    extract_torchvision_vit_attentions,
    get_transfer_transforms,
    plot_scratch_vs_transfer_comparison,
)


class TestTransfer:
    """Test suite for torchvision ViT adaptation and attention extraction."""

    def test_build_transfer_vit(self) -> None:
        """Verify model head adaptation and backbone freezing."""
        model = build_transfer_vit(num_classes=10, pretrained=False, freeze_backbone=True)

        assert model.heads.head.out_features == 10
        assert model.heads.head.weight.requires_grad is True

        # Verify backbone is frozen
        assert model.conv_proj.weight.requires_grad is False
        assert model.encoder.layers[0].self_attention.in_proj_weight.requires_grad is False

    def test_get_transfer_transforms(self) -> None:
        """Verify transfer transform shapes and pipeline."""
        train_tf, val_tf = get_transfer_transforms(image_size=224)
        assert train_tf is not None
        assert val_tf is not None

    def test_extract_torchvision_vit_attentions(self) -> None:
        """Verify forward hooks collect multi-head attention weights from all 12 layers."""
        model = build_transfer_vit(num_classes=10, pretrained=False)
        dummy_input = torch.randn(1, 3, 224, 224)

        logits, attentions = extract_torchvision_vit_attentions(model, dummy_input)

        assert logits.shape == (1, 10)
        assert len(attentions) == 12  # 12 layers in ViT-B/16
        # 14x14 = 196 patches + 1 [CLS] token = 197 tokens, 12 attention heads
        assert attentions[0].shape == (1, 12, 197, 197)
        # Check row stochasticity
        row_sum = attentions[0][0, 0, 0].sum().item()
        assert row_sum == pytest.approx(1.0, abs=1e-4)

    def test_plot_scratch_vs_transfer_comparison(self, tmp_path) -> None:
        """Verify comparison visualization renders without error."""
        image = torch.rand(3, 96, 96)
        heat_scratch = torch.rand(96, 96)
        heat_transfer = torch.rand(224, 224)

        save_path = tmp_path / "transfer_comp.png"
        plot_scratch_vs_transfer_comparison(
            image=image,
            rollout_scratch=heat_scratch,
            rollout_transfer=heat_transfer,
            label="Airplane",
            save_path=str(save_path),
        )

        assert save_path.exists()
        assert save_path.stat().st_size > 0
