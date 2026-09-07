"""Unit tests for visualization utilities."""

from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T

from src.utils.visualize import (
    create_sample_image,
    denormalize,
    plot_attention_heads,
    plot_attention_layer_progression,
    plot_augmentation_examples,
    plot_dataset_samples,
    plot_patch_grid,
    plot_prediction_grid,
    plot_prediction_topk,
)


class DummyViT(nn.Module):
    """Minimal dummy model for prediction plotting tests."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.classifier = nn.Linear(3 * 32 * 32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = nn.functional.adaptive_avg_pool2d(x, (32, 32)).flatten(1)
        return self.classifier(flat)


def test_denormalize_invariants() -> None:
    """Verify that denormalize correctly restores values to [0, 1]."""
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    orig = torch.rand(2, 3, 32, 32)
    norm = T.Normalize(mean=mean, std=std)(orig)

    denorm = denormalize(norm, mean=mean, std=std)
    assert denorm.shape == orig.shape
    assert denorm.min().item() >= 0.0
    assert denorm.max().item() <= 1.0
    assert torch.allclose(orig, denorm, atol=1e-5)


def test_create_sample_image() -> None:
    """Verify sample image generation dimensions and bounds."""
    img = create_sample_image(image_size=64)
    assert img.shape == (1, 3, 64, 64)
    assert img.min().item() >= 0.0
    assert img.max().item() <= 1.0


def test_plot_patch_grid(tmp_path: Path) -> None:
    """Verify patch grid plot generation to disk."""
    img = create_sample_image(image_size=32)
    out_file = str(tmp_path / "patch_grid.png")
    plot_patch_grid(img, patch_size=8, save_path=out_file)
    assert Path(out_file).exists()
    assert Path(out_file).stat().st_size > 0


def test_plot_dataset_samples(tmp_path: Path) -> None:
    """Verify dataset samples plot generation."""
    imgs = torch.rand(6, 3, 32, 32)
    labels = torch.tensor([0, 1, 2, 3, 4, 5])
    class_names = ["c0", "c1", "c2", "c3", "c4", "c5"]
    out_file = str(tmp_path / "samples.png")
    plot_dataset_samples(imgs, labels, class_names, num_samples=6, nrows=2, save_path=out_file)
    assert Path(out_file).exists()


def test_plot_augmentation_examples(tmp_path: Path) -> None:
    """Verify augmentation comparison plot generation."""
    img = torch.rand(3, 32, 32)
    transform = T.Compose([T.RandomHorizontalFlip(p=1.0), T.ToTensor()])
    out_file = str(tmp_path / "augmentations.png")
    plot_augmentation_examples(img, transform, num_examples=2, save_path=out_file)
    assert Path(out_file).exists()


def test_plot_prediction_grid_and_topk(tmp_path: Path) -> None:
    """Verify prediction grid and top-k bar chart generation."""
    model = DummyViT(num_classes=5)
    imgs = torch.rand(4, 3, 32, 32)
    labels = torch.tensor([0, 1, 2, 3])
    class_names = ["c0", "c1", "c2", "c3", "c4"]

    out_grid = str(tmp_path / "pred_grid.png")
    plot_prediction_grid(model, imgs, labels, class_names, num_samples=4, save_path=out_grid)
    assert Path(out_grid).exists()

    probs = torch.softmax(torch.randn(5), dim=-1)
    out_topk = str(tmp_path / "topk.png")
    plot_prediction_topk(imgs[0], true_label=1, probs=probs, class_names=class_names, top_k=3, save_path=out_topk)
    assert Path(out_topk).exists()


def test_plot_attention_heads_and_progression(tmp_path: Path) -> None:
    """Verify multi-head attention map and depth progression plots."""
    img = torch.rand(1, 3, 32, 32)
    # (B, H, N, N) where N = (32/8)^2 + 1 = 16 + 1 = 17
    N = 17
    attn = torch.softmax(torch.randn(1, 3, N, N), dim=-1)

    out_heads = str(tmp_path / "attn_heads.png")
    plot_attention_heads(img, attn, layer_idx=0, patch_size=8, save_path=out_heads)
    assert Path(out_heads).exists()

    all_attns = [torch.softmax(torch.randn(1, 3, N, N), dim=-1) for _ in range(4)]
    out_prog = str(tmp_path / "attn_prog.png")
    plot_attention_layer_progression(img, all_attns, layers=[0, 1, 2, 3], patch_size=8, save_path=out_prog)
    assert Path(out_prog).exists()
