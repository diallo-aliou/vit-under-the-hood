"""Domain Gap and Texture vs. Shape Bias Evaluation.

Evaluates how Vision Transformers respond to domain shift by transforming
natural photos into edge sketches (retaining shape while removing texture).
Quantifies the Shape Retention Index (SRI) following Geirhos et al. (ICLR 2019).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.utils.visualize import _display_and_close, denormalize


def create_sobel_kernels(device: torch.device | str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    r"""Create 3x3 horizontal and vertical Sobel convolution kernels.

    Args:
        device: Device to allocate kernel tensors on.

    Returns:
        Tuple of `(kernel_x, kernel_y)` tensors with shape :math:`(1, 1, 3, 3)`.
    """
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=device)
    ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=device)
    return kx.view(1, 1, 3, 3), ky.view(1, 1, 3, 3)


def to_edge_sketch(
    images: torch.Tensor,
    invert: bool = True,
    threshold: float = 0.15,
) -> torch.Tensor:
    r"""Transform natural RGB images into clean 3-channel edge sketches.

    Applies differentiable Sobel edge detection on luminance, followed by
    contrast normalization and optional sketch inversion (dark lines on white).

    Args:
        images: Image tensor of shape :math:`(B, 3, H, W)` or :math:`(3, H, W)`.
        invert: If True, produces dark pencil sketch on white background.
        threshold: Noise suppression threshold :math:`[0.0, 1.0]`.

    Returns:
        Edge sketch tensor of shape :math:`(B, 3, H, W)` in range :math:`[0.0, 1.0]`.
    """
    single_image = images.ndim == 3
    if single_image:
        images = images.unsqueeze(0)

    B, C, H, W = images.shape
    device = images.device

    # Denormalize if image has negative values from standardization
    if images.min() < 0.0 or images.max() > 1.0:
        clean_imgs = torch.stack([denormalize(img) for img in images])
    else:
        clean_imgs = images

    clean_imgs = torch.clamp(clean_imgs, 0.0, 1.0)

    # Convert RGB to grayscale luminance: Y = 0.2989*R + 0.5870*G + 0.1140*B
    weights = torch.tensor([0.2989, 0.5870, 0.1140], device=device).view(1, 3, 1, 1)
    gray = (clean_imgs * weights).sum(dim=1, keepdim=True)  # (B, 1, H, W)

    kx, ky = create_sobel_kernels(device=device)

    # Convolve with reflection padding to preserve boundaries
    gray_pad = F.pad(gray, (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(gray_pad, kx)
    gy = F.conv2d(gray_pad, ky)

    # Compute gradient magnitude
    grad_mag = torch.sqrt(gx**2 + gy**2 + 1e-8)

    # Min-max normalize per image
    mins = grad_mag.view(B, -1).min(dim=-1)[0].view(B, 1, 1, 1)
    maxs = grad_mag.view(B, -1).max(dim=-1)[0].view(B, 1, 1, 1)
    norm_edges = (grad_mag - mins) / (maxs - mins + 1e-8)

    # Soft thresholding to suppress sensor noise
    edges = torch.clamp((norm_edges - threshold) / (1.0 - threshold + 1e-8), 0.0, 1.0)

    if invert:
        sketch = 1.0 - edges  # White background with dark pencil strokes
    else:
        sketch = edges

    # Replicate across 3 channels for ViT compatibility: (B, 1, H, W) -> (B, 3, H, W)
    sketch_3ch = sketch.repeat(1, 3, 1, 1)

    return sketch_3ch.squeeze(0) if single_image else sketch_3ch


def evaluate_domain_shift(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device | str = "cpu",
    max_batches: int | None = None,
) -> dict[str, float]:
    r"""Evaluate model accuracy on natural photos vs. edge sketches to measure shape bias.

    Args:
        model: ViT model instance.
        dataloader: Test DataLoader providing natural images.
        device: Target execution device.
        max_batches: Optional limit on evaluation batches.

    Returns:
        Dictionary containing:
        - `photo_acc`: Validation accuracy on natural photos (%).
        - `sketch_acc`: Validation accuracy on edge sketches (%).
        - `shape_retention_index`: Ratio :math:`(\text{sketch\_acc} / \text{photo\_acc}) \times 100\%`.
    """
    device = torch.device(device)
    model.to(device)
    model.eval()

    photo_correct = 0
    sketch_correct = 0
    total = 0

    with torch.no_grad():
        for b_idx, (images, targets) in enumerate(dataloader):
            if max_batches is not None and b_idx >= max_batches:
                break

            images = images.to(device)
            targets = targets.to(device)

            # 1. Natural Photo evaluation
            photo_logits = model(images)
            photo_preds = photo_logits.argmax(dim=-1)
            photo_correct += (photo_preds == targets).sum().item()

            # 2. Sketch evaluation (remove texture, preserve shape)
            sketches = to_edge_sketch(images, invert=True)
            sketch_logits = model(sketches)
            sketch_preds = sketch_logits.argmax(dim=-1)
            sketch_correct += (sketch_preds == targets).sum().item()

            total += targets.size(0)

    photo_acc = (photo_correct / max(total, 1)) * 100.0
    sketch_acc = (sketch_correct / max(total, 1)) * 100.0
    sri = (sketch_acc / max(photo_acc, 1e-8)) * 100.0

    print(f"Domain Shift Evaluation (N={total} samples):")
    print(f"  Natural Photos Accuracy:  {photo_acc:.2f}%")
    print(f"  Edge Sketches Accuracy:   {sketch_acc:.2f}%")
    print(f"  Shape Retention Index:    {sri:.1f}%")

    return {
        "photo_acc": photo_acc,
        "sketch_acc": sketch_acc,
        "shape_retention_index": sri,
    }


def plot_photo_vs_sketch_comparison(
    photo: torch.Tensor,
    sketch: torch.Tensor,
    attn_photo: torch.Tensor | np.ndarray,
    attn_sketch: torch.Tensor | np.ndarray,
    label: str = "Test Sample",
    save_path: str | None = None,
) -> None:
    r"""Plot side-by-side comparison of Photo vs. Sketch and their corresponding attention maps.

    Args:
        photo: Single image tensor :math:`(3, H, W)`.
        sketch: Single sketch tensor :math:`(3, H, W)`.
        attn_photo: 2D attention heatmap for photo :math:`(H, W)`.
        attn_sketch: 2D attention heatmap for sketch :math:`(H, W)`.
        label: Class label name for header.
        save_path: Optional destination file path.
    """
    if photo.ndim == 4:
        photo = photo.squeeze(0)
    if sketch.ndim == 4:
        sketch = sketch.squeeze(0)

    disp_photo = denormalize(photo) if (photo.min() < 0.0 or photo.max() > 1.0) else photo
    photo_np = np.clip(disp_photo.permute(1, 2, 0).detach().cpu().numpy(), 0.0, 1.0)
    sketch_np = np.clip(sketch.permute(1, 2, 0).detach().cpu().numpy(), 0.0, 1.0)

    if isinstance(attn_photo, torch.Tensor):
        attn_photo_np = attn_photo.detach().cpu().numpy()
    else:
        attn_photo_np = np.asarray(attn_photo)

    if isinstance(attn_sketch, torch.Tensor):
        attn_sketch_np = attn_sketch.detach().cpu().numpy()
    else:
        attn_sketch_np = np.asarray(attn_sketch)

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), dpi=130)

    # Panel 1: Natural Photo
    axes[0].imshow(photo_np)
    axes[0].set_title(f"Natural Photo\n[{label}]", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: Photo Attention Overlay
    axes[1].imshow(photo_np)
    axes[1].imshow(attn_photo_np, cmap="inferno", alpha=0.55)
    axes[1].set_title("Photo Attention\n[Texture + Shape Cues]", fontsize=10, fontweight="bold")
    axes[1].axis("off")

    # Panel 3: Edge Sketch
    axes[2].imshow(sketch_np)
    axes[2].set_title("Edge Sketch (Sobel)\n[Texture Stripped]", fontsize=10, fontweight="bold")
    axes[2].axis("off")

    # Panel 4: Sketch Attention Overlay
    axes[3].imshow(sketch_np)
    axes[3].imshow(attn_sketch_np, cmap="inferno", alpha=0.55)
    axes[3].set_title("Sketch Attention\n[Pure Geometry Focus]", fontsize=10, fontweight="bold")
    axes[3].axis("off")

    plt.suptitle("Domain Gap Analysis: Attention Robustness under Texture Removal", fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig.subplots_adjust(top=0.84)

    _display_and_close(fig, save_path)
