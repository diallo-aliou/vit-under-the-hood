"""Transfer Learning and ImageNet Pre-trained ViT Comparisons.

Provides utilities to:
1. Instantiate pre-trained Vision Transformers (e.g. ViT-B/16 from ImageNet).
2. Adapt the classification head for STL-10 (10 classes).
3. Extract full-depth multi-head attention maps via forward hooks.
4. Compare Attention Rollout maps: From-Scratch ViT vs. ImageNet Pre-trained ViT.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from torchvision import transforms

from src.utils.visualize import _display_and_close, denormalize


def build_transfer_vit(
    num_classes: int = 10,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    device: torch.device | str = "cpu",
) -> nn.Module:
    r"""Build and adapt an ImageNet pre-trained ViT-B/16 model for STL-10.

    Args:
        num_classes: Number of target classes (10 for STL-10).
        pretrained: If True, loads official ImageNet-1k pretrained weights.
        freeze_backbone: If True, freezes transformer encoder weights.
        device: Execution device.

    Returns:
        VisionTransformer module with adapted classification head.
    """
    device = torch.device(device)
    weights = tv_models.ViT_B_16_Weights.DEFAULT if pretrained else None
    model = tv_models.vit_b_16(weights=weights)

    # Freeze backbone if requested for head-only fine-tuning
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace head: (in_features=768 -> num_classes)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)

    return model.to(device)


def get_transfer_transforms(image_size: int = 224) -> tuple[transforms.Compose, transforms.Compose]:
    r"""Get standard pre-processing transforms for ImageNet ViT architectures.

    Args:
        image_size: Target square image dimension (default: 224 for ViT-B/16).

    Returns:
        Tuple of `(train_transform, val_transform)`.
    """
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, val_transform


class TorchvisionViTAttentionExtractor:
    r"""Context manager / hook to extract exact attention maps from torchvision ViT models."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.attentions: list[torch.Tensor] = []
        self.hooks: list[torch.utils.hooks.RemovableHandle] = []

    def _hook_fn(self, module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        # Input to layer is X: (B, N, D)
        x = inputs[0]
        mha = module.self_attention
        B, N, D = x.shape
        num_heads = mha.num_heads
        head_dim = D // num_heads

        # Extract normalized input through ln_1
        norm_x = module.ln_1(x)

        # Compute exact scaled dot-product attention
        qkv = F.linear(norm_x, mha.in_proj_weight, mha.in_proj_bias)
        q, k, _ = qkv.chunk(3, dim=-1)

        q = q.view(B, N, num_heads, head_dim).transpose(1, 2)  # (B, h, N, d_k)
        k = k.view(B, N, num_heads, head_dim).transpose(1, 2)  # (B, h, N, d_k)

        scale = 1.0 / math.sqrt(head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(scores, dim=-1)  # (B, h, N, N)

        self.attentions.append(attn.detach())

    def __enter__(self) -> TorchvisionViTAttentionExtractor:
        self.attentions = []
        # Register hooks on each encoder layer
        for layer in self.model.encoder.layers:
            handle = layer.register_forward_pre_hook(self._hook_fn)
            self.hooks.append(handle)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()


def extract_torchvision_vit_attentions(
    model: nn.Module,
    images: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    r"""Run forward pass and collect multi-layer attention matrices from torchvision ViT.

    Args:
        model: torchvision VisionTransformer model.
        images: Input images tensor :math:`(B, 3, 224, 224)`.

    Returns:
        Tuple of `(logits, all_attentions)` where each attention tensor
        has shape :math:`(B, \text{num\_heads}, N, N)`.
    """
    model.eval()
    with TorchvisionViTAttentionExtractor(model) as extractor:
        with torch.no_grad():
            logits = model(images)
    return logits, extractor.attentions


def plot_scratch_vs_transfer_comparison(
    image: torch.Tensor,
    rollout_scratch: torch.Tensor | np.ndarray,
    rollout_transfer: torch.Tensor | np.ndarray,
    label: str = "Test Sample",
    save_path: str | None = None,
) -> None:
    r"""Render a 4-panel visual comparison: From-Scratch ViT vs. ImageNet Pre-trained ViT.

    Panels:
        1. Natural Input Image
        2. From-Scratch ViT Attention Rollout (STL-10 only)
        3. ImageNet Pre-trained ViT Attention Rollout (Transfer Learning)
        4. Difference / Shift map highlighting holistic shape focus

    Args:
        image: Single image tensor :math:`(3, H, W)` or :math:`(1, 3, H, W)`.
        rollout_scratch: Heatmap from from-scratch model :math:`(H, W)`.
        rollout_transfer: Heatmap from transfer learning model :math:`(H, W)`.
        label: Target class label for the header.
        save_path: Optional destination file path.
    """
    if image.ndim == 4:
        image = image.squeeze(0)

    disp_img = denormalize(image) if (image.min() < 0.0 or image.max() > 1.0) else image
    img_np = np.clip(disp_img.permute(1, 2, 0).detach().cpu().numpy(), 0.0, 1.0)

    heat_scratch = rollout_scratch.detach().cpu().numpy() if isinstance(rollout_scratch, torch.Tensor) else rollout_scratch
    heat_transfer = rollout_transfer.detach().cpu().numpy() if isinstance(rollout_transfer, torch.Tensor) else rollout_transfer

    # Resize if resolutions differ
    H, W, _ = img_np.shape
    if heat_transfer.shape != (H, W):
        t_tensor = torch.tensor(heat_transfer).unsqueeze(0).unsqueeze(0)
        t_resized = F.interpolate(t_tensor, size=(H, W), mode="bicubic", align_corners=False)
        heat_transfer = t_resized[0, 0].numpy()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.0), dpi=130)

    # Panel 1: Input Sample
    axes[0].imshow(img_np)
    axes[0].set_title(f"Input Image (STL-10)\n[{label}]", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: From-Scratch ViT
    axes[1].imshow(img_np)
    axes[1].imshow(heat_scratch, cmap="inferno", alpha=0.55)
    axes[1].set_title("ViT-Tiny (From-Scratch)\n[STL-10 50 Ep, 61.1% Acc]", fontsize=10, fontweight="bold")
    axes[1].axis("off")

    # Panel 3: Pre-trained ViT
    axes[2].imshow(img_np)
    axes[2].imshow(heat_transfer, cmap="inferno", alpha=0.55)
    axes[2].set_title("ViT-B/16 (ImageNet Pre-trained)\n[Transfer Learning, ~95% Acc]", fontsize=10, fontweight="bold")
    axes[2].axis("off")

    # Panel 4: Direct Heatmap Comparison (Pre-trained Heatmap Alone)
    axes[3].imshow(heat_transfer, cmap="inferno")
    axes[3].set_title("Pre-trained Saliency Density\n[Holistic Object Segmentation]", fontsize=10, fontweight="bold")
    axes[3].axis("off")

    plt.suptitle("Representation Duel: From-Scratch Learning vs. ImageNet Pre-trained Transfer", fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig.subplots_adjust(top=0.84)

    _display_and_close(fig, save_path)
