"""Attention Rollout implementation following Abnar & Zuidema (ACL 2020).

Quantifying Attention Flow in Transformers:
Tracks the information flow from input tokens to deep representations by
accounting for residual connections and multi-layer attention routing.
"""


import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from src.utils.visualize import _display_and_close, denormalize


def compute_attention_rollout(
    all_attentions: list[torch.Tensor] | torch.Tensor,
    discard_ratio: float = 0.0,
    head_fusion: str = "mean",
) -> torch.Tensor:
    r"""Compute Attention Rollout across transformer encoder layers.

    Implements the recursive formulation from Abnar & Zuidema (2020):
    .. math::
        \hat{A}_l = 0.5 \cdot \bar{A}_l + 0.5 \cdot I
    .. math::
        R_0 = \hat{A}_0, \quad R_l = \hat{A}_l \cdot R_{l-1}

    Args:
        all_attentions: Sequence of attention weight tensors from all layers.
            Each tensor has shape :math:`(B, H, N, N)` or :math:`(H, N, N)`.
        discard_ratio: Fraction of smallest attention weights to discard (0.0 to 1.0).
        head_fusion: Strategy to aggregate attention heads ('mean', 'max', 'min').

    Returns:
        Cumulative rollout matrix :math:`R \in \mathbb{R}^{B \times N \times N}`.
    """
    if isinstance(all_attentions, torch.Tensor) and all_attentions.ndim == 5:
        # Shape: (L, B, H, N, N)
        layers = [all_attentions[i] for i in range(all_attentions.shape[0])]
    else:
        layers = list(all_attentions)

    if not layers:
        raise ValueError("all_attentions cannot be empty.")

    # Normalize input tensors to shape (B, H, N, N)
    standardized_layers: list[torch.Tensor] = []
    for layer_attn in layers:
        if layer_attn.ndim == 3:
            standardized_layers.append(layer_attn.unsqueeze(0))
        elif layer_attn.ndim == 4:
            standardized_layers.append(layer_attn)
        else:
            raise ValueError(f"Expected 3D or 4D attention tensor, got ndim={layer_attn.ndim}")

    B, _, N, _ = standardized_layers[0].shape
    device = standardized_layers[0].device
    dtype = standardized_layers[0].dtype

    rollout = torch.eye(N, device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)

    for layer_attn in standardized_layers:
        # 1. Aggregate attention heads
        if head_fusion == "mean":
            fused = layer_attn.mean(dim=1)  # (B, N, N)
        elif head_fusion == "max":
            fused, _ = layer_attn.max(dim=1)
            fused = fused / (fused.sum(dim=-1, keepdim=True) + 1e-8)
        elif head_fusion == "min":
            fused, _ = layer_attn.min(dim=1)
            fused = fused / (fused.sum(dim=-1, keepdim=True) + 1e-8)
        else:
            raise ValueError(f"Unknown head fusion '{head_fusion}'. Supported: 'mean', 'max', 'min'")

        # 2. Add identity matrix to account for residual connections
        identity = torch.eye(N, device=device, dtype=dtype).unsqueeze(0)
        a_hat = 0.5 * fused + 0.5 * identity

        # 3. Normalize rows to sum to 1.0 (row stochasticity)
        a_hat = a_hat / a_hat.sum(dim=-1, keepdim=True)

        # 4. Optional discard thresholding for noise reduction
        if discard_ratio > 0.0:
            flat = a_hat.view(B * N, N)
            k = int(N * discard_ratio)
            if k > 0:
                top_thresh, _ = torch.kthvalue(flat, k=k, dim=-1, keepdim=True)
                flat_masked = torch.where(flat < top_thresh, torch.zeros_like(flat), flat)
                a_hat = flat_masked.view(B, N, N)
                a_hat = a_hat / (a_hat.sum(dim=-1, keepdim=True) + 1e-8)

        # 5. Recursive matrix multiplication
        rollout = torch.bmm(a_hat, rollout)

    return rollout


def extract_cls_rollout(
    rollout_matrix: torch.Tensor,
    patch_size: int = 8,
    image_size: int = 96,
    upsample: bool = True,
) -> torch.Tensor:
    r"""Extract the [CLS] token's cumulative attention to all spatial patches.

    Args:
        rollout_matrix: Rollout matrix of shape :math:`(B, N, N)` or :math:`(N, N)`.
        patch_size: Spatial patch dimension :math:`P`.
        image_size: Original image height/width :math:`H=W`.
        upsample: If True, interpolates the heatmap to :math:`(image\_size, image\_size)`.

    Returns:
        Heatmap tensor of shape :math:`(B, H, W)` or :math:`(H, W)` if :math:`B=1`.
    """
    if rollout_matrix.ndim == 2:
        rollout_matrix = rollout_matrix.unsqueeze(0)

    B, N, _ = rollout_matrix.shape
    num_h = image_size // patch_size
    num_w = image_size // patch_size
    expected_patches = num_h * num_w

    if N != expected_patches + 1:
        raise ValueError(
            f"Rollout matrix size N={N} does not match expected tokens {expected_patches} + 1 for image_size={image_size} and patch_size={patch_size}"
        )

    # First row is [CLS] token attention to all tokens. Exclude [CLS] self-attention at idx 0.
    cls_to_patches = rollout_matrix[:, 0, 1:]  # (B, num_patches)
    cls_grid = cls_to_patches.view(B, 1, num_h, num_w)

    if not upsample:
        return cls_grid.squeeze(1).squeeze(0) if B == 1 else cls_grid.squeeze(1)

    upsampled = F.interpolate(cls_grid, size=(image_size, image_size), mode="bicubic", align_corners=False)
    upsampled = torch.clamp(upsampled, min=0.0)

    # Min-max normalize per sample to [0.0, 1.0]
    mins = upsampled.view(B, -1).min(dim=-1)[0].view(B, 1, 1, 1)
    maxs = upsampled.view(B, -1).max(dim=-1)[0].view(B, 1, 1, 1)
    norm = (upsampled - mins) / (maxs - mins + 1e-8)

    return norm.squeeze(1).squeeze(0) if B == 1 else norm.squeeze(1)


def plot_rollout_comparison(
    image: torch.Tensor,
    raw_attention: torch.Tensor,
    rollout_attention: torch.Tensor,
    patch_size: int = 8,
    save_path: str | None = None,
) -> None:
    r"""Plot side-by-side comparison between Raw Attention and Attention Rollout.

    Args:
        image: Single image tensor of shape :math:`(1, 3, H, W)` or :math:`(3, H, W)`.
        raw_attention: Final layer raw attention tensor :math:`(1, H, N, N)` or :math:`(H, N, N)`.
        rollout_attention: Cumulative rollout matrix :math:`(1, N, N)` or :math:`(N, N)`.
        patch_size: Spatial patch dimension :math:`P`.
        save_path: Destination file path or None to show interactively.
    """
    if image.ndim == 4:
        image = image.squeeze(0)

    _, H, W = image.shape
    disp_img = denormalize(image) if (image.min() < 0.0 or image.max() > 1.0) else image
    img_np = disp_img.permute(1, 2, 0).detach().cpu().numpy()

    # 1. Raw Attention from last layer (averaged across heads)
    if raw_attention.ndim == 4:
        raw_attention = raw_attention.squeeze(0)
    raw_cls = raw_attention[:, 0, 1:].mean(dim=0, keepdim=True)  # (1, num_patches)
    num_h = H // patch_size
    num_w = W // patch_size
    raw_grid = raw_cls.view(1, 1, num_h, num_w)
    raw_upsampled = F.interpolate(raw_grid, size=(H, W), mode="bicubic", align_corners=False)
    raw_map = torch.clamp(raw_upsampled[0, 0], min=0.0).detach().cpu().numpy()
    raw_norm = (raw_map - raw_map.min()) / (raw_map.max() - raw_map.min() + 1e-8)

    # 2. Rollout Attention
    rollout_map = extract_cls_rollout(rollout_attention, patch_size=patch_size, image_size=H, upsample=True)
    rollout_norm = rollout_map.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 4, figsize=(15, 4))

    # Panel 1: Original Image
    axes[0].imshow(img_np)
    axes[0].set_title("Input Image", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: Raw Final Layer Attention
    axes[1].imshow(raw_norm, cmap="inferno")
    axes[1].set_title("Raw Attention (Layer 8)\n[No Residuals Tracked]", fontsize=10, fontweight="bold")
    axes[1].axis("off")

    # Panel 3: Attention Rollout Heatmap
    axes[2].imshow(rollout_norm, cmap="inferno")
    axes[2].set_title("Attention Rollout\n[Abnar & Zuidema, 2020]", fontsize=10, fontweight="bold")
    axes[2].axis("off")

    # Panel 4: Rollout Overlaid on Original Image
    axes[3].imshow(img_np)
    axes[3].imshow(rollout_norm, cmap="inferno", alpha=0.55)
    axes[3].set_title("Rollout Saliency Overlay\n[Semantic Object Focus]", fontsize=10, fontweight="bold")
    axes[3].axis("off")

    plt.suptitle("Attention Interpretability: Raw Attention vs. Attention Rollout", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig.subplots_adjust(top=0.85)

    _display_and_close(fig, save_path)
