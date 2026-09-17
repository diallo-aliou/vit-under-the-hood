"""Attention Head Specialization and Systematic Ablation Studies.

Analyzes individual attention head behaviors:
1. Spatial receptive fields via Mean Attention Distance (Local vs. Global heads).
2. Mission-critical vs. redundant heads via Single-Head Ablation Surgery.
3. Multi-head compression capacity via Cumulative Greedy Pruning.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader

from src.models.vit import VisionTransformer
from src.utils.visualize import _display_and_close


def compute_patch_distance_matrix(
    image_size: int = 96,
    patch_size: int = 8,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    r"""Compute Euclidean physical distance matrix between all pairs of spatial patches.

    Args:
        image_size: Spatial resolution of square image :math:`(H = W)`.
        patch_size: Spatial resolution of square patch :math:`(P)`.
        device: Device to allocate distance matrix on.

    Returns:
        Pairwise distance matrix of shape :math:`(N, N)` in pixel units,
        where :math:`N = (image\_size / patch\_size)^2`.
    """
    grid_size = image_size // patch_size
    num_patches = grid_size * grid_size

    # Center coordinates of each patch in pixels
    coords = []
    for i in range(num_patches):
        row = i // grid_size
        col = i % grid_size
        y = row * patch_size + patch_size / 2.0
        x = col * patch_size + patch_size / 2.0
        coords.append((x, y))

    coords_tensor = torch.tensor(coords, dtype=torch.float32, device=device)  # (N, 2)

    # Compute Euclidean distance: sqrt((x_i - x_j)^2 + (y_i - y_j)^2)
    diff = coords_tensor.unsqueeze(0) - coords_tensor.unsqueeze(1)  # (N, N, 2)
    dist_matrix = torch.norm(diff, dim=-1)  # (N, N)

    return dist_matrix


def compute_mean_attention_distance(
    model: VisionTransformer,
    images: torch.Tensor,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    r"""Compute mean spatial attention distance for each head across all layers.

    Measures whether individual attention heads specialize in local patterns
    (small distance, short-range connections) or global semantic context
    (large distance, wide-receptive fields).

    Args:
        model: ViT model instance with attention map extraction enabled.
        images: Batch of images of shape :math:`(B, 3, H, W)`.
        device: Device to run computation on.

    Returns:
        Tensor of shape :math:`(L, H)` representing average distance in pixels
        for each head :math:`h` at layer :math:`l`.
    """
    device = torch.device(device)
    model.to(device)
    model.eval()

    patch_size = model.embedding.patch_size
    image_size = images.shape[-1]

    # (N, N) pairwise distance in pixels
    dist_matrix = compute_patch_distance_matrix(image_size, patch_size, device=device)

    images_dev = images.to(device)
    with torch.no_grad():
        _, all_attentions = model(images_dev, return_all_attentions=True)

    # all_attentions is list of L tensors of shape (B, num_heads, N+1, N+1)
    depth = len(all_attentions)
    num_heads = all_attentions[0].shape[1]
    head_distances = torch.zeros(depth, num_heads, device=device)

    for l_idx, layer_attn in enumerate(all_attentions):
        # Exclude [CLS] token (idx 0) to evaluate inter-patch spatial distance
        patch_attn = layer_attn[:, :, 1:, 1:]  # (B, num_heads, N, N)

        # Normalize rows so each patch's attention distribution sums to 1.0
        row_sums = patch_attn.sum(dim=-1, keepdim=True) + 1e-8
        norm_attn = patch_attn / row_sums

        # Weighted distance: sum_j (attn(i, j) * dist(i, j)) averaged over all patches i and batch B
        # (B, num_heads, N, N) * (1, 1, N, N) -> (B, num_heads, N, N)
        weighted_dist = norm_attn * dist_matrix.unsqueeze(0).unsqueeze(0)
        # Average over source patches (dim=-2) and batch (dim=0), then sum over targets (dim=-1)
        mean_dist = weighted_dist.sum(dim=-1).mean(dim=[-1, 0])  # (num_heads,)

        head_distances[l_idx] = mean_dist

    return head_distances


def evaluate_model_accuracy(
    model: VisionTransformer,
    dataloader: DataLoader,
    device: torch.device | str = "cpu",
    head_masks: list[torch.Tensor | None] | None = None,
    max_batches: int | None = None,
) -> float:
    r"""Evaluate top-1 classification accuracy with optional head masks.

    Args:
        model: ViT model to evaluate.
        dataloader: Validation DataLoader.
        device: Device to run evaluation on.
        head_masks: Optional list of head masks per layer for ablation experiments.
        max_batches: Optional limit on evaluation batches for rapid testing.

    Returns:
        Top-1 classification accuracy in percentage :math:`[0.0, 100.0]`.
    """
    device = torch.device(device)
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for b_idx, (images, targets) in enumerate(dataloader):
            if max_batches is not None and b_idx >= max_batches:
                break
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images, head_masks=head_masks)
            preds = logits.argmax(dim=-1)

            correct += (preds == targets).sum().item()
            total += targets.size(0)

    return (correct / max(total, 1)) * 100.0


def evaluate_single_head_ablation(
    model: VisionTransformer,
    dataloader: DataLoader,
    device: torch.device | str = "cpu",
    max_batches: int | None = None,
    progress_callback: Callable[[int, int, float], None] | None = None,
) -> tuple[float, np.ndarray]:
    r"""Systematically ablate each attention head individually and measure accuracy drop.

    Args:
        model: ViT model instance.
        dataloader: Validation DataLoader.
        device: Target execution device.
        max_batches: Optional limit on batches per evaluation for speed.
        progress_callback: Optional progress callback `(step, total_steps, acc)`.

    Returns:
        Tuple of `(baseline_accuracy, accuracy_drop_matrix)` where matrix has
        shape :math:`(depth, num\_heads)`. Positive drop indicates a critical head.
    """
    device = torch.device(device)
    depth = model.depth
    num_heads = model.encoder.layers[0].attn.num_heads

    baseline_acc = evaluate_model_accuracy(model, dataloader, device=device, max_batches=max_batches)
    print(f"Baseline Accuracy (0 heads ablated): {baseline_acc:.2f}%")

    drop_matrix = np.zeros((depth, num_heads), dtype=np.float32)
    total_heads = depth * num_heads
    step = 0

    for layer_idx in range(depth):
        for h in range(num_heads):
            step += 1
            # Create mask for this head
            mask = torch.ones(num_heads, device=device)
            mask[h] = 0.0  # Zero out this specific head

            head_masks: list[torch.Tensor | None] = [None] * depth
            head_masks[layer_idx] = mask

            ablated_acc = evaluate_model_accuracy(
                model,
                dataloader,
                device=device,
                head_masks=head_masks,
                max_batches=max_batches,
            )
            drop = baseline_acc - ablated_acc
            drop_matrix[layer_idx, h] = drop

            if progress_callback is not None:
                progress_callback(step, total_heads, drop)

    return baseline_acc, drop_matrix


def prune_heads_cumulative(
    model: VisionTransformer,
    dataloader: DataLoader,
    drop_matrix: np.ndarray,
    prune_ratios: list[float] | None = None,
    device: torch.device | str = "cpu",
    max_batches: int | None = None,
) -> dict[float, float]:
    r"""Progressively prune the least important heads greedily and measure accuracy retention.

    Args:
        model: ViT model instance.
        dataloader: Validation DataLoader.
        drop_matrix: Matrix of shape :math:`(depth, num\_heads)` from single-head ablation.
        prune_ratios: List of pruning fractions (default: `[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]`).
        device: Device to run evaluation on.
        max_batches: Optional limit on evaluation batches.

    Returns:
        Dictionary mapping prune ratio to retained validation accuracy percentage.
    """
    if prune_ratios is None:
        prune_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    device = torch.device(device)
    depth, num_heads = drop_matrix.shape
    total_heads = depth * num_heads

    # Rank all (layer, head) pairs by impact (smallest drop = least important)
    ranked_heads: list[tuple[int, int, float]] = []
    for layer_idx in range(depth):
        for h in range(num_heads):
            ranked_heads.append((layer_idx, h, float(drop_matrix[layer_idx, h])))

    # Sort ascending: heads with lowest drop (or negative drop / redundant) first
    ranked_heads.sort(key=lambda item: item[2])

    results: dict[float, float] = {}

    for ratio in prune_ratios:
        num_to_prune = int(math.ceil(total_heads * ratio))
        heads_to_prune = ranked_heads[:num_to_prune]

        # Build head_masks
        layer_masks: list[torch.Tensor] = [torch.ones(num_heads, device=device) for _ in range(depth)]
        for layer_idx, h, _ in heads_to_prune:
            layer_masks[layer_idx][h] = 0.0

        head_masks_arg = [m if (m == 0.0).any() else None for m in layer_masks]

        acc = evaluate_model_accuracy(
            model,
            dataloader,
            device=device,
            head_masks=head_masks_arg,
            max_batches=max_batches,
        )
        results[ratio] = acc
        print(f"Pruned {num_to_prune}/{total_heads} heads ({ratio*100:.0f}%) -> Acc: {acc:.2f}%")

    return results


# =============================================================================
# Visualizations
# =============================================================================


def plot_attention_distance_heatmap(
    distances: np.ndarray | torch.Tensor,
    save_path: str | None = None,
) -> None:
    r"""Plot heatmap of mean attention distance across layers and heads.

    Args:
        distances: Array or Tensor of shape :math:`(depth, num\_heads)` in pixels.
        save_path: Destination path for figure, or None to display.
    """
    if isinstance(distances, torch.Tensor):
        dist_np = distances.detach().cpu().numpy()
    else:
        dist_np = np.asarray(distances)

    depth, num_heads = dist_np.shape
    layer_labels = [f"L{i+1}" for i in range(depth)]
    head_labels = [f"Head {j+1}" for j in range(num_heads)]

    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=130)
    sns.heatmap(
        dist_np,
        annot=True,
        fmt=".1f",
        cmap="mako",
        xticklabels=head_labels,
        yticklabels=layer_labels,
        cbar_kws={"label": "Mean Attention Distance (Pixels)"},
        ax=ax,
        linewidths=1.0,
        linecolor="white",
    )

    ax.set_title(
        "ViT Head Specialization: Mean Spatial Attention Distance\n[Dosovitskiy et al. 2020 Receptive Fields]",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Attention Heads", fontsize=10, fontweight="bold")
    ax.set_ylabel("Encoder Depth (Layers)", fontsize=10, fontweight="bold")

    plt.tight_layout()
    _display_and_close(fig, save_path)


def plot_ablation_matrix(
    drop_matrix: np.ndarray | torch.Tensor,
    save_path: str | None = None,
) -> None:
    r"""Plot heatmap of accuracy drop when individual heads are ablated.

    Args:
        drop_matrix: Array or Tensor of shape :math:`(depth, num\_heads)` in % points.
        save_path: Destination path for figure, or None to display.
    """
    if isinstance(drop_matrix, torch.Tensor):
        drops_np = drop_matrix.detach().cpu().numpy()
    else:
        drops_np = np.asarray(drop_matrix)

    depth, num_heads = drops_np.shape
    layer_labels = [f"L{i+1}" for i in range(depth)]
    head_labels = [f"Head {j+1}" for j in range(num_heads)]

    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=130)
    sns.heatmap(
        drops_np,
        annot=True,
        fmt="+.2f",
        cmap="coolwarm",
        center=0.0,
        xticklabels=head_labels,
        yticklabels=layer_labels,
        cbar_kws={"label": "Accuracy Drop ΔAcc (% points)"},
        ax=ax,
        linewidths=1.0,
        linecolor="white",
    )

    ax.set_title(
        "Systematic Head Ablation Surgery\n[Impact of Individual Heads on Classification Accuracy]",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Ablated Attention Head", fontsize=10, fontweight="bold")
    ax.set_ylabel("Encoder Depth (Layers)", fontsize=10, fontweight="bold")

    plt.tight_layout()
    _display_and_close(fig, save_path)


def plot_pruning_retention_curve(
    pruning_results: dict[float, float],
    baseline_acc: float | None = None,
    save_path: str | None = None,
) -> None:
    r"""Plot validation accuracy vs. fraction of pruned heads.

    Args:
        pruning_results: Dictionary mapping prune ratio (0.0 to 1.0) to accuracy.
        baseline_acc: Optional baseline unpruned accuracy reference line.
        save_path: Destination path for figure, or None to display.
    """
    ratios = sorted(pruning_results.keys())
    accuracies = [pruning_results[r] for r in ratios]
    percent_pruned = [r * 100 for r in ratios]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=130)
    ax.plot(percent_pruned, accuracies, marker="o", color="crimson", lw=2.2, label="Greedy Pruned ViT")

    if baseline_acc is not None:
        ax.axhline(
            baseline_acc,
            color="black",
            linestyle="--",
            alpha=0.7,
            label=f"Baseline ({baseline_acc:.1f}%)",
        )

    ax.set_title(
        "Multi-Head Compression: Accuracy vs. Pruned Attention Heads",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Attention Heads Pruned (%)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Validation Top-1 Accuracy (%)", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left")

    plt.tight_layout()
    _display_and_close(fig, save_path)
