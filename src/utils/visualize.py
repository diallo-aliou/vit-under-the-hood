"""Visualization utilities for Vision Transformer patches, predictions, and attention dynamics."""

from collections.abc import Callable
from pathlib import Path

import einops
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageDraw


def denormalize(
    tensor: torch.Tensor,
    mean: tuple[float, ...] = (0.485, 0.456, 0.406),
    std: tuple[float, ...] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    r"""Revert channel-wise normalization back to [0, 1] RGB range.

    Args:
        tensor: Normalized tensor of shape (3, H, W) or (B, 3, H, W).
        mean: Channel means used during normalization.
        std: Channel standard deviations used during normalization.

    Returns:
        Tensor in [0.0, 1.0] range with same shape as input.
    """
    device = tensor.device
    dtype = tensor.dtype
    mean_t = torch.tensor(mean, device=device, dtype=dtype).view(-1, 1, 1)
    std_t = torch.tensor(std, device=device, dtype=dtype).view(-1, 1, 1)

    if tensor.ndim == 4:
        mean_t = mean_t.unsqueeze(0)
        std_t = std_t.unsqueeze(0)

    return torch.clamp(tensor * std_t + mean_t, 0.0, 1.0)


def create_sample_image(image_size: int = 96) -> torch.Tensor:
    r"""Create a synthetic landscape scene tensor of shape (1, 3, H, W)."""
    img = Image.new("RGB", (image_size, image_size), color=(135, 206, 235))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, int(image_size * 0.625), image_size, image_size], fill=(34, 139, 34))
    draw.polygon(
        [
            (image_size // 2, int(image_size * 0.25)),
            (int(image_size * 0.15), int(image_size * 0.75)),
            (int(image_size * 0.85), int(image_size * 0.75)),
        ],
        fill=(139, 69, 19),
    )
    draw.ellipse(
        [
            int(image_size * 0.7),
            int(image_size * 0.1),
            int(image_size * 0.9),
            int(image_size * 0.3),
        ],
        fill=(255, 215, 0),
    )

    transform = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])
    return transform(img).unsqueeze(0)


def load_image(image_path: str, image_size: int = 96) -> torch.Tensor:
    r"""Load an image from disk and resize to tensor of shape (1, 3, H, W)."""
    img = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])
    return transform(img).unsqueeze(0)



def _display_and_close(fig: plt.Figure, save_path: str | None = None) -> None:
    """Helper to save and/or display a figure across Colab, Jupyter, CLI, and test environments."""
    if save_path:
        out_path = Path(save_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    is_displayed = False
    try:
        from IPython import get_ipython
        from IPython.display import display

        if get_ipython() is not None:
            display(fig)
            is_displayed = True
    except Exception:
        pass

    if not is_displayed and matplotlib.get_backend().lower() != "agg":
        plt.show()

    plt.close(fig)


def plot_patch_grid(
    image: torch.Tensor,
    patch_size: int = 8,
    save_path: str | None = "outputs/patch_grid.png",
) -> None:
    r"""Plot the image side-by-side with its separated spatial patches.

    Args:
        image: Image tensor of shape (1, C, H, W) or (C, H, W).
        patch_size: Square patch spatial dimension P.
        save_path: Destination file path, or None to display interactively.
    """
    if image.ndim == 4:
        image = image.squeeze(0)

    C, H, W = image.shape
    num_h = H // patch_size
    num_w = W // patch_size
    num_patches = num_h * num_w

    patches = einops.rearrange(
        image.unsqueeze(0),
        "b c (h p1) (w p2) -> (b h w) c p1 p2",
        p1=patch_size,
        p2=patch_size,
    )

    fig, (ax_orig, ax_grid) = plt.subplots(1, 2, figsize=(11, 5))

    # 1. Original image with patch boundary grid overlay
    img_np = image.permute(1, 2, 0).detach().cpu().numpy()
    if img_np.min() < 0.0 or img_np.max() > 1.0:
        img_np = denormalize(image).permute(1, 2, 0).detach().cpu().numpy()

    ax_orig.imshow(img_np)
    ax_orig.set_title(
        f"Input Image ({H}x{W}) with {patch_size}x{patch_size} Grid",
        fontsize=11,
        fontweight="bold",
    )
    for y in range(0, H, patch_size):
        ax_orig.axhline(y - 0.5, color="white", linestyle="--", linewidth=0.6)
    for x in range(0, W, patch_size):
        ax_orig.axvline(x - 0.5, color="white", linestyle="--", linewidth=0.6)
    ax_orig.axis("off")

    # 2. Separated patches layout with spacing
    canvas = torch.zeros(C, H + num_h, W + num_w)
    idx = 0
    for i in range(num_h):
        for j in range(num_w):
            y = i * (patch_size + 1)
            x = j * (patch_size + 1)
            canvas[:, y : y + patch_size, x : x + patch_size] = patches[idx]
            idx += 1

    canvas_np = canvas.permute(1, 2, 0).detach().cpu().numpy()
    if canvas_np.min() < 0.0 or canvas_np.max() > 1.0:
        canvas_np = denormalize(canvas).permute(1, 2, 0).detach().cpu().numpy()

    ax_grid.imshow(canvas_np)
    ax_grid.set_title(
        f"Separated Patches (N = {num_patches} tokens)",
        fontsize=11,
        fontweight="bold",
    )
    ax_grid.axis("off")

    plt.tight_layout()
    _display_and_close(fig, save_path)


def plot_dataset_samples(
    images: torch.Tensor,
    labels: torch.Tensor,
    class_names: list[str],
    num_samples: int = 10,
    nrows: int = 2,
    save_path: str | None = None,
) -> None:
    r"""Display a grid of real dataset samples with their class labels.

    Args:
        images: Batch of image tensors (B, 3, H, W).
        labels: Tensor of integer class labels (B,).
        class_names: Ordered list of human-readable class names.
        num_samples: Number of images to render.
        nrows: Number of rows in the subplot grid.
        save_path: Destination path or None to show interactively.
    """
    num_samples = min(num_samples, len(images))
    ncols = int(np.ceil(num_samples / nrows))

    if images.min() < 0.0 or images.max() > 1.0:
        disp_images = denormalize(images[:num_samples])
    else:
        disp_images = images[:num_samples]

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.4, nrows * 2.4))
    axes = np.array(axes).reshape(-1)

    for i in range(num_samples):
        img_np = disp_images[i].permute(1, 2, 0).detach().cpu().numpy()
        lbl = int(labels[i].item()) if hasattr(labels[i], "item") else int(labels[i])
        class_name = class_names[lbl] if lbl < len(class_names) else f"Class {lbl}"

        axes[i].imshow(img_np)
        axes[i].set_title(f"{class_name}\n(class {lbl})", fontsize=10, fontweight="bold")
        axes[i].axis("off")

    for j in range(num_samples, len(axes)):
        axes[j].axis("off")

    plt.suptitle("STL-10 Dataset Samples (96x96 RGB)", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()

    _display_and_close(fig, save_path)


def plot_augmentation_examples(
    image: torch.Tensor,
    transform: Callable,
    num_examples: int = 4,
    save_path: str | None = None,
) -> None:
    r"""Display an original image alongside stochastic augmentations.

    Args:
        image: Single image tensor of shape (1, 3, H, W) or (3, H, W).
        transform: Callable torchvision transform pipeline.
        num_examples: Number of augmented variations to generate.
        save_path: Destination path or None to show interactively.
    """
    if image.ndim == 4:
        image = image.squeeze(0)

    img_disp = denormalize(image) if (image.min() < 0.0 or image.max() > 1.0) else image
    pil_img = T.ToPILImage()(img_disp.detach().cpu())

    fig, axes = plt.subplots(1, num_examples + 1, figsize=((num_examples + 1) * 2.5, 3.0))

    axes[0].imshow(np.array(pil_img))
    axes[0].set_title("Original Image", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    for i in range(num_examples):
        aug_item = transform(pil_img)
        if isinstance(aug_item, torch.Tensor):
            if aug_item.min() < 0.0 or aug_item.max() > 1.0:
                aug_item = denormalize(aug_item)
            aug_np = aug_item.permute(1, 2, 0).detach().cpu().numpy()
        else:
            aug_np = np.array(aug_item)

        axes[i + 1].imshow(aug_np)
        axes[i + 1].set_title(f"Augmentation #{i + 1}", fontsize=10, fontweight="bold")
        axes[i + 1].axis("off")

    plt.suptitle("Data Augmentation Dynamics for ViT Regularization", fontsize=12, fontweight="bold")
    plt.tight_layout()

    _display_and_close(fig, save_path)


def plot_prediction_grid(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    class_names: list[str],
    device: str | torch.device = "cpu",
    num_samples: int = 8,
    save_path: str | None = None,
) -> None:
    r"""Plot a grid of test predictions with green/red status indicators.

    Args:
        model: Trained Vision Transformer model.
        images: Batch of image tensors (B, 3, H, W).
        labels: Ground truth labels (B,).
        class_names: List of class names.
        device: Device where inference is computed.
        num_samples: Number of samples to plot (default 8).
        save_path: Destination path or None to show interactively.
    """
    model.eval()
    num_samples = min(num_samples, len(images))
    sub_images = images[:num_samples].to(device)

    with torch.no_grad():
        logits = model(sub_images)
        probs = F.softmax(logits, dim=-1)
        pred_labels = torch.argmax(probs, dim=-1).cpu()
        confidences = probs[torch.arange(num_samples), pred_labels].cpu()

    disp_images = denormalize(images[:num_samples]) if (images.min() < 0.0 or images.max() > 1.0) else images[:num_samples]

    ncols = 4
    nrows = int(np.ceil(num_samples / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.0, nrows * 3.2))
    axes = np.array(axes).reshape(-1)

    for i in range(num_samples):
        img_np = disp_images[i].permute(1, 2, 0).detach().cpu().numpy()
        pred_idx = int(pred_labels[i].item())
        true_idx = int(labels[i].item()) if hasattr(labels[i], "item") else int(labels[i])
        conf = float(confidences[i].item()) * 100.0

        is_correct = pred_idx == true_idx
        title_color = "#1b8a24" if is_correct else "#c72828"
        pred_name = class_names[pred_idx] if pred_idx < len(class_names) else f"C{pred_idx}"
        true_name = class_names[true_idx] if true_idx < len(class_names) else f"C{true_idx}"

        axes[i].imshow(img_np)
        axes[i].set_title(
            f"Pred: {pred_name} ({conf:.1f}%)\nTrue: {true_name}",
            fontsize=10,
            fontweight="bold",
            color=title_color,
            pad=6,
        )
        for spine in axes[i].spines.values():
            spine.set_color(title_color)
            spine.set_linewidth(2.5)
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    for j in range(num_samples, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Model Evaluation on Test Samples (Green: Correct, Red: Error)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.subplots_adjust(top=0.88, hspace=0.45, wspace=0.20)

    _display_and_close(fig, save_path)


def plot_prediction_topk(
    image: torch.Tensor,
    true_label: int,
    probs: torch.Tensor,
    class_names: list[str],
    top_k: int = 5,
    save_path: str | None = None,
) -> None:
    r"""Plot an image alongside a horizontal bar chart of its Top-k class probabilities.

    Args:
        image: Single image tensor (1, 3, H, W) or (3, H, W).
        true_label: Ground truth integer class label.
        probs: 1D probability distribution tensor (C,).
        class_names: List of class names.
        top_k: Number of highest probabilities to plot.
        save_path: Destination path or None to show interactively.
    """
    if image.ndim == 4:
        image = image.squeeze(0)

    disp_img = denormalize(image) if (image.min() < 0.0 or image.max() > 1.0) else image
    img_np = disp_img.permute(1, 2, 0).detach().cpu().numpy()

    top_probs, top_indices = torch.topk(probs.detach().cpu(), k=top_k)
    top_probs = top_probs.numpy()[::-1]
    top_indices = top_indices.numpy()[::-1]
    top_names = [class_names[idx] if idx < len(class_names) else f"Class {idx}" for idx in top_indices]

    fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(9, 4), gridspec_kw={"width_ratios": [1, 1.4]})

    true_name = class_names[true_label] if true_label < len(class_names) else f"Class {true_label}"
    ax_img.imshow(img_np)
    ax_img.set_title(f"Ground Truth: {true_name}", fontsize=11, fontweight="bold")
    ax_img.axis("off")

    colors = ["#1b8a24" if idx == true_label else "#4285f4" for idx in top_indices]
    bars = ax_bar.barh(range(top_k), top_probs * 100, color=colors, edgecolor="black", linewidth=0.8)
    ax_bar.set_yticks(range(top_k))
    ax_bar.set_yticklabels(top_names, fontsize=10, fontweight="bold")
    ax_bar.set_xlabel("Predicted Probability (%)", fontsize=10)
    ax_bar.set_xlim(0, 105)
    ax_bar.set_title(f"Top-{top_k} Confidence Distribution", fontsize=11, fontweight="bold")

    for bar, val in zip(bars, top_probs * 100, strict=False):
        ax_bar.text(val + 1.5, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    _display_and_close(fig, save_path)


def plot_attention_heads(
    image: torch.Tensor,
    attention_weights: torch.Tensor,
    layer_idx: int = -1,
    patch_size: int = 8,
    save_path: str | None = None,
) -> None:
    r"""Plot individual attention heads and their mean overlay on the original image.

    Args:
        image: Single image tensor of shape (1, 3, H, W) or (3, H, W).
        attention_weights: Attention tensor of shape (1, num_heads, N, N) or (num_heads, N, N).
        layer_idx: Transformer layer index for plot titling.
        patch_size: Spatial patch dimension P.
        save_path: Destination path or None to show interactively.
    """
    if image.ndim == 4:
        image = image.squeeze(0)
    if attention_weights.ndim == 4:
        attention_weights = attention_weights.squeeze(0)

    _, H, W = image.shape
    num_heads, _, _ = attention_weights.shape
    num_h = H // patch_size
    num_w = W // patch_size

    cls_attn = attention_weights[:, 0, 1:]
    cls_grid = cls_attn.view(num_heads, 1, num_h, num_w)

    upsampled = F.interpolate(cls_grid, size=(H, W), mode="bicubic", align_corners=False)
    upsampled = torch.clamp(upsampled, min=0.0)

    disp_img = denormalize(image) if (image.min() < 0.0 or image.max() > 1.0) else image
    img_np = disp_img.permute(1, 2, 0).detach().cpu().numpy()

    fig, axes = plt.subplots(1, num_heads + 2, figsize=((num_heads + 2) * 2.8, 3.2))

    axes[0].imshow(img_np)
    axes[0].set_title("Original Input", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    for h in range(num_heads):
        head_map = upsampled[h, 0].detach().cpu().numpy()
        head_norm = (head_map - head_map.min()) / (head_map.max() - head_map.min() + 1e-8)
        axes[h + 1].imshow(head_norm, cmap="inferno")
        axes[h + 1].set_title(f"Head #{h + 1} Attention", fontsize=10, fontweight="bold")
        axes[h + 1].axis("off")

    mean_map = upsampled.mean(dim=0, keepdim=True)[0, 0].detach().cpu().numpy()
    mean_norm = (mean_map - mean_map.min()) / (mean_map.max() - mean_map.min() + 1e-8)

    axes[-1].imshow(img_np)
    axes[-1].imshow(mean_norm, cmap="inferno", alpha=0.55)
    axes[-1].set_title("Mean Attention Overlay", fontsize=10, fontweight="bold")
    axes[-1].axis("off")

    layer_title = f"Layer {layer_idx + 1}" if layer_idx >= 0 else "Final Layer"
    plt.suptitle(
        f"Multi-Head Attention Specialization ({layer_title}) — [CLS] Token Routing",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()

    _display_and_close(fig, save_path)


def plot_attention_layer_progression(
    image: torch.Tensor,
    all_attentions: list[torch.Tensor],
    layers: list[int] | None = None,
    patch_size: int = 8,
    save_path: str | None = None,
) -> None:
    r"""Visualize how attention overlays mature from early to deep layers.

    Args:
        image: Single image tensor of shape (1, 3, H, W) or (3, H, W).
        all_attentions: List of attention tensors for all transformer layers.
        layers: Explicit list of layer indices to display. Defaults to 4 evenly spaced layers.
        patch_size: Spatial patch dimension P.
        save_path: Destination path or None to show interactively.
    """
    if image.ndim == 4:
        image = image.squeeze(0)

    total_layers = len(all_attentions)
    if layers is None:
        layers = [0, total_layers // 3, 2 * total_layers // 3, total_layers - 1]

    _, H, W = image.shape
    num_h = H // patch_size
    num_w = W // patch_size

    disp_img = denormalize(image) if (image.min() < 0.0 or image.max() > 1.0) else image
    img_np = disp_img.permute(1, 2, 0).detach().cpu().numpy()

    fig, axes = plt.subplots(1, len(layers) + 1, figsize=((len(layers) + 1) * 2.7, 3.2))

    axes[0].imshow(img_np)
    axes[0].set_title("Input Image", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    for col_idx, l_idx in enumerate(layers):
        attn = all_attentions[l_idx]
        if attn.ndim == 4:
            attn = attn.squeeze(0)

        cls_attn = attn[:, 0, 1:].mean(dim=0, keepdim=True)
        cls_grid = cls_attn.view(1, 1, num_h, num_w)
        upsampled = F.interpolate(cls_grid, size=(H, W), mode="bicubic", align_corners=False)
        layer_map = torch.clamp(upsampled[0, 0], min=0.0).detach().cpu().numpy()
        layer_norm = (layer_map - layer_map.min()) / (layer_map.max() - layer_map.min() + 1e-8)

        axes[col_idx + 1].imshow(img_np)
        axes[col_idx + 1].imshow(layer_norm, cmap="inferno", alpha=0.55)
        axes[col_idx + 1].set_title(f"Layer {l_idx + 1}\nAttention", fontsize=10, fontweight="bold")
        axes[col_idx + 1].axis("off")

    plt.suptitle("Attention Depth Progression — From Diffuse Context to Semantic Salience", fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout()

    _display_and_close(fig, save_path)
