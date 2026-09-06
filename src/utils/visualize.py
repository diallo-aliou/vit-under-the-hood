"""Visualization utilities for Vision Transformer patches."""

from pathlib import Path

import einops
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw


def create_sample_image(image_size: int = 96) -> torch.Tensor:
    r"""Create a simple landscape scene tensor of shape :math:`(1, 3, H, W)`."""
    img = Image.new("RGB", (image_size, image_size), color=(135, 206, 235))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, int(image_size * 0.625), image_size, image_size], fill=(34, 139, 34))
    draw.polygon(
        [(image_size // 2, int(image_size * 0.25)),
         (int(image_size * 0.15), int(image_size * 0.75)),
         (int(image_size * 0.85), int(image_size * 0.75))],
        fill=(139, 69, 19),
    )
    draw.ellipse(
        [int(image_size * 0.7), int(image_size * 0.1),
         int(image_size * 0.9), int(image_size * 0.3)],
        fill=(255, 215, 0),
    )

    transform = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])
    return transform(img).unsqueeze(0)


def load_image(image_path: str, image_size: int = 96) -> torch.Tensor:
    r"""Load an image from disk and resize to tensor of shape :math:`(1, 3, H, W)`."""
    img = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])
    return transform(img).unsqueeze(0)


def plot_patch_grid(
    image: torch.Tensor,
    patch_size: int = 8,
    save_path: str | None = "outputs/patch_grid.png",
) -> None:
    r"""Plot the image side-by-side with its separated spatial patches."""
    _, C, H, W = image.shape
    num_h = H // patch_size
    num_w = W // patch_size
    num_patches = num_h * num_w

    patches = einops.rearrange(
        image,
        "b c (h p1) (w p2) -> (b h w) c p1 p2",
        p1=patch_size,
        p2=patch_size,
    )

    fig, (ax_orig, ax_grid) = plt.subplots(1, 2, figsize=(12, 6))

    # 1. Original image with grid overlay
    img_np = image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    ax_orig.imshow(img_np)
    ax_orig.set_title(f"Input Image ({H}x{W}) with {patch_size}x{patch_size} Grid")
    for y in range(0, H, patch_size):
        ax_orig.axhline(y - 0.5, color="white", linestyle="--", linewidth=0.6)
    for x in range(0, W, patch_size):
        ax_orig.axvline(x - 0.5, color="white", linestyle="--", linewidth=0.6)
    ax_orig.axis("off")

    # 2. Separated patches layout
    canvas = torch.zeros(C, H + num_h, W + num_w)
    idx = 0
    for i in range(num_h):
        for j in range(num_w):
            y = i * (patch_size + 1)
            x = j * (patch_size + 1)
            canvas[:, y : y + patch_size, x : x + patch_size] = patches[idx]
            idx += 1

    ax_grid.imshow(canvas.permute(1, 2, 0).detach().cpu().numpy())
    ax_grid.set_title(f"Extracted Patches (N = {num_patches} tokens)")
    ax_grid.axis("off")

    plt.tight_layout()
    if save_path:
        out_path = Path(save_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
        print(f"Patch grid saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    sample = create_sample_image(image_size=96)
    plot_patch_grid(sample, patch_size=8, save_path="outputs/patch_grid.png")
