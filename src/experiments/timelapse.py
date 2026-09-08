"""Attention Timelapse generation across training checkpoints.

Visualizes the emergence of semantic spatial attention in Vision Transformer (ViT)
from random initialization (Epoch 1) to focused semantic object localization (Epoch 50).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.models.vit import VisionTransformer, vit_tiny
from src.utils.rollout import compute_attention_rollout, extract_cls_rollout
from src.utils.visualize import create_sample_image, denormalize


def render_timelapse_frame(
    image: torch.Tensor,
    rollout_map: torch.Tensor | np.ndarray,
    epoch: int,
    total_epochs: int = 50,
    val_acc: float | None = None,
    title_suffix: str = "",
) -> np.ndarray:
    r"""Render a single 3-panel visualization frame for the timelapse animation.

    Panels:
        1. Input Image
        2. Attention Rollout Heatmap
        3. Saliency Overlay (Alpha Blended)

    Args:
        image: Single image tensor of shape :math:`(3, H, W)` or :math:`(1, 3, H, W)`.
        rollout_map: Saliency heatmap of shape :math:`(H, W)` or :math:`(1, H, W)`.
        epoch: Current training epoch number.
        total_epochs: Total number of training epochs.
        val_acc: Optional validation accuracy percentage at this epoch.
        title_suffix: Optional additional label in header.

    Returns:
        RGB numpy array of shape :math:`(H_{frame}, W_{frame}, 3)` with dtype uint8.
    """
    if image.ndim == 4:
        image = image.squeeze(0)
    if isinstance(rollout_map, torch.Tensor):
        if rollout_map.ndim == 3:
            rollout_map = rollout_map.squeeze(0)
        rollout_np = rollout_map.detach().cpu().numpy()
    else:
        rollout_np = np.squeeze(rollout_map)

    # Denormalize image if standardized
    disp_img = denormalize(image) if (image.min() < 0.0 or image.max() > 1.0) else image
    img_np = disp_img.permute(1, 2, 0).detach().cpu().numpy()
    img_np = np.clip(img_np, 0.0, 1.0)

    # Normalize heatmap to [0, 1]
    denom = rollout_np.max() - rollout_np.min()
    norm_heat = (rollout_np - rollout_np.min()) / (denom + 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), dpi=120)

    # Panel 1: Original Image
    axes[0].imshow(img_np)
    axes[0].set_title("Input Sample (STL-10)", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: Attention Rollout Heatmap
    im_heat = axes[1].imshow(norm_heat, cmap="inferno")
    axes[1].set_title("Attention Rollout\n[Abnar & Zuidema, 2020]", fontsize=10, fontweight="bold")
    axes[1].axis("off")
    fig.colorbar(im_heat, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: Composite Overlay
    axes[2].imshow(img_np)
    axes[2].imshow(norm_heat, cmap="inferno", alpha=0.55)
    axes[2].set_title("Saliency Overlay\n[Emergent Semantic Focus]", fontsize=10, fontweight="bold")
    axes[2].axis("off")

    acc_str = f" | Val Acc: {val_acc:.2f}%" if val_acc is not None else ""
    plt.suptitle(
        f"ViT Attention Dynamics — Epoch {epoch:02d}/{total_epochs:02d}{acc_str}{title_suffix}",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    fig.subplots_adjust(top=0.86)

    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)

    return frame.copy()


def discover_checkpoints(checkpoint_dir: str | Path) -> list[tuple[int, Path]]:
    r"""Find and sort epoch snapshot checkpoints by epoch index.

    Args:
        checkpoint_dir: Directory containing checkpoint files.

    Returns:
        List of tuples `(epoch_number, file_path)` sorted ascending by epoch.
    """
    ckpt_path = Path(checkpoint_dir).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_path}")

    pattern = re.compile(r"epoch_(\d+)\.pth$")
    discovered: list[tuple[int, Path]] = []

    for file in ckpt_path.glob("epoch_*.pth"):
        match = pattern.search(file.name)
        if match:
            epoch = int(match.group(1))
            discovered.append((epoch, file))

    discovered.sort(key=lambda item: item[0])
    return discovered


def generate_attention_timelapse(
    checkpoint_dir: str | Path,
    image: torch.Tensor,
    model: VisionTransformer,
    output_gif_path: str | Path = "outputs/timelapse_attention.gif",
    fps: int = 5,
    device: str | torch.device = "cpu",
    head_fusion: str = "mean",
    discard_ratio: float = 0.0,
    max_frames: int | None = None,
) -> list[np.ndarray]:
    r"""Generate an animated GIF showing attention evolution across training snapshots.

    Args:
        checkpoint_dir: Directory containing `epoch_*.pth` files.
        image: Single input image tensor :math:`(1, 3, H, W)` or :math:`(3, H, W)`.
        model: ViT model instance matching checkpoint architecture.
        output_gif_path: Destination path for the output animated GIF.
        fps: Frames per second for the animation. Default: 5.
        device: Device to run evaluation on. Default: 'cpu'.
        head_fusion: Head aggregation strategy ('mean', 'max', 'min').
        discard_ratio: Ratio of smallest attention weights to discard.
        max_frames: Optional upper limit on number of frames to render.

    Returns:
        List of RGB video frames as numpy arrays.
    """
    checkpoints = discover_checkpoints(checkpoint_dir)
    if not checkpoints:
        raise FileNotFoundError(
            f"No epoch checkpoints (epoch_*.pth) found in {checkpoint_dir}. "
            "Ensure training completed with snapshot saving enabled."
        )

    if max_frames is not None and len(checkpoints) > max_frames:
        # Subsample evenly
        indices = np.linspace(0, len(checkpoints) - 1, max_frames, dtype=int)
        checkpoints = [checkpoints[i] for i in indices]

    if image.ndim == 3:
        image = image.unsqueeze(0)

    device = torch.device(device)
    model.to(device)
    model.eval()
    image_dev = image.to(device)

    total_epochs = checkpoints[-1][0]
    frames: list[np.ndarray] = []

    out_gif = Path(output_gif_path).resolve()
    out_gif.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating timelapse across {len(checkpoints)} snapshots -> {out_gif.name}...")

    patch_size = model.embedding.patch_size
    img_size = image.shape[-1]

    for epoch, ckpt_file in checkpoints:
        checkpoint = torch.load(ckpt_file, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)

        val_acc = checkpoint.get("val_acc", None)

        with torch.no_grad():
            _, all_attentions = model(image_dev, return_all_attentions=True)

        rollout = compute_attention_rollout(
            all_attentions,
            discard_ratio=discard_ratio,
            head_fusion=head_fusion,
        )
        heatmap = extract_cls_rollout(
            rollout,
            patch_size=patch_size,
            image_size=img_size,
            upsample=True,
        )

        frame = render_timelapse_frame(
            image=image[0],
            rollout_map=heatmap,
            epoch=epoch,
            total_epochs=total_epochs,
            val_acc=val_acc,
        )
        frames.append(frame)

    # Save animated GIF using imageio (duration in ms per frame)
    frame_duration = 1000.0 / max(fps, 1)
    imageio.mimsave(str(out_gif), frames, duration=frame_duration, loop=0)
    print(f"Saved animated timelapse GIF ({len(frames)} frames, {fps} fps) to {out_gif}")

    return frames


def create_progression_strip(
    checkpoint_dir: str | Path,
    image: torch.Tensor,
    model: VisionTransformer,
    target_epochs: list[int] | None = None,
    output_path: str | Path = "outputs/timelapse_progression.png",
    device: str | torch.device = "cpu",
    head_fusion: str = "mean",
) -> Path:
    r"""Generate a static horizontal progression strip showing key training milestones.

    Args:
        checkpoint_dir: Directory containing `epoch_*.pth` checkpoints.
        image: Single image tensor :math:`(1, 3, H, W)` or :math:`(3, H, W)`.
        model: ViT model instance.
        target_epochs: Milestones to display (default: [1, 5, 10, 25, 50]).
        output_path: Destination path for the saved composite PNG.
        device: Device to run evaluation on.
        head_fusion: Attention head fusion strategy.

    Returns:
        Resolved path to the saved PNG strip.
    """
    if target_epochs is None:
        target_epochs = [1, 5, 10, 25, 50]

    checkpoints = discover_checkpoints(checkpoint_dir)
    available_dict = {epoch: path for epoch, path in checkpoints}

    # Select closest available epoch for each target
    selected_epochs: list[int] = []
    selected_files: list[Path] = []
    for tgt in target_epochs:
        if tgt in available_dict:
            selected_epochs.append(tgt)
            selected_files.append(available_dict[tgt])
        elif checkpoints:
            closest = min(checkpoints, key=lambda x: abs(x[0] - tgt))
            if closest[0] not in selected_epochs:
                selected_epochs.append(closest[0])
                selected_files.append(closest[1])

    if not selected_files:
        raise FileNotFoundError(f"No valid checkpoints found in {checkpoint_dir}")

    if image.ndim == 3:
        image = image.unsqueeze(0)

    device = torch.device(device)
    model.to(device)
    model.eval()
    image_dev = image.to(device)

    disp_img = denormalize(image[0]) if (image.min() < 0.0 or image.max() > 1.0) else image[0]
    img_np = np.clip(disp_img.permute(1, 2, 0).detach().cpu().numpy(), 0.0, 1.0)

    patch_size = model.embedding.patch_size
    img_size = image.shape[-1]

    num_cols = len(selected_files) + 1
    fig, axes = plt.subplots(1, num_cols, figsize=(3 * num_cols, 3.2), dpi=130)

    # Col 0: Input Image
    axes[0].imshow(img_np)
    axes[0].set_title("Input Image", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    for col_idx, (epoch, ckpt_file) in enumerate(zip(selected_epochs, selected_files, strict=False), start=1):
        checkpoint = torch.load(ckpt_file, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        val_acc = checkpoint.get("val_acc", None)

        with torch.no_grad():
            _, all_attentions = model(image_dev, return_all_attentions=True)

        rollout = compute_attention_rollout(all_attentions, head_fusion=head_fusion)
        heatmap = extract_cls_rollout(rollout, patch_size=patch_size, image_size=img_size, upsample=True)
        norm_heat = heatmap.detach().cpu().numpy()

        axes[col_idx].imshow(img_np)
        axes[col_idx].imshow(norm_heat, cmap="inferno", alpha=0.55)

        title = f"Epoch {epoch:02d}"
        if val_acc is not None:
            title += f"\n({val_acc:.1f}%)"
        axes[col_idx].set_title(title, fontsize=10, fontweight="bold")
        axes[col_idx].axis("off")

    plt.suptitle(
        "Emergence of Visual Attention Saliency Across Training (Vision Transformer on STL-10)",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    fig.subplots_adjust(top=0.82)

    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_file), bbox_inches="tight", dpi=150)
    plt.close(fig)

    print(f"Saved progression strip to {out_file}")
    return out_file


def main() -> None:
    """CLI entrypoint for attention timelapse generation."""
    parser = argparse.ArgumentParser(description="Generate ViT Attention Rollout Timelapse GIF.")
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="outputs/checkpoints",
        help="Directory containing epoch_*.pth checkpoint snapshots.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/timelapse_attention.gif",
        help="Destination path for output GIF.",
    )
    parser.add_argument(
        "--strip-output",
        type=str,
        default="outputs/timelapse_progression.png",
        help="Destination path for output progression strip PNG.",
    )
    parser.add_argument("--fps", type=int, default=5, help="Animation frame rate (fps).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fusion", type=str, default="mean", choices=["mean", "max", "min"])
    parser.add_argument("--discard", type=float, default=0.0, help="Attention discard ratio (0.0 to 1.0).")
    args = parser.parse_args()

    model = vit_tiny(num_classes=10)
    sample_img = create_sample_image(size=96).unsqueeze(0)

    ckpt_dir = Path(args.checkpoints)
    if not ckpt_dir.exists() or not list(ckpt_dir.glob("epoch_*.pth")):
        print(f"Warning: No epoch checkpoints found in {ckpt_dir}.")
        return

    generate_attention_timelapse(
        checkpoint_dir=ckpt_dir,
        image=sample_img,
        model=model,
        output_gif_path=args.output,
        fps=args.fps,
        device=args.device,
        head_fusion=args.fusion,
        discard_ratio=args.discard,
    )

    create_progression_strip(
        checkpoint_dir=ckpt_dir,
        image=sample_img,
        model=model,
        output_path=args.strip_output,
        device=args.device,
        head_fusion=args.fusion,
    )


if __name__ == "__main__":
    main()
