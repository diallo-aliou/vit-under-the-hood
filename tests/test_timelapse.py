"""Unit tests for Attention Rollout Timelapse generation."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.experiments.timelapse import (
    create_progression_strip,
    discover_checkpoints,
    generate_attention_timelapse,
    render_timelapse_frame,
)
from src.models.vit import VisionTransformer


class TestTimelapse:
    """Test suite for timelapse frames, checkpoint discovery, and GIF generation."""

    def test_render_timelapse_frame(self) -> None:
        """Verify render_timelapse_frame returns a valid RGB image buffer."""
        image = torch.rand(3, 96, 96)
        heatmap = torch.rand(96, 96)

        frame = render_timelapse_frame(
            image=image,
            rollout_map=heatmap,
            epoch=5,
            total_epochs=50,
            val_acc=42.5,
        )

        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3
        assert frame.shape[2] == 3  # RGB channels
        assert frame.shape[0] > 100 and frame.shape[1] > 200
        assert frame.dtype == np.uint8

    def test_discover_checkpoints(self, tmp_path: Path) -> None:
        """Verify checkpoint files are detected and sorted in epoch order."""
        (tmp_path / "epoch_010.pth").touch()
        (tmp_path / "epoch_002.pth").touch()
        (tmp_path / "epoch_001.pth").touch()
        (tmp_path / "best_model.pth").touch()
        (tmp_path / "random.txt").touch()

        discovered = discover_checkpoints(tmp_path)
        epochs = [item[0] for item in discovered]

        assert epochs == [1, 2, 10]
        assert discovered[0][1].name == "epoch_001.pth"
        assert discovered[1][1].name == "epoch_002.pth"
        assert discovered[2][1].name == "epoch_010.pth"

    def test_discover_checkpoints_not_found(self) -> None:
        """Verify discover_checkpoints raises FileNotFoundError for non-existent path."""
        with pytest.raises(FileNotFoundError):
            discover_checkpoints("non_existent_directory_xyz123")

    def test_generate_timelapse_and_strip(self) -> None:
        """Verify end-to-end timelapse GIF and progression strip creation with mock model."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_dir = Path(tmp_dir) / "checkpoints"
            ckpt_dir.mkdir()

            # Create miniature ViT model for instant test execution
            model = VisionTransformer(
                image_size=32,
                patch_size=8,
                in_channels=3,
                num_classes=4,
                embed_dim=32,
                depth=2,
                num_heads=2,
            )

            # Save 2 mock checkpoints
            for ep in [1, 2]:
                ckpt = {
                    "epoch": ep,
                    "model_state_dict": model.state_dict(),
                    "val_acc": 25.0 * ep,
                }
                torch.save(ckpt, ckpt_dir / f"epoch_{ep:03d}.pth")

            sample_img = torch.rand(1, 3, 32, 32)
            gif_path = Path(tmp_dir) / "test_timelapse.gif"
            strip_path = Path(tmp_dir) / "test_strip.png"

            frames = generate_attention_timelapse(
                checkpoint_dir=ckpt_dir,
                image=sample_img,
                model=model,
                output_gif_path=gif_path,
                fps=2,
                device="cpu",
            )

            assert len(frames) == 2
            assert gif_path.exists()
            assert gif_path.stat().st_size > 0

            # Test progression strip
            out_strip = create_progression_strip(
                checkpoint_dir=ckpt_dir,
                image=sample_img,
                model=model,
                target_epochs=[1, 2],
                output_path=strip_path,
                device="cpu",
            )

            assert out_strip.exists()
            assert out_strip.stat().st_size > 0
