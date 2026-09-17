"""Unit tests for the Gradio app backend callbacks."""

import numpy as np
import pytest
import torch

from src.app_backend import (
    _overlay_heatmap,
    _pil_to_tensor,
    _tensor_to_display,
    fig_to_numpy,
    get_model_summary,
    load_scratch_model,
    predict_per_head,
    predict_with_rollout,
    run_domain_gap_demo,
    run_head_surgery,
)


@pytest.fixture(scope="module")
def _load_model():
    """Load the scratch model once for the entire test module."""
    load_scratch_model(checkpoint_path="outputs/checkpoints/best_model.pth")


@pytest.fixture()
def dummy_image() -> np.ndarray:
    """Create a random 96x96 RGB numpy image."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, (96, 96, 3), dtype=np.uint8)


class TestHelpers:
    """Tests for internal helper functions."""

    def test_fig_to_numpy_returns_rgb_array(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        ax.plot([0, 1], [0, 1])
        result = fig_to_numpy(fig)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 3
        assert result.shape[2] == 3  # RGB

    def test_overlay_heatmap_preserves_shape(self):
        img = np.zeros((96, 96, 3), dtype=np.uint8)
        heatmap = np.random.rand(96, 96).astype(np.float32)
        result = _overlay_heatmap(img, heatmap, alpha=0.5)
        assert result.shape == (96, 96, 3)
        assert result.dtype == np.uint8

    def test_overlay_heatmap_resizes_if_mismatch(self):
        img = np.zeros((96, 96, 3), dtype=np.uint8)
        heatmap = np.random.rand(12, 12).astype(np.float32)
        result = _overlay_heatmap(img, heatmap, alpha=0.5)
        assert result.shape == (96, 96, 3)

    def test_pil_to_tensor_shape(self):
        from PIL import Image

        from src.app_backend import _preprocess_scratch

        pil_img = Image.new("RGB", (200, 200), color=(128, 64, 32))
        tensor = _pil_to_tensor(pil_img, _preprocess_scratch)
        assert tensor.shape == (1, 3, 96, 96)

    def test_tensor_to_display_uint8(self):
        tensor = torch.randn(1, 3, 96, 96)
        result = _tensor_to_display(tensor)
        assert result.shape == (96, 96, 3)
        assert result.dtype == np.uint8


class TestPredictWithRollout:
    """Tests for the main rollout prediction callback."""

    def test_returns_correct_tuple_shape(self, _load_model, dummy_image):
        confs, rollout, raw, heatmap = predict_with_rollout(dummy_image)
        assert isinstance(confs, dict)
        assert len(confs) <= 5
        assert rollout.shape == (96, 96, 3)
        assert raw.shape == (96, 96, 3)
        assert heatmap.shape == (96, 96, 3)

    def test_none_image_returns_blanks(self, _load_model):
        confs, rollout, raw, heatmap = predict_with_rollout(None)
        assert confs == {}
        assert rollout.sum() == 0

    def test_different_fusion_strategies(self, _load_model, dummy_image):
        for fusion in ["mean", "max", "min"]:
            confs, _, _, _ = predict_with_rollout(
                dummy_image, discard_ratio=0.0, head_fusion=fusion,
            )
            assert isinstance(confs, dict)


class TestPerHead:
    """Tests for per-head attention visualization."""

    def test_returns_concatenated_image(self, _load_model, dummy_image):
        result = predict_per_head(dummy_image, layer_idx=0)
        assert result.shape == (96, 96 * 3, 3)
        assert result.dtype == np.uint8

    def test_different_layers(self, _load_model, dummy_image):
        for layer in range(8):
            result = predict_per_head(dummy_image, layer_idx=layer)
            assert result.shape[0] == 96


class TestHeadSurgery:
    """Tests for the head surgery callback."""

    def test_no_disabled_heads_returns_same_prediction(self, _load_model, dummy_image):
        before, after, ov_b, ov_a, impact = run_head_surgery(dummy_image, [])
        assert before == after  # No heads disabled = same prediction
        assert ov_b.shape == (96, 96, 3)
        assert ov_a.shape == (96, 96, 3)

    def test_with_disabled_heads_produces_impact(self, _load_model, dummy_image):
        _, _, _, _, impact = run_head_surgery(dummy_image, ["L1-H3", "L3-H1"])
        assert "2 head(s) disabled" in impact
        assert "Before" in impact


class TestDomainGap:
    """Tests for the domain gap (texture vs. shape) callback."""

    def test_returns_valid_outputs(self, _load_model, dummy_image):
        photo, photo_attn, sketch, sketch_attn, text = run_domain_gap_demo(dummy_image)
        assert photo.shape == (96, 96, 3)
        assert sketch.shape == (96, 96, 3)
        assert "Prediction" in text

    def test_sketch_differs_from_photo(self, _load_model, dummy_image):
        photo, _, sketch, _, _ = run_domain_gap_demo(dummy_image)
        # Sketch should look different from photo (not identical)
        assert not np.array_equal(photo, sketch)


class TestModelSummary:
    """Tests for the model architecture card."""

    def test_returns_markdown_string(self, _load_model):
        md = get_model_summary()
        assert isinstance(md, str)
        assert "ViT-Tiny" in md
        assert "Phase 7" in md
        assert "Head Specialization" in md

    def test_contains_parameter_count(self, _load_model):
        md = get_model_summary()
        assert "3." in md  # 3.xx M params
        assert "192" in md  # embed_dim
