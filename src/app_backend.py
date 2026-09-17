"""Backend callback functions for the Gradio interactive ViT explorer.

All functions accept numpy/PIL inputs and return numpy arrays or strings
suitable for direct use as Gradio component outputs.
Models are loaded once at startup and cached in module-level singletons.
"""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

matplotlib.use("Agg")  # Non-interactive backend for server rendering


# ---------------------------------------------------------------------------
# Module-level singletons (loaded once, reused across all Gradio callbacks)
# ---------------------------------------------------------------------------
_model_scratch: torch.nn.Module | None = None
_model_transfer: torch.nn.Module | None = None
_device: torch.device = torch.device("cpu")

STL10_CLASSES: tuple[str, ...] = (
    "airplane", "bird", "car", "cat", "deer",
    "dog", "horse", "monkey", "ship", "truck",
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Pre-processing for from-scratch ViT-Tiny (96x96, ImageNet normalization)
_preprocess_scratch = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Pre-processing for ImageNet ViT-B/16 (224x224)
_preprocess_transfer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fig_to_numpy(fig: plt.Figure) -> np.ndarray:
    """Convert a matplotlib Figure to an RGB numpy array without saving to disk."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    return np.array(img)


def _pil_to_tensor(image: Image.Image | np.ndarray, transform: transforms.Compose) -> torch.Tensor:
    """Convert a PIL image or numpy array to a preprocessed 4D tensor."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    elif not isinstance(image, Image.Image):
        image = Image.fromarray(np.array(image).astype(np.uint8)).convert("RGB")
    tensor = transform(image).unsqueeze(0)  # (1, 3, H, W)
    return tensor.to(_device)


def _tensor_to_display(tensor: torch.Tensor) -> np.ndarray:
    """Convert a (3, H, W) or (1, 3, H, W) tensor to a displayable uint8 numpy array."""
    from src.utils.visualize import denormalize

    if tensor.ndim == 4:
        tensor = tensor[0]
    img = denormalize(tensor).cpu()
    img_np = (img.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    return img_np


def _overlay_heatmap(
    image_np: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.55,
    colormap: str = "inferno",
) -> np.ndarray:
    """Overlay a [0, 1] heatmap on an RGB image using a matplotlib colormap."""
    cmap = plt.get_cmap(colormap)
    heatmap_colored = cmap(heatmap)[..., :3]  # (H, W, 3) float64
    heatmap_rgb = (heatmap_colored * 255).astype(np.uint8)

    # Resize heatmap to match image if needed
    if heatmap_rgb.shape[:2] != image_np.shape[:2]:
        heatmap_pil = Image.fromarray(heatmap_rgb).resize(
            (image_np.shape[1], image_np.shape[0]), Image.BICUBIC
        )
        heatmap_rgb = np.array(heatmap_pil)

    blended = (
        (1 - alpha) * image_np.astype(np.float32)
        + alpha * heatmap_rgb.astype(np.float32)
    ).clip(0, 255).astype(np.uint8)
    return blended


# ---------------------------------------------------------------------------
# 1. Model Loading
# ---------------------------------------------------------------------------
def load_scratch_model(
    checkpoint_path: str | Path = "outputs/checkpoints/best_model.pth",
) -> tuple[torch.nn.Module, str]:
    """Load (or reload) the from-scratch ViT-Tiny model. Returns (model, status_message)."""
    global _model_scratch, _device

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from src.models.vit import vit_tiny

    model = vit_tiny(num_classes=10)
    ckpt_path = Path(checkpoint_path)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=_device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)
        val_acc = ckpt.get("val_acc", None)
        acc_str = f" (Val Acc: {val_acc:.1f}%)" if val_acc is not None else ""
        status = f"✅ Loaded checkpoint `{ckpt_path.name}`{acc_str} on **{_device}**"
    else:
        status = (
            f"⚠️ No checkpoint found at `{ckpt_path}`. "
            "Using randomly initialized weights (results will be meaningless)."
        )

    model.to(_device).eval()
    _model_scratch = model

    total, trainable = model.get_num_params()
    status += f"\n\n**ViT-Tiny**: {total:,} params ({total / 1e6:.2f}M)"
    return model, status


def load_transfer_model() -> tuple[torch.nn.Module, str]:
    """Lazy-load the ImageNet pre-trained ViT-B/16. Called on first use of Tab 3."""
    global _model_transfer

    if _model_transfer is not None:
        return _model_transfer, "✅ ImageNet ViT-B/16 already loaded."

    from src.experiments.transfer import build_transfer_vit

    model = build_transfer_vit(
        num_classes=10, pretrained=True, freeze_backbone=True, device=_device,
    )
    model.eval()
    _model_transfer = model
    return model, "✅ ImageNet ViT-B/16 loaded (~86M params)"


# ---------------------------------------------------------------------------
# 2. Tab 1: Attention Rollout Explorer
# ---------------------------------------------------------------------------
def predict_with_rollout(
    image: Image.Image | np.ndarray | None,
    discard_ratio: float = 0.0,
    head_fusion: str = "mean",
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """Run inference and attention rollout on a single image.

    Returns:
        (top5_confidences, rollout_overlay, raw_attn_overlay, heatmap_only)
    """
    if image is None or _model_scratch is None:
        blank = np.zeros((96, 96, 3), dtype=np.uint8)
        return {}, blank, blank, blank

    from src.utils.rollout import compute_attention_rollout, extract_cls_rollout

    img_tensor = _pil_to_tensor(image, _preprocess_scratch)

    with torch.no_grad():
        logits, all_attns = _model_scratch(img_tensor, return_all_attentions=True)

    # Top-5 predictions
    probs = torch.softmax(logits[0], dim=0)
    top5_vals, top5_idx = probs.topk(5)
    confidences = {
        STL10_CLASSES[i.item()]: round(v.item(), 4) for v, i in zip(top5_vals, top5_idx, strict=True)
    }

    # Attention Rollout
    rollout = compute_attention_rollout(
        all_attns, discard_ratio=discard_ratio, head_fusion=head_fusion,
    )
    cls_map = extract_cls_rollout(rollout, patch_size=8, image_size=96, upsample=True)
    heatmap_np = cls_map.cpu().numpy()

    # Raw last-layer attention (mean across heads)
    raw_attn = all_attns[-1][0]  # (H, N, N)
    raw_cls = raw_attn[:, 0, 1:].mean(dim=0)  # (144,)
    raw_grid = raw_cls.view(1, 1, 12, 12)
    raw_up = F.interpolate(raw_grid, size=(96, 96), mode="bicubic", align_corners=False)
    raw_map = torch.clamp(raw_up[0, 0], min=0.0).cpu().numpy()
    raw_norm = (raw_map - raw_map.min()) / (raw_map.max() - raw_map.min() + 1e-8)

    # Build display images
    img_display = _tensor_to_display(img_tensor)
    rollout_overlay = _overlay_heatmap(img_display, heatmap_np, alpha=0.55)
    raw_overlay = _overlay_heatmap(img_display, raw_norm, alpha=0.55)

    # Heatmap standalone
    cmap = plt.get_cmap("inferno")
    heatmap_colored = (cmap(heatmap_np)[..., :3] * 255).astype(np.uint8)

    return confidences, rollout_overlay, raw_overlay, heatmap_colored


def predict_per_head(
    image: Image.Image | np.ndarray | None,
    layer_idx: int = 0,
) -> np.ndarray:
    """Visualize the 3 individual attention heads for a given layer.

    Returns a single numpy image showing Head 1, Head 2, Head 3 side by side.
    """
    if image is None or _model_scratch is None:
        return np.zeros((96, 96 * 3, 3), dtype=np.uint8)

    img_tensor = _pil_to_tensor(image, _preprocess_scratch)

    with torch.no_grad():
        _, all_attns = _model_scratch(img_tensor, return_all_attentions=True)

    layer_attn = all_attns[layer_idx][0]  # (num_heads, N, N)
    num_heads = layer_attn.shape[0]
    img_display = _tensor_to_display(img_tensor)

    panels = []
    for h in range(num_heads):
        head_cls = layer_attn[h, 0, 1:]  # (144,)
        head_grid = head_cls.view(1, 1, 12, 12)
        head_up = F.interpolate(
            head_grid, size=(96, 96), mode="bicubic", align_corners=False,
        )
        head_map = torch.clamp(head_up[0, 0], min=0.0).cpu().numpy()
        head_norm = (head_map - head_map.min()) / (head_map.max() - head_map.min() + 1e-8)
        overlay = _overlay_heatmap(img_display, head_norm, alpha=0.55)
        panels.append(overlay)

    return np.concatenate(panels, axis=1)


# ---------------------------------------------------------------------------
# 3. Tab 2: Head Surgery Lab
# ---------------------------------------------------------------------------
def run_head_surgery(
    image: Image.Image | np.ndarray | None,
    disabled_heads_str: list[str],
) -> tuple[dict[str, float], dict[str, float], np.ndarray, np.ndarray, str]:
    """Run inference with and without disabled heads.

    Args:
        image: Input image.
        disabled_heads_str: List of strings like "L1-H2", "L3-H1", etc.

    Returns:
        (before_preds, after_preds, before_overlay, after_overlay, impact_text)
    """
    if image is None or _model_scratch is None:
        blank = np.zeros((96, 96, 3), dtype=np.uint8)
        return {}, {}, blank, blank, ""

    from src.utils.rollout import compute_attention_rollout, extract_cls_rollout

    img_tensor = _pil_to_tensor(image, _preprocess_scratch)
    depth = _model_scratch.depth
    num_heads = 3

    # Parse disabled heads
    disabled = set()
    for s in disabled_heads_str:
        s = s.strip()
        if not s:
            continue
        # Expected format: "L1-H2"
        parts = s.upper().replace("L", "").replace("H", "").split("-")
        if len(parts) == 2:
            try:
                layer_i, h = int(parts[0]) - 1, int(parts[1]) - 1
                disabled.add((layer_i, h))
            except ValueError:
                pass

    # --- BEFORE (no masks) ---
    with torch.no_grad():
        logits_before, attns_before = _model_scratch(
            img_tensor, return_all_attentions=True,
        )

    probs_before = torch.softmax(logits_before[0], dim=0)
    top5_b = probs_before.topk(5)
    preds_before = {
        STL10_CLASSES[i.item()]: round(v.item(), 4)
        for v, i in zip(top5_b.values, top5_b.indices, strict=True)
    }

    rollout_b = compute_attention_rollout(attns_before)
    map_b = extract_cls_rollout(rollout_b, patch_size=8, image_size=96).cpu().numpy()
    img_display = _tensor_to_display(img_tensor)
    overlay_before = _overlay_heatmap(img_display, map_b, alpha=0.55)

    # --- AFTER (with masks) ---
    head_masks: list[torch.Tensor | None] = [None] * depth
    if disabled:
        for layer_i in range(depth):
            mask = torch.ones(num_heads, device=_device)
            for dl, dh in disabled:
                if dl == layer_i:
                    mask[dh] = 0.0
            if (mask == 0.0).any():
                head_masks[layer_i] = mask

    with torch.no_grad():
        logits_after, attns_after = _model_scratch(
            img_tensor, return_all_attentions=True, head_masks=head_masks,
        )

    probs_after = torch.softmax(logits_after[0], dim=0)
    top5_a = probs_after.topk(5)
    preds_after = {
        STL10_CLASSES[i.item()]: round(v.item(), 4)
        for v, i in zip(top5_a.values, top5_a.indices, strict=True)
    }

    rollout_a = compute_attention_rollout(attns_after)
    map_a = extract_cls_rollout(rollout_a, patch_size=8, image_size=96).cpu().numpy()
    overlay_after = _overlay_heatmap(img_display, map_a, alpha=0.55)

    # Impact text
    pred_class_b = STL10_CLASSES[probs_before.argmax().item()]
    conf_b = probs_before.max().item() * 100
    pred_class_a = STL10_CLASSES[probs_after.argmax().item()]
    conf_a = probs_after.max().item() * 100

    n_disabled = len(disabled)
    impact = (
        f"**{n_disabled} head(s) disabled**\n\n"
        f"| | Before | After Surgery |\n"
        f"|:---|:---|:---|\n"
        f"| **Prediction** | {pred_class_b} | {pred_class_a} |\n"
        f"| **Confidence** | {conf_b:.1f}% | {conf_a:.1f}% |\n"
        f"| **Δ Confidence** | — | {conf_a - conf_b:+.1f}% |"
    )

    return preds_before, preds_after, overlay_before, overlay_after, impact


# ---------------------------------------------------------------------------
# 4. Tab 3: Representation Duel
# ---------------------------------------------------------------------------
def run_representation_duel(
    image: Image.Image | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Compare from-scratch ViT-Tiny vs. ImageNet pre-trained ViT-B/16.

    Returns:
        (scratch_overlay, transfer_overlay, scratch_density, transfer_density, status)
    """
    blank = np.zeros((96, 96, 3), dtype=np.uint8)
    if image is None or _model_scratch is None:
        return blank, blank, blank, blank, "⚠️ No image provided."

    from src.experiments.transfer import extract_torchvision_vit_attentions
    from src.utils.rollout import compute_attention_rollout, extract_cls_rollout

    # Ensure transfer model is loaded
    model_t, load_status = load_transfer_model()

    # From-scratch inference (96x96)
    img_96 = _pil_to_tensor(image, _preprocess_scratch)
    with torch.no_grad():
        logits_s, attns_s = _model_scratch(img_96, return_all_attentions=True)

    rollout_s = compute_attention_rollout(attns_s)
    map_s = extract_cls_rollout(rollout_s, patch_size=8, image_size=96).cpu().numpy()

    # Transfer inference (224x224)
    img_224 = _pil_to_tensor(image, _preprocess_transfer)
    with torch.no_grad():
        logits_t, attns_t = extract_torchvision_vit_attentions(model_t, img_224)

    rollout_t = compute_attention_rollout(attns_t)
    map_t = extract_cls_rollout(
        rollout_t, patch_size=16, image_size=224,
    ).cpu().numpy()

    # Build display images
    img_display_96 = _tensor_to_display(img_96)
    img_display_224 = _tensor_to_display(img_224)

    scratch_overlay = _overlay_heatmap(img_display_96, map_s, alpha=0.55)
    transfer_overlay = _overlay_heatmap(img_display_224, map_t, alpha=0.55)

    # Standalone saliency density maps
    cmap = plt.get_cmap("inferno")
    scratch_density = (cmap(map_s)[..., :3] * 255).astype(np.uint8)
    transfer_density = (cmap(map_t)[..., :3] * 255).astype(np.uint8)

    # Predictions
    pred_s = STL10_CLASSES[logits_s.argmax(dim=-1).item()]
    conf_s = torch.softmax(logits_s[0], dim=0).max().item() * 100
    pred_t = STL10_CLASSES[logits_t.argmax(dim=-1).item()]
    conf_t = torch.softmax(logits_t[0], dim=0).max().item() * 100

    status = (
        f"**From-Scratch ViT-Tiny** → {pred_s} ({conf_s:.1f}%)\n\n"
        f"**ImageNet ViT-B/16** → {pred_t} ({conf_t:.1f}%)\n\n"
        f"{load_status}"
    )
    return scratch_overlay, transfer_overlay, scratch_density, transfer_density, status


# ---------------------------------------------------------------------------
# 5. Tab 4: Texture vs. Shape Bias
# ---------------------------------------------------------------------------
def run_domain_gap_demo(
    image: Image.Image | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Apply Sobel edge sketch and compare attention.

    Returns:
        (original_display, photo_attention, sketch_display, sketch_attention, results_text)
    """
    blank = np.zeros((96, 96, 3), dtype=np.uint8)
    if image is None or _model_scratch is None:
        return blank, blank, blank, blank, ""

    from src.experiments.domain_gap import to_edge_sketch
    from src.utils.rollout import compute_attention_rollout, extract_cls_rollout

    img_tensor = _pil_to_tensor(image, _preprocess_scratch)

    # Photo inference
    with torch.no_grad():
        logits_photo, attns_photo = _model_scratch(
            img_tensor, return_all_attentions=True,
        )

    rollout_p = compute_attention_rollout(attns_photo)
    map_p = extract_cls_rollout(rollout_p, patch_size=8, image_size=96).cpu().numpy()

    # Edge sketch
    sketch_tensor = to_edge_sketch(img_tensor, invert=True)

    # Sketch inference
    with torch.no_grad():
        logits_sketch, attns_sketch = _model_scratch(
            sketch_tensor, return_all_attentions=True,
        )

    rollout_sk = compute_attention_rollout(attns_sketch)
    map_sk = extract_cls_rollout(rollout_sk, patch_size=8, image_size=96).cpu().numpy()

    # Display images
    img_display = _tensor_to_display(img_tensor)
    photo_overlay = _overlay_heatmap(img_display, map_p, alpha=0.55)

    # Sketch display (already in [0, 1] range, 3 channels)
    sketch_np = (
        (sketch_tensor[0].permute(1, 2, 0).cpu().numpy() * 255)
        .clip(0, 255)
        .astype(np.uint8)
    )
    sketch_overlay = _overlay_heatmap(sketch_np, map_sk, alpha=0.55)

    # Predictions
    pred_photo = STL10_CLASSES[logits_photo.argmax(dim=-1).item()]
    conf_photo = torch.softmax(logits_photo[0], dim=0).max().item() * 100
    pred_sketch = STL10_CLASSES[logits_sketch.argmax(dim=-1).item()]
    conf_sketch = torch.softmax(logits_sketch[0], dim=0).max().item() * 100

    # SRI for this single image
    sri = (conf_sketch / conf_photo * 100) if conf_photo > 0 else 0.0

    text = (
        f"### Results\n\n"
        f"| | Natural Photo | Edge Sketch (Sobel) |\n"
        f"|:---|:---|:---|\n"
        f"| **Prediction** | {pred_photo} | {pred_sketch} |\n"
        f"| **Confidence** | {conf_photo:.1f}% | {conf_sketch:.1f}% |\n\n"
        f"**Shape Retention (Confidence)**: {sri:.1f}%\n\n"
        f"*If this is low, the model relies heavily on texture and color "
        f"rather than geometric shape.*"
    )

    return img_display, photo_overlay, sketch_np, sketch_overlay, text


# ---------------------------------------------------------------------------
# 6. Tab 5: Model Architecture Card
# ---------------------------------------------------------------------------
def get_model_summary() -> str:
    """Return a Markdown string describing the model architecture."""
    if _model_scratch is None:
        return "⚠️ Model not loaded yet. Click **Load Model** first."

    total, trainable = _model_scratch.get_num_params()

    md = f"""## ViT-Tiny Architecture Summary

| Property | Value |
|:---|:---|
| **Total Parameters** | {total:,} ({total / 1e6:.2f}M) |
| **Trainable Parameters** | {trainable:,} |
| **Depth (Layers)** | {_model_scratch.depth} |
| **Attention Heads** | 3 per layer ({_model_scratch.depth * 3} total) |
| **Embedding Dim** | {_model_scratch.embed_dim} |
| **MLP Hidden Dim** | {_model_scratch.embed_dim * 4} |
| **Patch Size** | 8 x 8 pixels |
| **Image Resolution** | 96 x 96 → 12 x 12 grid (144 patches + [CLS]) |
| **Dataset** | STL-10 (5,000 train / 8,000 test) |
| **Classes** | {', '.join(STL10_CLASSES)} |

---

## Phase 7 Key Findings

### Head Specialization
- **Most local head**: Layer 2, Head 2 (43.0 px) — convolutional-like receptive field
- **Most global head**: Layer 5, Head 2 (52.0 px) — full-image semantic attention

### Head Ablation Surgery
- **Baseline Accuracy**: 62.03%
- **Mission-critical**: L1-H3 (ΔAcc = +6.88%), L3-H1 (ΔAcc = +6.41%)
- **Redundant**: L4-H3 (ΔAcc = 0.00%)
- **Pruning tolerance**: ~20% of heads removable with <2.2% accuracy loss

### Texture vs. Shape Bias
- **Photo Accuracy**: 60.62%
- **Sketch Accuracy**: 16.88%
- **Shape Retention Index**: 27.8%
- The model relies ~72% on texture/color cues
"""
    return md


# ---------------------------------------------------------------------------
# 7. Example images extraction
# ---------------------------------------------------------------------------
def extract_example_images(
    output_dir: str | Path = "examples",
    n_samples: int = 6,
) -> list[str]:
    """Extract a few STL-10 test images to disk for use as Gradio examples.

    Returns list of file paths (strings).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(out_dir.glob("*.png"))
    if len(existing) >= n_samples:
        return [str(p) for p in existing[:n_samples]]

    # Download STL-10 test set and extract samples
    try:
        from torchvision import datasets

        test_set = datasets.STL10(root="data", split="test", download=True)

        # Pick one image per class (first n_samples classes)
        saved_paths: list[str] = []
        seen_classes: set[int] = set()
        for img_pil, label in test_set:
            if label in seen_classes:
                continue
            seen_classes.add(label)
            class_name = STL10_CLASSES[label]
            path = out_dir / f"{class_name}.png"
            img_pil.save(str(path))
            saved_paths.append(str(path))
            if len(saved_paths) >= n_samples:
                break
        return saved_paths
    except Exception as e:
        print(f"Could not extract example images: {e}")
        return []
