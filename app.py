"""ViT Under the Hood — Interactive Gradio Explorer.

Launch with:
    python app.py

Or from Google Colab:
    demo.launch(share=True)
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from src.app_backend import (
    extract_example_images,
    get_model_summary,
    load_scratch_model,
    predict_per_head,
    predict_with_rollout,
    run_domain_gap_demo,
    run_head_surgery,
    run_representation_duel,
)

# ---------------------------------------------------------------------------
# Load model at startup
# ---------------------------------------------------------------------------
_, STARTUP_STATUS = load_scratch_model()

# Extract example images (downloads STL-10 on first run if needed)
EXAMPLE_PATHS = extract_example_images(output_dir="examples", n_samples=6)
EXAMPLES = [[p] for p in EXAMPLE_PATHS] if EXAMPLE_PATHS else None

# All 24 heads for the checkbox grid
ALL_HEADS = [f"L{layer_num}-H{h}" for layer_num in range(1, 9) for h in range(1, 4)]

# Pre-generated Phase 7 figure paths
OUTPUTS_DIR = Path("outputs")


# ---------------------------------------------------------------------------
# Build Gradio Interface
# ---------------------------------------------------------------------------
def create_app() -> gr.Blocks:
    """Build and return the full Gradio Blocks app."""

    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="orange",
        font=gr.themes.GoogleFont("Inter"),
    )

    with gr.Blocks(
        theme=theme,
        title="ViT Under the Hood",
        css="""
        .header-title { text-align: center; margin-bottom: 0.5em; }
        .sub-header { text-align: center; color: #666; font-size: 0.95em; }
        """,
    ) as demo:
        # ---- Header ----
        gr.Markdown(
            "# 🔬 ViT Under the Hood — Interactive Explorer\n"
            "Explore what a Vision Transformer sees, head by head, layer by layer.",
            elem_classes=["header-title"],
        )
        gr.Markdown(
            f"**Status**: {STARTUP_STATUS}",
            elem_classes=["sub-header"],
        )

        with gr.Tabs():
            # ==============================================================
            # TAB 1: Attention Rollout Explorer
            # ==============================================================
            with gr.TabItem("🔍 Attention Rollout"):
                gr.Markdown(
                    "### Attention Rollout Explorer\n"
                    "Upload an image or pick an example. Adjust the rollout "
                    "parameters and see where the model focuses."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        img_input_1 = gr.Image(
                            label="Upload Image",
                            type="pil",
                            height=250,
                        )
                        discard_slider = gr.Slider(
                            minimum=0.0,
                            maximum=0.9,
                            step=0.05,
                            value=0.0,
                            label="Discard Ratio (noise reduction)",
                        )
                        fusion_radio = gr.Radio(
                            choices=["mean", "max", "min"],
                            value="mean",
                            label="Head Fusion Strategy",
                        )
                        btn_rollout = gr.Button(
                            "🔍 Analyze Attention", variant="primary",
                        )

                    with gr.Column(scale=2):
                        top5_output = gr.Label(
                            label="Top-5 Predictions", num_top_classes=5,
                        )
                        with gr.Row():
                            rollout_img = gr.Image(
                                label="Attention Rollout Overlay",
                                height=250,
                            )
                            raw_img = gr.Image(
                                label="Raw Layer-8 Attention",
                                height=250,
                            )
                            heatmap_img = gr.Image(
                                label="Rollout Heatmap (standalone)",
                                height=250,
                            )

                # Per-head inspector
                gr.Markdown("#### Per-Head Attention Inspector")
                with gr.Row():
                    layer_slider = gr.Slider(
                        minimum=1,
                        maximum=8,
                        step=1,
                        value=1,
                        label="Encoder Layer",
                    )
                    btn_heads = gr.Button("🧠 Show Heads")
                per_head_img = gr.Image(
                    label="Head 1 | Head 2 | Head 3",
                    height=200,
                )

                # Examples
                if EXAMPLES:
                    gr.Examples(
                        examples=EXAMPLES,
                        inputs=[img_input_1],
                        label="STL-10 Example Images",
                    )

                # Event handlers
                btn_rollout.click(
                    fn=predict_with_rollout,
                    inputs=[img_input_1, discard_slider, fusion_radio],
                    outputs=[top5_output, rollout_img, raw_img, heatmap_img],
                )

                def _per_head_wrapper(image, layer):
                    return predict_per_head(image, int(layer) - 1)

                btn_heads.click(
                    fn=_per_head_wrapper,
                    inputs=[img_input_1, layer_slider],
                    outputs=[per_head_img],
                )

            # ==============================================================
            # TAB 2: Head Surgery Lab
            # ==============================================================
            with gr.TabItem("🔬 Head Surgery Lab"):
                gr.Markdown(
                    "### Interactive Head Ablation Surgery\n"
                    "Select which attention heads to disable and observe "
                    "how the model's prediction and attention change."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        img_input_2 = gr.Image(
                            label="Upload Image",
                            type="pil",
                            height=250,
                        )
                        surgery_checkboxes = gr.CheckboxGroup(
                            choices=ALL_HEADS,
                            label="Disable Heads (e.g. L1-H3, L3-H1)",
                            info="Select heads to ablate. Format: Layer-Head",
                        )
                        btn_surgery = gr.Button(
                            "⚡ Run Surgery", variant="primary",
                        )

                    with gr.Column(scale=2):
                        impact_md = gr.Markdown(label="Impact Summary")
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("**Before Surgery**")
                                preds_before = gr.Label(
                                    label="Predictions (Before)",
                                    num_top_classes=5,
                                )
                                overlay_before = gr.Image(
                                    label="Attention (Before)",
                                    height=200,
                                )
                            with gr.Column():
                                gr.Markdown("**After Surgery**")
                                preds_after = gr.Label(
                                    label="Predictions (After)",
                                    num_top_classes=5,
                                )
                                overlay_after = gr.Image(
                                    label="Attention (After)",
                                    height=200,
                                )

                if EXAMPLES:
                    gr.Examples(
                        examples=EXAMPLES,
                        inputs=[img_input_2],
                        label="STL-10 Example Images",
                    )

                btn_surgery.click(
                    fn=run_head_surgery,
                    inputs=[img_input_2, surgery_checkboxes],
                    outputs=[
                        preds_before,
                        preds_after,
                        overlay_before,
                        overlay_after,
                        impact_md,
                    ],
                )

            # ==============================================================
            # TAB 3: Representation Duel
            # ==============================================================
            with gr.TabItem("⚔️ Representation Duel"):
                gr.Markdown(
                    "### From-Scratch ViT-Tiny vs. ImageNet Pre-trained ViT-B/16\n"
                    "Compare how a small model trained from scratch on STL-10 "
                    "attends vs. a large pre-trained model.\n\n"
                    "> **Note**: The ImageNet ViT-B/16 (~330 MB) will be "
                    "downloaded on first use of this tab."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        img_input_3 = gr.Image(
                            label="Upload Image",
                            type="pil",
                            height=250,
                        )
                        btn_duel = gr.Button(
                            "⚔️ Compare Models", variant="primary",
                        )
                        duel_status = gr.Markdown()

                    with gr.Column(scale=2):
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown(
                                    "**ViT-Tiny (From-Scratch)**\n\n"
                                    "3.63M params · 61.1% Acc"
                                )
                                scratch_overlay = gr.Image(
                                    label="Scratch Rollout",
                                    height=220,
                                )
                                scratch_density = gr.Image(
                                    label="Scratch Saliency",
                                    height=220,
                                )
                            with gr.Column():
                                gr.Markdown(
                                    "**ViT-B/16 (ImageNet Pre-trained)**\n\n"
                                    "86M params · ~95% Acc"
                                )
                                transfer_overlay = gr.Image(
                                    label="Transfer Rollout",
                                    height=220,
                                )
                                transfer_density = gr.Image(
                                    label="Transfer Saliency",
                                    height=220,
                                )

                if EXAMPLES:
                    gr.Examples(
                        examples=EXAMPLES,
                        inputs=[img_input_3],
                        label="STL-10 Example Images",
                    )

                btn_duel.click(
                    fn=run_representation_duel,
                    inputs=[img_input_3],
                    outputs=[
                        scratch_overlay,
                        transfer_overlay,
                        scratch_density,
                        transfer_density,
                        duel_status,
                    ],
                )

            # ==============================================================
            # TAB 4: Texture vs. Shape Bias
            # ==============================================================
            with gr.TabItem("🎨 Texture vs. Shape"):
                gr.Markdown(
                    "### Domain Gap: Texture vs. Shape Bias\n"
                    "Strip away texture using Sobel edge detection and see "
                    "how the model's attention and prediction change.\n\n"
                    "A low **Shape Retention Index** means the model relies "
                    "heavily on texture/color rather than geometric contours."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        img_input_4 = gr.Image(
                            label="Upload Image",
                            type="pil",
                            height=250,
                        )
                        btn_bias = gr.Button(
                            "🎨 Analyze Bias", variant="primary",
                        )

                    with gr.Column(scale=2):
                        bias_results = gr.Markdown(label="Results")
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("**Natural Photo**")
                                photo_display = gr.Image(
                                    label="Original",
                                    height=200,
                                )
                                photo_attn = gr.Image(
                                    label="Photo Attention",
                                    height=200,
                                )
                            with gr.Column():
                                gr.Markdown("**Edge Sketch (Sobel)**")
                                sketch_display = gr.Image(
                                    label="Sketch",
                                    height=200,
                                )
                                sketch_attn = gr.Image(
                                    label="Sketch Attention",
                                    height=200,
                                )

                if EXAMPLES:
                    gr.Examples(
                        examples=EXAMPLES,
                        inputs=[img_input_4],
                        label="STL-10 Example Images",
                    )

                btn_bias.click(
                    fn=run_domain_gap_demo,
                    inputs=[img_input_4],
                    outputs=[
                        photo_display,
                        photo_attn,
                        sketch_display,
                        sketch_attn,
                        bias_results,
                    ],
                )

            # ==============================================================
            # TAB 5: Model Architecture Card
            # ==============================================================
            with gr.TabItem("📊 Model Card"):
                gr.Markdown("### Model Architecture & Phase 7 Results")
                gr.Markdown(value=get_model_summary())

                # Show pre-generated Phase 7 figures if they exist
                phase7_figs = [
                    ("Head Attention Distances", "head_attention_distances.png"),
                    ("Head Ablation Matrix", "head_ablation_matrix.png"),
                    ("Pruning Retention Curve", "head_pruning_curve.png"),
                    ("Representation Duel", "from_scratch_vs_transfer_attention.png"),
                    ("Domain Gap Analysis", "photo_vs_sketch_domain_gap.png"),
                ]

                gr.Markdown("### Phase 7 Experimental Figures")
                existing_figs = [
                    (OUTPUTS_DIR / fname, label)
                    for label, fname in phase7_figs
                    if (OUTPUTS_DIR / fname).exists()
                ]
                if existing_figs:
                    gallery_items = [
                        (str(path), label) for path, label in existing_figs
                    ]
                    gr.Gallery(
                        value=gallery_items,
                        label="Phase 7 Results Gallery",
                        columns=3,
                        height=350,
                    )
                else:
                    gr.Markdown(
                        "*No pre-generated figures found in `outputs/`. "
                        "Run notebook 03 on Colab to generate them.*"
                    )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Launch the Gradio app."""
    demo = create_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
