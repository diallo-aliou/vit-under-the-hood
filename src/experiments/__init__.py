"""Experiments package for Vision Transformer analysis and interpretability."""

from src.experiments.ablation import (
    compute_mean_attention_distance,
    compute_patch_distance_matrix,
    evaluate_single_head_ablation,
    plot_ablation_matrix,
    plot_attention_distance_heatmap,
    plot_pruning_retention_curve,
    prune_heads_cumulative,
)
from src.experiments.domain_gap import (
    create_sobel_kernels,
    evaluate_domain_shift,
    plot_photo_vs_sketch_comparison,
    to_edge_sketch,
)
from src.experiments.timelapse import (
    create_progression_strip,
    discover_checkpoints,
    generate_attention_timelapse,
    render_timelapse_frame,
)
from src.experiments.transfer import (
    build_transfer_vit,
    extract_torchvision_vit_attentions,
    get_transfer_transforms,
    plot_scratch_vs_transfer_comparison,
)

__all__ = [
    "build_transfer_vit",
    "compute_mean_attention_distance",
    "compute_patch_distance_matrix",
    "create_progression_strip",
    "create_sobel_kernels",
    "discover_checkpoints",
    "evaluate_domain_shift",
    "evaluate_single_head_ablation",
    "extract_torchvision_vit_attentions",
    "generate_attention_timelapse",
    "get_transfer_transforms",
    "plot_ablation_matrix",
    "plot_attention_distance_heatmap",
    "plot_photo_vs_sketch_comparison",
    "plot_pruning_retention_curve",
    "plot_scratch_vs_transfer_comparison",
    "prune_heads_cumulative",
    "render_timelapse_frame",
    "to_edge_sketch",
]
