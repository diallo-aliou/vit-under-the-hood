"""Experiments package for Vision Transformer analysis and interpretability."""

from src.experiments.timelapse import (
    create_progression_strip,
    discover_checkpoints,
    generate_attention_timelapse,
    render_timelapse_frame,
)

__all__ = [
    "create_progression_strip",
    "discover_checkpoints",
    "generate_attention_timelapse",
    "render_timelapse_frame",
]
