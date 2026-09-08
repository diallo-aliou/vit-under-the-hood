# vit-under-the-hood

A PyTorch implementation of the Vision Transformer (ViT) architecture built from scratch, along with tools to visualize and inspect what self-attention is learning during training.

---
## Overview

The purpose of this repository is to implement ViT from scratch, compare **from-scratch training vs. transfer learning (fine-tuning)** on STL-10, and run experiments on the learned attention patterns:

- **From-Scratch vs. Transfer Learning**: Training a custom lightweight ViT from scratch and comparing its representations against an ImageNet fine-tuned model.
- **Attention Timelapse**: Extracting attention maps across training epochs to see how the model transitions from random noise to focusing on relevant object regions.
- **Head Specialization & Ablation**: Analyzing whether different attention heads specialize in local vs. global features, and measuring the drop in accuracy when individual heads are disabled.
- **Photo vs. Sketch Generalization**: Testing how well the model handles simple edge sketches compared to natural photos to evaluate its reliance on texture vs. shape.

### Visual Attention Emergence Across Training (50 Epochs on STL-10)

![ViT Attention Dynamics Timelapse](outputs/timelapse_progression.png)
*Attention Rollout ([Abnar & Zuidema, 2020](https://arxiv.org/abs/2005.00928)) tracked across 50 training epochs on STL-10. Attention transitions from diffuse edge noise (Epoch 1, 28.3% Val Acc) to sharp, localized semantic focus on the subject (Epoch 50, 60.9% Val Acc).*

---

## Project Structure

```text
├── configs/              # Model and training YAML configurations
├── src/
│   ├── models/           # Patch embedding, Multi-Head Attention, and ViT modules
│   ├── training/         # DataLoaders, trainer loops, and checkpoint hooks
│   ├── experiments/      # Standalone scripts for the 3 attention experiments
│   └── utils/            # Attention rollout algorithms and visualization tools
├── notebooks/            # Exploratory and Google Colab training notebooks
├── tests/                # Unit tests for tensor shapes and attention properties
├── requirements.txt      # Project dependencies
└── README.md
```

---

## Getting Started

### Installation
```bash
git clone https://github.com/diallo-aliou/vit-under-the-hood.git
cd vit-under-the-hood
pip install -r requirements.txt
pip install -e .
```

### Training & Visual Exploration (Google Colab / Local GPU)
```bash
# Train ViT-Tiny on STL-10 (AdamW + Cosine Warmup, 50 epochs)
python train.py --config configs/vit_tiny_stl10.yaml --device cuda

# Interactive Colab walkthrough: notebooks/01_train_stl10_colab.ipynb
```

### Attention Rollout & Dynamics Timelapse
```bash
# Generate 50-epoch attention evolution GIF & progression strip across saved snapshots
python -m src.experiments.timelapse --checkpoints outputs/checkpoints --output outputs/timelapse_attention.gif

# Interactive Rollout & Timelapse notebook: notebooks/02_attention_timelapse.ipynb
```

---

## Roadmap

- [x] **Phase 1**: Project setup & architecture design
- [x] **Phase 2**: Patch Embedding & Positional Encodings
- [x] **Phase 3**: Multi-Head Self-Attention from scratch
- [x] **Phase 4**: Full ViT model assembly & validation tests
- [x] **Phase 5**: Training pipeline, visual exploration & Google Colab acceleration
- [x] **Phase 6**: Attention Rollout & Timelapse generation
- [ ] **Phase 7**: Head specialization and Domain Gap analysis
- [ ] **Phase 8**: Interactive visualization interface & results write-up
