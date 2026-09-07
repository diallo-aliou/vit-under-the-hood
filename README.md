# vit-under-the-hood

A PyTorch implementation of the Vision Transformer (ViT) architecture built from scratch, along with tools to visualize and inspect what self-attention is learning during training.

---
## Overview

The purpose of this repository is to implement ViT from scratch, compare **from-scratch training vs. transfer learning (fine-tuning)** on STL-10, and run experiments on the learned attention patterns:

- **From-Scratch vs. Transfer Learning**: Training a custom lightweight ViT from scratch and comparing its representations against an ImageNet fine-tuned model.
- **Attention Timelapse**: Extracting attention maps across training epochs to see how the model transitions from random noise to focusing on relevant object regions.
- **Head Specialization & Ablation**: Analyzing whether different attention heads specialize in local vs. global features, and measuring the drop in accuracy when individual heads are disabled.
- **Photo vs. Sketch Generalization**: Testing how well the model handles simple edge sketches compared to natural photos to evaluate its reliance on texture vs. shape.

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
git clone https://github.com/your-username/vit-under-the-hood.git
cd vit-under-the-hood
pip install -r requirements.txt
```

---

## Roadmap

- [x] **Phase 1**: Project setup & architecture design
- [x] **Phase 2**: Patch Embedding & Positional Encodings
- [x] **Phase 3**: Multi-Head Self-Attention from scratch
- [x] **Phase 4**: Full ViT model assembly & validation tests
- [ ] **Phase 5**: Training pipeline & Google Colab acceleration
- [ ] **Phase 6**: Attention Rollout & Timelapse generation
- [ ] **Phase 7**: Head specialization and Domain Gap analysis
- [ ] **Phase 8**: Interactive visualization interface & results write-up
