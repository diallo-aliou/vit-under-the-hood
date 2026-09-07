"""Main entrypoint script for training Vision Transformer on STL-10."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from src.models.vit import vit_tiny
from src.training.dataset import get_stl10_dataloaders
from src.training.scheduler import create_cosine_warmup_scheduler
from src.training.trainer import Trainer


def set_seed(seed: int = 42) -> None:
    """Set global seeds for deterministic training runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Vision Transformer on STL-10")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/vit_tiny_stl10.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on ('cuda' or 'cpu')",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override peak learning rate")

    args = parser.parse_args()

    # 1. Load configuration
    config_path = Path(args.config).resolve()
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Apply CLI overrides if specified
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["lr"] = args.lr

    set_seed(config["training"].get("seed", 42))

    print("=" * 60)
    print("      Vision Transformer (ViT-Tiny) — STL-10 Training")
    print("=" * 60)
    print(f"Device:       {args.device}")
    print(f"Batch size:   {config['training']['batch_size']}")
    print(f"Epochs:       {config['training']['epochs']}")
    print(f"Base LR:      {config['training']['lr']}")
    print(f"Warmup:       {config['training']['warmup_epochs']} epochs")
    print("=" * 60)

    # 2. Prepare DataLoaders
    train_loader, test_loader = get_stl10_dataloaders(
        data_dir=config["data"]["data_dir"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        download=True,
    )

    # 3. Instantiate ViT-Tiny model
    model = vit_tiny(
        num_classes=config["model"]["num_classes"],
        drop_rate=config["model"].get("drop_rate", 0.1),
        attn_drop_rate=config["model"].get("attn_drop_rate", 0.0),
    )

    total_p, trainable_p = model.get_num_params()
    print(f"Model Parameters: {total_p:,} total, {trainable_p:,} trainable")

    # 4. Criterion, Optimizer, and Scheduler
    criterion = nn.CrossEntropyLoss(
        label_smoothing=config["training"].get("label_smoothing", 0.1)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    scheduler = create_cosine_warmup_scheduler(
        optimizer=optimizer,
        warmup_epochs=config["training"]["warmup_epochs"],
        max_epochs=config["training"]["epochs"],
        min_lr=float(config["training"].get("min_lr", 1e-6)),
        base_lr=float(config["training"]["lr"]),
    )

    # 5. Trainer setup
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        checkpoint_dir=config["checkpoints"]["save_dir"],
        clip_grad_norm=config["training"].get("clip_grad_norm", 1.0),
    )

    # 6. Fit model
    trainer.fit(
        epochs=config["training"]["epochs"],
        save_every_epoch=config["checkpoints"].get("save_every_epoch", True),
    )

    # 7. Persist metrics and visual curves
    trainer.save_history(config["logging"].get("history_file", "outputs/history.json"))
    trainer.plot_curves(config["logging"].get("plot_save_dir", "outputs"))
    print(f"Artifacts saved to {config['logging'].get('plot_save_dir', 'outputs')}/")


if __name__ == "__main__":
    main()
