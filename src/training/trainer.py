"""PyTorch Trainer module with metrics, checkpoints, and visualization."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    r"""Supervised training orchestrator for Vision Transformer.

    Handles training epochs, validation, gradient clipping, checkpoint snapshotting,
    history serialization, and performance curves generation.

    Args:
        model: PyTorch model module to train.
        train_loader: DataLoader providing training samples.
        val_loader: DataLoader providing evaluation samples.
        criterion: Loss function. Default: nn.CrossEntropyLoss(label_smoothing=0.1).
        optimizer: PyTorch optimizer (e.g. AdamW).
        scheduler: Optional learning rate scheduler.
        device: Target execution device ("cuda" or "cpu").
        checkpoint_dir: Directory where model checkpoints will be stored.
        clip_grad_norm: Maximum norm for gradient clipping. Default: 1.0.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler: LambdaLR | None = None,
        device: str | torch.device = "cpu",
        checkpoint_dir: str = "outputs/checkpoints",
        clip_grad_norm: float | None = 1.0,
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.clip_grad_norm = clip_grad_norm

        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": [],
        }
        self.best_val_acc: float = 0.0

    def train_one_epoch(self, epoch: int, total_epochs: int) -> tuple[float, float]:
        r"""Execute one complete training epoch over train_loader."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch:02d}/{total_epochs:02d} [Train]",
            leave=False,
        )

        for images, targets in pbar:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, targets)
            loss.backward()

            if self.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

            self.optimizer.step()

            total_loss += loss.item() * targets.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

            # Update progress bar description
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_loss = total_loss / total if total > 0 else 0.0
        epoch_acc = 100.0 * correct / total if total > 0 else 0.0
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def evaluate(self) -> tuple[float, float]:
        r"""Evaluate the model on val_loader."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, targets in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            logits = self.model(images)
            loss = self.criterion(logits, targets)

            total_loss += loss.item() * targets.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        val_loss = total_loss / total if total > 0 else 0.0
        val_acc = 100.0 * correct / total if total > 0 else 0.0
        return val_loss, val_acc

    def save_checkpoint(
        self,
        epoch: int,
        val_acc: float,
        is_best: bool = False,
        save_snapshot: bool = True,
    ) -> None:
        r"""Save model weights for best validation accuracy and epoch snapshots."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_acc": val_acc,
        }

        # 1. Periodic snapshot for Phase 6 Attention Timelapse
        if save_snapshot:
            snapshot_path = self.checkpoint_dir / f"epoch_{epoch:03d}.pth"
            torch.save(checkpoint, snapshot_path)

        # 2. Best performing model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)

    def fit(
        self,
        epochs: int,
        save_every_epoch: bool = True,
    ) -> dict[str, list[float]]:
        r"""Run full training and validation loop for specified number of epochs."""
        print(f"Starting training on device: {self.device} for {epochs} epochs...")

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_one_epoch(epoch, epochs)
            val_loss, val_acc = self.evaluate()

            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler is not None:
                self.scheduler.step()

            # Record metrics
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)

            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc

            self.save_checkpoint(
                epoch=epoch,
                val_acc=val_acc,
                is_best=is_best,
                save_snapshot=save_every_epoch,
            )

            best_marker = " [*BEST*]" if is_best else ""
            print(
                f"Epoch {epoch:02d}/{epochs:02d} | "
                f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}% (LR: {current_lr:.2e}){best_marker}"
            )

        print(f"Training completed! Best Validation Accuracy: {self.best_val_acc:.2f}%")
        return self.history

    def save_history(self, save_path: str = "outputs/history.json") -> None:
        r"""Serialize training history metrics to disk as JSON."""
        out_path = Path(save_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

    def plot_curves(self, save_dir: str = "outputs") -> None:
        r"""Render and save loss, accuracy, and learning rate curves."""
        out_dir = Path(save_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        epochs_range = range(1, len(self.history["train_loss"]) + 1)

        # 1. Loss & Accuracy curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Loss
        ax1.plot(epochs_range, self.history["train_loss"], label="Train Loss", color="royalblue", lw=2)
        ax1.plot(epochs_range, self.history["val_loss"], label="Val Loss", color="crimson", lw=2)
        ax1.set_title("Cross-Entropy Loss vs. Epochs", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend()

        # Accuracy
        ax2.plot(epochs_range, self.history["train_acc"], label="Train Acc", color="royalblue", lw=2)
        ax2.plot(epochs_range, self.history["val_acc"], label="Val Acc", color="crimson", lw=2)
        ax2.set_title("Top-1 Accuracy (%) vs. Epochs", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(str(out_dir / "training_curves.png"), dpi=150)
        plt.close(fig)
