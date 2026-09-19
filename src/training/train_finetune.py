"""Script to fine-tune the ImageNet pre-trained ViT-B/16 on STL-10."""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.experiments.transfer import build_transfer_vit, get_transfer_transforms
from src.training.dataset import get_stl10_dataloaders


def main():
    parser = argparse.ArgumentParser(description="Fine-tune ViT-B/16 on STL-10")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="outputs/checkpoints")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    print(f"Running fine-tuning on: {device}")

    # Build model (frozen backbone, trainable head)
    model = build_transfer_vit(num_classes=10, pretrained=True, freeze_backbone=True, device=device)
    
    # Verify trainability
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params} (Head only)")

    # Transforms (ViT-B/16 expects 224x224)
    train_transform, val_transform = get_transfer_transforms(image_size=224)

    # Dataloaders
    train_loader, val_loader = get_stl10_dataloaders(
        batch_size=args.batch_size,
        train_transform=train_transform,
        val_transform=val_transform,
    )

    criterion = nn.CrossEntropyLoss()
    # Optimizer only on trainable params (the head)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    best_val_acc = 0.0

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{100.*correct/total:.2f}%"})

        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        print(f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%")

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        val_loss = val_loss / total
        val_acc = 100. * correct / total
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%\n")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(args.output_dir, "best_transfer_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved to {save_path} with Acc: {val_acc:.2f}%")


if __name__ == "__main__":
    main()
