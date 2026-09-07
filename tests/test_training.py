"""Unit tests for dataset loading, LR scheduling, and Trainer execution."""

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from src.models.vit import vit_tiny
from src.training.dataset import get_stl10_transforms
from src.training.scheduler import create_cosine_warmup_scheduler
from src.training.trainer import Trainer


def test_stl10_transforms() -> None:
    """Verify STL-10 train and test transformations output valid normalized tensors."""
    dummy_pil = Image.new("RGB", (120, 120), color=(100, 150, 200))
    train_tf, test_tf = get_stl10_transforms(image_size=96)

    train_out = train_tf(dummy_pil)
    test_out = test_tf(dummy_pil)

    assert isinstance(train_out, torch.Tensor)
    assert train_out.shape == (3, 96, 96)
    assert isinstance(test_out, torch.Tensor)
    assert test_out.shape == (3, 96, 96)


def test_cosine_warmup_scheduler() -> None:
    """Verify learning rate warmup phase and cosine annealing decay phase."""
    model = nn.Linear(10, 2)
    base_lr = 1e-3
    min_lr = 1e-5
    warmup_epochs = 5
    max_epochs = 20

    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr)
    scheduler = create_cosine_warmup_scheduler(
        optimizer=optimizer,
        warmup_epochs=warmup_epochs,
        max_epochs=max_epochs,
        min_lr=min_lr,
        base_lr=base_lr,
    )

    lrs = []
    for _ in range(max_epochs):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    # 1. Warmup: initial LR must be strictly smaller than base_lr
    assert lrs[0] < base_lr
    # 2. Monotonically increasing during warmup
    for i in range(1, warmup_epochs):
        assert lrs[i] > lrs[i - 1]
    # 3. Peak reached around warmup_epochs
    assert abs(lrs[warmup_epochs - 1] - base_lr) < 1e-6
    # 4. Decaying afterwards
    assert lrs[-1] < lrs[warmup_epochs]
    assert lrs[-1] >= min_lr * 0.99


def test_trainer_execution_synthetic(tmp_path: object) -> None:
    """Verify Trainer can run training, evaluation, and checkpoint saving on synthetic data."""
    # Create small synthetic dataset (16 samples, 2 classes)
    x = torch.randn(16, 3, 96, 96)
    y = torch.randint(0, 10, (16,))
    dataset = TensorDataset(x, y)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=4, shuffle=False)

    model = vit_tiny(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=None,
        device="cpu",
        checkpoint_dir=str(tmp_path),
        clip_grad_norm=1.0,
    )

    # Run 1 epoch
    train_loss, train_acc = trainer.train_one_epoch(epoch=1, total_epochs=1)
    val_loss, val_acc = trainer.evaluate()

    assert train_loss > 0.0
    assert 0.0 <= train_acc <= 100.0
    assert val_loss > 0.0
    assert 0.0 <= val_acc <= 100.0

    # Test checkpoint saving
    trainer.save_checkpoint(epoch=1, val_acc=val_acc, is_best=True, save_snapshot=True)
    assert (tmp_path / "epoch_001.pth").exists()
    assert (tmp_path / "best_model.pth").exists()
