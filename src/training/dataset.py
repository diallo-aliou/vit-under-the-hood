"""STL-10 Dataset and DataLoader utilities with data augmentations."""

from pathlib import Path

import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision import datasets

STL10_CLASSES: tuple[str, ...] = (
    "airplane",
    "bird",
    "car",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

# Standard ImageNet normalization parameters
IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]


def get_stl10_transforms(
    image_size: int = 96,
) -> tuple[T.Compose, T.Compose]:
    r"""Build data augmentation pipelines for STL-10 training and evaluation.

    Args:
        image_size: Target square image spatial resolution (H=W). Default: 96.

    Returns:
        tuple (train_transform, test_transform).
    """
    train_transform = T.Compose(
        [
            T.RandomCrop(image_size, padding=4, padding_mode="reflect"),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    test_transform = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    return train_transform, test_transform


def get_stl10_dataloaders(
    data_dir: str = "data",
    batch_size: int = 64,
    num_workers: int = 2,
    download: bool = True,
) -> tuple[DataLoader, DataLoader]:
    r"""Construct PyTorch DataLoaders for the labeled STL-10 train and test splits.

    Args:
        data_dir: Local path to download or locate the dataset. Default: "data".
        batch_size: Number of images per training batch. Default: 64.
        num_workers: Subprocesses for data loading. Default: 2.
        download: If True, downloads dataset files if not already present. Default: True.

    Returns:
        tuple (train_loader, test_loader).
    """
    root_path = Path(data_dir).resolve()
    root_path.mkdir(parents=True, exist_ok=True)

    train_tf, test_tf = get_stl10_transforms(image_size=96)

    train_dataset = datasets.STL10(
        root=str(root_path),
        split="train",
        download=download,
        transform=train_tf,
    )

    test_dataset = datasets.STL10(
        root=str(root_path),
        split="test",
        download=download,
        transform=test_tf,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, test_loader
