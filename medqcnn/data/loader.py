"""
MedMNIST data loaders for MedQCNN.

Provides a unified interface to load MedMNIST benchmark datasets
as PyTorch DataLoaders with standard train/val/test splits.
"""

from __future__ import annotations

from pathlib import Path

import medmnist
from medmnist import INFO
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


def get_medmnist_loaders(
    dataset_name: str = "breastmnist",
    batch_size: int = 16,
    data_dir: str = "data",
    download: bool = True,
    num_workers: int = 0,
    image_size: int = 28,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Load a MedMNIST dataset and return train/val/test DataLoaders.

    Args:
        dataset_name: Name of the MedMNIST dataset (e.g., "breastmnist",
            "pathmnist", "dermamnist", "bloodmnist", "organamnist").
        batch_size: Batch size for all loaders.
        data_dir: Directory to store/load downloaded data.
        download: Whether to download the dataset if not present.
        num_workers: Number of dataloader worker processes.
        image_size: Image size for the dataset (MedMNIST default is 28).

    Returns:
        Tuple of (train_loader, val_loader, test_loader).

    Raises:
        ValueError: If dataset_name is not a valid MedMNIST dataset.
    """
    if dataset_name not in INFO:
        valid = sorted(INFO.keys())
        msg = f"Unknown dataset: '{dataset_name}'. Valid options: {valid}"
        raise ValueError(msg)

    info = INFO[dataset_name]
    data_class = getattr(medmnist, info["python_class"])

    # Standard transforms: convert to tensor, single-channel grayscale
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    data_path = str(Path(data_dir).resolve())

    train_dataset = data_class(
        split="train",
        transform=transform,
        download=download,
        root=data_path,
        size=image_size,
    )
    val_dataset = data_class(
        split="val",
        transform=transform,
        download=download,
        root=data_path,
        size=image_size,
    )
    test_dataset = data_class(
        split="test",
        transform=transform,
        download=download,
        root=data_path,
        size=image_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader


def get_custom_loaders(
    data_dir: str,
    batch_size: int = 16,
    num_workers: int = 0,
    image_size: int = 224,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Load a custom image dataset organized as ImageFolder and return loaders.

    Expects the following directory structure::

        data_dir/
        ├── train/
        │   ├── class_a/   (e.g. "healthy/")
        │   └── class_b/   (e.g. "diseased/")
        ├── val/
        │   └── ...
        └── test/
            └── ...

    Args:
        data_dir: Root directory containing train/, val/, test/ subdirectories.
        batch_size: Batch size for all loaders.
        num_workers: Number of dataloader worker processes.
        image_size: Target image size (images are resized to this).

    Returns:
        Tuple of (train_loader, val_loader, test_loader, label_names).
        ``label_names`` is a sorted list of class names derived from
        the subdirectory names in ``train/``.

    Raises:
        FileNotFoundError: If data_dir or required subdirectories are missing.
    """
    root = Path(data_dir).resolve()
    for split in ("train", "val", "test"):
        split_dir = root / split
        if not split_dir.is_dir():
            msg = (
                f"Missing '{split}/' directory in {root}. "
                f"Expected structure: {root}/train/, {root}/val/, {root}/test/"
            )
            raise FileNotFoundError(msg)

    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    train_dataset = ImageFolder(root=str(root / "train"), transform=transform)
    val_dataset = ImageFolder(root=str(root / "val"), transform=transform)
    test_dataset = ImageFolder(root=str(root / "test"), transform=transform)

    label_names: list[str] = list(train_dataset.classes)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader, label_names
