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
