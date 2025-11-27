"""
Data utilities for the Food-10 dataset (Kaggle)
"""
import os
import shutil
from pathlib import Path
from typing import Tuple, Optional

from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from .config import *


def _get_kaggle_api():
    """Authenticate and return a Kaggle API client."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing Kaggle API. Install it via `pip install kaggle` "
            "and place kaggle.json credentials in ~/.kaggle/."
        ) from exc

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:  # pragma: no cover - authentication env-specific
        raise RuntimeError(
            "Kaggle API authentication failed. Ensure KAGGLE_USERNAME/KAGGLE_KEY "
            "environment variables are set or ~/.kaggle/kaggle.json exists."
        ) from exc
    return api


def _locate_split_dirs(root_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """Search extracted contents for train/test folders."""
    for current_root, dirs, _ in os.walk(root_dir):
        if "train" in dirs and "test" in dirs:
            base = Path(current_root)
            return base / "train", base / "test"
    return None, None


def download_food10(
    data_dir: Path = DATA_DIR,
    force_download: bool = False,
    dataset_slug: str = KAGGLE_DATASET,
) -> None:
    """
    Download and prepare the Food-10 dataset from Kaggle.

    Args:
        data_dir: Target directory to store normalized train/test splits.
        force_download: Re-download even if directories exist.
        dataset_slug: Kaggle dataset slug.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    if train_dir.exists() and test_dir.exists() and not force_download:
        print("Food-10 dataset already prepared. Skipping download.")
        return

    api = _get_kaggle_api()
    raw_dir = data_dir.parent / "food10_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading Food-10 dataset from Kaggle...")
    api.dataset_download_files(
        dataset=dataset_slug,
        path=str(raw_dir),
        unzip=True,
        quiet=False,
    )

    train_source, test_source = _locate_split_dirs(raw_dir)
    if train_source is None or test_source is None:
        raise FileNotFoundError(
            "Unable to locate 'train' and 'test' folders after extraction. "
            "Please inspect the downloaded files in "
            f"{raw_dir} and adjust the data_utils.py logic accordingly."
        )

    print("Organizing dataset into PyTorch-friendly structure...")
    if train_dir.exists():
        shutil.rmtree(train_dir)
    if test_dir.exists():
        shutil.rmtree(test_dir)

    shutil.copytree(train_source, train_dir)
    shutil.copytree(test_source, test_dir)

    print("Dataset preparation complete!")


def get_transforms(
    split: str = "train",
    image_size: int = IMAGE_SIZE,
    mean: list = MEAN,
    std: list = STD
) -> transforms.Compose:
    """
    Get data augmentation transforms
    
    Args:
        split: "train" or "test"
        image_size: Target image size
        mean: Normalization mean
        std: Normalization std
    
    Returns:
        Composed transforms
    """
    if split == "train":
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


def get_data_loaders(
    train_dir: Path = TRAIN_DIR,
    test_dir: Path = TEST_DIR,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = PIN_MEMORY,
    image_size: int = IMAGE_SIZE
) -> Tuple[DataLoader, DataLoader, list]:
    """
    Create train and test data loaders
    
    Args:
        train_dir: Training data directory
        test_dir: Test data directory
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory
        image_size: Image size
    
    Returns:
        Tuple of (train_loader, test_loader, class_names)
    """
    # Download dataset if needed
    if not train_dir.exists() or not test_dir.exists():
        print("Dataset not found. Downloading...")
        download_food10()
    
    # Create datasets
    train_dataset = ImageFolder(
        root=str(train_dir),
        transform=get_transforms("train", image_size)
    )
    
    test_dataset = ImageFolder(
        root=str(test_dir),
        transform=get_transforms("test", image_size)
    )
    
    # Get class names
    class_names = train_dataset.classes
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, test_loader, class_names


def get_class_distribution(dataset: ImageFolder) -> dict:
    """
    Get class distribution from dataset
    
    Args:
        dataset: ImageFolder dataset
    
    Returns:
        Dictionary mapping class names to counts
    """
    class_counts = {}
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        class_name = dataset.classes[label]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    return class_counts

