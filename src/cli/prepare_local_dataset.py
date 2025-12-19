"""
Utility script to prepare a local Food-10 style dataset for training.

Assumes you already have images organized like:

classification_dataset/
  images/
    class_1/
      img1.jpg
      ...
    class_2/
      ...

This script will create:

classification_dataset/
  train/   # training set
  val/     # validation set
  test/    # (optional) held-out test set

By default, it uses an 80/10/10 split (train/val/test) per class.
"""

import random
import shutil
from pathlib import Path

from src.core.config import DATA_DIR, TRAIN_DIR, TEST_DIR, VAL_DIR


def prepare_local_dataset(
    source_subdir: str = "images",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    """
    Split images from `DATA_DIR / source_subdir` into
    `TRAIN_DIR`, `VAL_DIR` and `TEST_DIR`.

    Args:
        source_subdir: Subdirectory inside DATA_DIR that contains class folders.
        train_ratio: Fraction of images per class to use for training.
        val_ratio: Fraction of images per class to use for validation.
        seed: Random seed for reproducible splits.
    """
    source_root = DATA_DIR / source_subdir

    if train_ratio + val_ratio >= 1.0:
        raise ValueError(
            f"train_ratio + val_ratio must be < 1.0 (got {train_ratio + val_ratio})"
        )

    if not source_root.exists():
        raise FileNotFoundError(
            f"Source directory {source_root} does not exist. "
            "Make sure your images are under classification_dataset/images/<class_name>/"
        )

    # Clean/create destination directories
    if TRAIN_DIR.exists():
        shutil.rmtree(TRAIN_DIR)
    if VAL_DIR.exists():
        shutil.rmtree(VAL_DIR)
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    random.seed(seed)

    class_dirs = sorted(
        [d for d in source_root.iterdir() if d.is_dir()],
        key=lambda p: p.name,
    )

    if not class_dirs:
        raise RuntimeError(
            f"No class subfolders found in {source_root}. "
            "Expected structure: classification_dataset/images/<class_name>/"
        )

    for class_dir in class_dirs:
        class_name = class_dir.name
        print(f"Processing class: {class_name}")

        # Collect image files (common extensions)
        image_files = [
            p
            for p in class_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
        ]

        if not image_files:
            print(f"  Warning: no image files found in {class_dir}, skipping.")
            continue

        image_files.sort()
        random.shuffle(image_files)

        n_total = len(image_files)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_files = image_files[:n_train]
        val_files = image_files[n_train : n_train + n_val]
        test_files = image_files[n_train + n_val :]

        # Create class subdirs in train/val/test
        dest_train_class = TRAIN_DIR / class_name
        dest_val_class = VAL_DIR / class_name
        dest_test_class = TEST_DIR / class_name
        dest_train_class.mkdir(parents=True, exist_ok=True)
        dest_val_class.mkdir(parents=True, exist_ok=True)
        dest_test_class.mkdir(parents=True, exist_ok=True)

        # Copy to destinations
        for f in train_files:
            shutil.copy2(f, dest_train_class / f.name)
        for f in val_files:
            shutil.copy2(f, dest_val_class / f.name)
        for f in test_files:
            shutil.copy2(f, dest_test_class / f.name)

        print(
            f"  {len(train_files)} images -> train/{class_name}, "
            f"{len(val_files)} images -> val/{class_name}, "
            f"{len(test_files)} images -> test/{class_name}"
        )

    print("\nDataset split complete!")
    print(f"Train directory: {TRAIN_DIR}")
    print(f"Val directory:   {VAL_DIR}")
    print(f"Test directory:  {TEST_DIR}")


if __name__ == "__main__":
    prepare_local_dataset()


