"""
Exploratory Data Analysis for the Food-10 dataset
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from collections import Counter
from PIL import Image
import pandas as pd
from torchvision.datasets import ImageFolder

from src.config import TRAIN_DIR, TEST_DIR, RESULTS_DIR
from src.data_utils import download_food10, get_class_distribution

RESULTS_DIR.mkdir(exist_ok=True)


def visualize_sample_images(
    dataset: ImageFolder,
    num_samples: int = 16,
    save_path: Path = None
) -> None:
    """
    Visualize sample images from dataset
    
    Args:
        dataset: ImageFolder dataset
        num_samples: Number of samples to visualize
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()
    
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for idx, ax in enumerate(axes):
        img_idx = indices[idx]
        image, label = dataset[img_idx]
        
        # Convert tensor to numpy for visualization
        if hasattr(image, 'numpy'):
            img = image.numpy().transpose(1, 2, 0)
            # Denormalize
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
        else:
            img = np.array(image)
            if img.max() > 1:
                img = img / 255.0
        
        ax.imshow(img)
        ax.set_title(dataset.classes[label], fontsize=8, wrap=True)
        ax.axis('off')
    
    plt.suptitle('Sample Images from Food-10 Dataset', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.savefig(RESULTS_DIR / "sample_images.png", dpi=300, bbox_inches='tight')
    
    plt.close()
    print(f"Sample images saved to {save_path or RESULTS_DIR / 'sample_images.png'}")


def plot_class_distribution(
    train_counts: dict,
    test_counts: dict,
    save_path: Path = None
) -> None:
    """
    Plot class distribution
    
    Args:
        train_counts: Training set class counts
        test_counts: Test set class counts
        save_path: Path to save figure
    """
    classes = sorted(set(list(train_counts.keys()) + list(test_counts.keys())))
    train_values = [train_counts.get(c, 0) for c in classes]
    test_values = [test_counts.get(c, 0) for c in classes]
    
    x = np.arange(len(classes))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.bar(x - width/2, train_values, width, label='Train', alpha=0.8)
    ax.bar(x + width/2, test_values, width, label='Test', alpha=0.8)
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Number of Images', fontsize=12)
    ax.set_title('Class Distribution: Train vs Test', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=90, ha='right', fontsize=6)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.savefig(RESULTS_DIR / "class_distribution.png", dpi=300, bbox_inches='tight')
    
    plt.close()
    print(f"Class distribution plot saved to {save_path or RESULTS_DIR / 'class_distribution.png'}")


def analyze_image_resolutions(
    dataset: ImageFolder,
    save_path: Path = None
) -> None:
    """
    Analyze image resolutions in dataset
    
    Args:
        dataset: ImageFolder dataset
        save_path: Path to save figure
    """
    widths = []
    heights = []
    
    print("Analyzing image resolutions...")
    for i in range(min(1000, len(dataset))):  # Sample first 1000 images
        img_path = dataset.imgs[i][0]
        with Image.open(img_path) as img:
            width, height = img.size
            widths.append(width)
            heights.append(height)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].hist(widths, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0].set_xlabel('Width (pixels)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Image Width Distribution', fontsize=14, fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    axes[1].hist(heights, bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[1].set_xlabel('Height (pixels)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Image Height Distribution', fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.savefig(RESULTS_DIR / "image_resolutions.png", dpi=300, bbox_inches='tight')
    
    plt.close()
    
    print(f"Resolution analysis saved to {save_path or RESULTS_DIR / 'image_resolutions.png'}")
    print(f"Width stats: min={min(widths)}, max={max(widths)}, mean={np.mean(widths):.1f}")
    print(f"Height stats: min={min(heights)}, max={max(heights)}, mean={np.mean(heights):.1f}")


def generate_eda_report() -> None:
    """
    Generate complete EDA report
    """
    print("=" * 60)
    print("Exploratory Data Analysis - Food-10 Dataset")
    print("=" * 60)
    
    # Download dataset if needed
    if not TRAIN_DIR.exists() or not TEST_DIR.exists():
        print("\nDataset not found. Downloading...")
        download_food10()
    
    # Load datasets (without transforms for EDA)
    from torchvision import transforms
    train_dataset = ImageFolder(
        root=str(TRAIN_DIR),
        transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    )
    test_dataset = ImageFolder(
        root=str(TEST_DIR),
        transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    )
    
    print(f"\nDataset Statistics:")
    print(f"  Number of classes: {len(train_dataset.classes)}")
    print(f"  Training images: {len(train_dataset)}")
    print(f"  Test images: {len(test_dataset)}")
    
    # Class distribution
    print("\nComputing class distribution...")
    train_counts = get_class_distribution(train_dataset)
    test_counts = get_class_distribution(test_dataset)
    
    print(f"\nClass Distribution Summary:")
    print(f"  Train - Min: {min(train_counts.values())}, Max: {max(train_counts.values())}, "
          f"Mean: {np.mean(list(train_counts.values())):.1f}")
    print(f"  Test - Min: {min(test_counts.values())}, Max: {max(test_counts.values())}, "
          f"Mean: {np.mean(list(test_counts.values())):.1f}")
    
    # Visualizations
    print("\nGenerating visualizations...")
    visualize_sample_images(train_dataset, save_path=RESULTS_DIR / "sample_images.png")
    plot_class_distribution(train_counts, test_counts, 
                           save_path=RESULTS_DIR / "class_distribution.png")
    analyze_image_resolutions(train_dataset, save_path=RESULTS_DIR / "image_resolutions.png")
    
    # Save statistics to CSV
    stats_df = pd.DataFrame({
        'class': list(train_counts.keys()),
        'train_count': [train_counts[c] for c in train_counts.keys()],
        'test_count': [test_counts.get(c, 0) for c in train_counts.keys()]
    })
    stats_df.to_csv(RESULTS_DIR / "class_statistics.csv", index=False)
    print(f"\nClass statistics saved to {RESULTS_DIR / 'class_statistics.csv'}")
    
    print("\n" + "=" * 60)
    print("EDA Complete! Check the 'results' folder for visualizations.")
    print("=" * 60)


if __name__ == "__main__":
    generate_eda_report()

