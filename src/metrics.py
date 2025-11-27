"""
Evaluation metrics for Food-101 classification
"""
import torch
import numpy as np
from sklearn.metrics import (
    f1_score, 
    accuracy_score, 
    confusion_matrix, 
    classification_report
)
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from .config import RESULTS_DIR


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str]
) -> Dict:
    """
    Compute evaluation metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    
    Returns:
        Dictionary of metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_per_class = f1_score(y_true, y_pred, average=None)
    
    metrics = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_per_class': f1_per_class,
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'classification_report': classification_report(
            y_true, y_pred, 
            target_names=class_names,
            output_dict=True
        )
    }
    
    return metrics


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Path = None,
    figsize: Tuple[int, int] = (20, 20)
) -> None:
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save figure
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=False,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=90, ha='right', fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_per_class_f1(
    f1_per_class: np.ndarray,
    class_names: List[str],
    save_path: Path = None,
    figsize: Tuple[int, int] = (15, 10)
) -> None:
    """
    Plot per-class F1 scores
    
    Args:
        f1_per_class: F1 scores per class
        class_names: List of class names
        save_path: Path to save figure
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    sorted_indices = np.argsort(f1_per_class)
    sorted_f1 = f1_per_class[sorted_indices]
    sorted_names = [class_names[i] for i in sorted_indices]
    
    colors = ['red' if f1 < 0.5 else 'orange' if f1 < 0.7 else 'green' 
              for f1 in sorted_f1]
    
    plt.barh(range(len(sorted_f1)), sorted_f1, color=colors)
    plt.yticks(range(len(sorted_f1)), sorted_names, fontsize=8)
    plt.xlabel('F1 Score', fontsize=12)
    plt.ylabel('Class', fontsize=12)
    plt.title('Per-Class F1 Scores', fontsize=16, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Per-class F1 plot saved to {save_path}")
    else:
        plt.savefig(RESULTS_DIR / "per_class_f1.png", dpi=300, bbox_inches='tight')
    
    plt.close()


def evaluate_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: str,
    class_names: List[str]
) -> Dict:
    """
    Evaluate model on dataset
    
    Args:
        model: Trained model
        data_loader: Data loader
        device: Device to run on
        class_names: List of class names
    
    Returns:
        Dictionary of metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    metrics = compute_metrics(all_labels, all_preds, class_names)
    return metrics

