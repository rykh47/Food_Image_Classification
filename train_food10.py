"""
Main training script for Food-10 classification
"""
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    StepLR,
)
from tqdm import tqdm

from src.config import *
from src.data_utils import get_data_loaders
from src.metrics import evaluate_model, plot_confusion_matrix, plot_per_class_f1
from src.models import get_model, load_checkpoint


class EarlyStopping:
    """Early stopping to stop training when validation metric doesn't improve"""

    def __init__(self, patience=7, metric="f1_macro", mode="max"):
        self.patience = patience
        self.metric = metric
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif (self.mode == "max" and score < self.best_score) or (
            self.mode == "min" and score > self.best_score
        ):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int,
) -> dict:
    """
    Train for one epoch

    Returns:
        Dictionary with training metrics
    """

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Update progress bar
        if (batch_idx + 1) % LOG_INTERVAL == 0:
            pbar.set_postfix(
                {
                    "loss": f"{running_loss/(batch_idx+1):.4f}",
                    "acc": f"{100.0 * correct / total:.2f}%",
                }
            )

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total

    return {"loss": epoch_loss, "accuracy": epoch_acc}


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str,
    class_names: list,
) -> dict:
    """
    Validate model

    Returns:
        Dictionary with validation metrics
    """

    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    from metrics import compute_metrics
    import numpy as np

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    metrics = compute_metrics(all_labels, all_preds, class_names)
    metrics["loss"] = running_loss / len(val_loader)

    return metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    epoch: int,
    best_f1: float,
    checkpoint_path: Path,
) -> None:
    """Save model checkpoint"""

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "best_f1": best_f1,
        },
        checkpoint_path,
    )
    print(f"Checkpoint saved to {checkpoint_path}")


def train():
    """Main training function"""

    print("=" * 60)
    print("Food-10 Classification Training")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print("=" * 60)

    # Set device
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("Warning: CUDA not available, using CPU")

    # Load data
    print("\nLoading data...")
    train_loader, test_loader, class_names = get_data_loaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        image_size=IMAGE_SIZE,
    )
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Number of classes: {len(class_names)}")

    # Create model
    print(f"\nCreating model: {MODEL_NAME}...")
    model = get_model(
        model_name=MODEL_NAME, num_classes=NUM_CLASSES, pretrained=PRETRAINED
    )
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # Learning rate scheduler
    scheduler = None
    if SCHEDULER == "cosine":
        scheduler = CosineAnnealingLR(optimizer, **SCHEDULER_PARAMS["cosine"])
    elif SCHEDULER == "step":
        scheduler = StepLR(optimizer, **SCHEDULER_PARAMS["step"])
    elif SCHEDULER == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, **SCHEDULER_PARAMS["plateau"])

    # Early stopping
    early_stopping = None
    if EARLY_STOPPING:
        early_stopping = EarlyStopping(
            patience=EARLY_STOPPING_PATIENCE, metric=EARLY_STOPPING_METRIC
        )

    # Training history
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1_macro": [],
    }

    best_f1 = 0.0
    start_epoch = 0

    # Resume from checkpoint if exists (optional - disabled by default)
    checkpoint_path = CHECKPOINT_DIR / f"{MODEL_NAME}_best.pth"
    resume_training = False  # Set to True to resume from checkpoint
    if resume_training and checkpoint_path.exists():
        print(f"\nResuming from checkpoint: {checkpoint_path}")
        checkpoint_info = load_checkpoint(str(checkpoint_path), model, device)
        if checkpoint_info.get("optimizer_state_dict"):
            optimizer.load_state_dict(checkpoint_info["optimizer_state_dict"])
        if scheduler and checkpoint_info.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint_info["scheduler_state_dict"])
        start_epoch = checkpoint_info["epoch"] + 1  # Start from next epoch
        best_f1 = checkpoint_info["best_f1"]
        print(f"Resumed from epoch {start_epoch}, best F1: {best_f1:.4f}")

    print("\nStarting training...")
    print("-" * 60)

    start_time = time.time()

    for epoch in range(start_epoch, NUM_EPOCHS):
        epoch_start = time.time()

        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_metrics = validate(model, test_loader, criterion, device, class_names)

        # Update learning rate
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_metrics["f1_macro"])
            else:
                scheduler.step()

        # Update history
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1_macro"].append(val_metrics["f1_macro"])

        # Print epoch summary
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS} ({epoch_time:.2f}s)")
        print(
            f"  Train Loss: {train_metrics['loss']:.4f}, "
            f"Train Acc: {train_metrics['accuracy']:.2f}%"
        )
        print(
            f"  Val Loss: {val_metrics['loss']:.4f}, "
            f"Val Acc: {val_metrics['accuracy']:.2f}%"
        )
        print(f"  Val F1 (Macro): {val_metrics['f1_macro']:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model
        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                best_f1,
                checkpoint_path,
            )
            print(f"  ✓ New best F1: {best_f1:.4f}")

        # Early stopping
        if early_stopping:
            early_stopping(val_metrics["f1_macro"])
            if early_stopping.early_stop:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

        print("-" * 60)

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time / 3600:.2f} hours")
    print(f"Best F1 (Macro): {best_f1:.4f}")

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)

    # Load best model
    checkpoint_info = load_checkpoint(str(checkpoint_path), model, device)
    print(f"Loaded best model (F1: {checkpoint_info['best_f1']:.4f})")

    # Evaluate on test set
    final_metrics = evaluate_model(model, test_loader, device, class_names)

    print("\nFinal Test Results:")
    print(f"  Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"  F1 (Macro): {final_metrics['f1_macro']:.4f}")

    # Save results
    results = {
        "model": MODEL_NAME,
        "best_f1_macro": best_f1,
        "final_accuracy": final_metrics["accuracy"],
        "final_f1_macro": final_metrics["f1_macro"],
        "training_history": history,
        "per_class_f1": final_metrics["f1_per_class"].tolist(),
        "class_names": class_names,
    }

    results_path = RESULTS_DIR / f"{MODEL_NAME}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_confusion_matrix(
        final_metrics["confusion_matrix"],
        class_names,
        save_path=RESULTS_DIR / f"{MODEL_NAME}_confusion_matrix.png",
    )
    plot_per_class_f1(
        final_metrics["f1_per_class"],
        class_names,
        save_path=RESULTS_DIR / f"{MODEL_NAME}_per_class_f1.png",
    )

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training History - Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"], label="Val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Training History - Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(history["val_f1_macro"], label="Val F1 (Macro)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("F1 Score")
    axes[2].set_title("Training History - F1 Score")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / f"{MODEL_NAME}_training_history.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Training history plot saved to {RESULTS_DIR / MODEL_NAME}_training_history.png")
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    train()


