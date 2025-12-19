"""
Quick smoke test: run a short training pass and validation to verify GPU and data pipeline.
"""
import time
import torch
import torch.nn as nn
import torch.optim as optim
from src.data_utils import get_data_loaders
from src.models import get_model
from src.config import MODEL_NAME, NUM_CLASSES, PRETRAINED, IMAGE_SIZE


def run_smoke_test(max_train_batches=5, max_val_batches=5, batch_size=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # smaller, deterministic loaders for smoke test
    train_loader, val_loader, class_names = get_data_loaders(
        batch_size=batch_size, num_workers=0, pin_memory=False, image_size=IMAGE_SIZE
    )

    print(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")
    print(f"Classes: {class_names}")

    model = get_model(MODEL_NAME, num_classes=NUM_CLASSES, pretrained=PRETRAINED)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    model.train()
    start = time.time()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for i, (images, labels) in enumerate(train_loader):
        if i >= max_train_batches:
            break
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    elapsed = time.time() - start
    print(f"Train (first {max_train_batches} batches): loss={total_loss/(max_train_batches):.4f}, "
          f"acc={100.0*total_correct/total_samples:.2f}%, time={elapsed:.2f}s")

    # Quick validation
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            if i >= max_val_batches:
                break
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    print(f"Val (first {max_val_batches} batches): loss={val_loss/max_val_batches:.4f}, "
          f"acc={100.0*val_correct/val_total:.2f}%")


if __name__ == '__main__':
    run_smoke_test()
