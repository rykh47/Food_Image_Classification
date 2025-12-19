"""
End-to-end test: short training -> save checkpoint -> run inference on a few test images.
"""
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from src.config import CHECKPOINT_DIR, MODEL_NAME, NUM_CLASSES, PRETRAINED, IMAGE_SIZE, DEVICE
from src.data_utils import get_data_loaders
from src.models import get_model
from inference import predict_image, load_model_for_inference


def run_e2e(max_train_batches=20, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, class_names = get_data_loaders(batch_size=batch_size, num_workers=0, pin_memory=False, image_size=IMAGE_SIZE)
    print(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")

    # Create model with correct number of classes from dataset
    num_classes = len(class_names)
    model = get_model(MODEL_NAME, num_classes=num_classes, pretrained=False)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    model.train()
    total_loss = 0.0
    total_samples = 0
    start = time.time()
    for i, (images, labels) in enumerate(train_loader):
        if i >= max_train_batches:
            break
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_samples += labels.size(0)

    elapsed = time.time() - start
    print(f"Trained {i+1} batches in {elapsed:.2f}s, avg loss={total_loss/(i+1):.4f}")

    # Save checkpoint
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / f"{MODEL_NAME}_best.pth"
    torch.save({
        'epoch': 0,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': None,
        'best_f1': 0.0
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

    # For inference, recreate model with matching num_classes and load the checkpoint
    lm = get_model(MODEL_NAME, num_classes=num_classes, pretrained=False)
    lm = lm.to(device)
    # load weights
    import torch as _torch
    chk = _torch.load(checkpoint_path, map_location=device)
    lm.load_state_dict(chk['model_state_dict'])
    lm.eval()
    print("Loaded model for inference")

    # get a few image paths from val dataset
    val_dataset = val_loader.dataset
    sample_paths = [p for p, _ in val_dataset.imgs[:3]] if hasattr(val_dataset, 'imgs') else []
    if not sample_paths:
        print("No sample image paths found in validation dataset")
        return

    for p in sample_paths:
        print(f"Predicting: {p}")
        res = predict_image(lm, p, class_names, device=device, top_k=min(5, num_classes))
        print(f"  Predicted: {res['predicted_class']} ({res['confidence']:.2%})")


if __name__ == '__main__':
    run_e2e()
