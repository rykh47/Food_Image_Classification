"""
Quick test script: train and test all 3 models with minimal epochs for validation.
"""
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from src.config import CHECKPOINT_DIR, SUPPORTED_MODELS, IMAGE_SIZE, DEVICE as CONFIG_DEVICE
from src.data_utils import get_data_loaders
from src.models import get_model
from inference import predict_image


def test_model(model_name, max_train_batches=10, max_val_batches=5, batch_size=16):
    """Test a single model with short training and inference."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"Testing Model: {model_name.upper()}")
    print(f"{'='*70}")
    print(f"Device: {device}")

    train_loader, val_loader, class_names = get_data_loaders(
        batch_size=batch_size, num_workers=0, pin_memory=False, image_size=IMAGE_SIZE
    )
    print(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")
    print(f"Classes: {len(class_names)} - {class_names}")

    num_classes = len(class_names)
    model = get_model(model_name, num_classes=num_classes, pretrained=False)
    model = model.to(device)
    print(f"Model loaded: {model_name}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # Train phase
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
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    elapsed = time.time() - start
    train_loss = total_loss / (i + 1)
    train_acc = 100.0 * total_correct / total_samples
    print(f"Train (first {i+1} batches): loss={train_loss:.4f}, acc={train_acc:.2f}%, time={elapsed:.2f}s")

    # Validation phase
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

    val_loss_avg = val_loss / (i + 1)
    val_acc = 100.0 * val_correct / val_total
    print(f"Val (first {i+1} batches): loss={val_loss_avg:.4f}, acc={val_acc:.2f}%")

    # Save checkpoint
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / f"{model_name}_best.pth"
    torch.save({
        'epoch': 0,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': None,
        'best_f1': 0.0
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

    # Inference test
    lm = get_model(model_name, num_classes=num_classes, pretrained=False)
    lm = lm.to(device)
    chk = torch.load(checkpoint_path, map_location=device)
    lm.load_state_dict(chk['model_state_dict'])
    lm.eval()

    val_dataset = val_loader.dataset
    sample_paths = [p for p, _ in val_dataset.imgs[:2]] if hasattr(val_dataset, 'imgs') else []
    
    inference_results = []
    for p in sample_paths:
        try:
            res = predict_image(lm, p, class_names, device=device, top_k=min(3, num_classes))
            inference_results.append({
                'image': Path(p).name,
                'predicted': res['predicted_class'],
                'confidence': f"{res['confidence']:.2%}"
            })
            print(f"  Inference: {Path(p).name} → {res['predicted_class']} ({res['confidence']:.2%})")
        except Exception as e:
            print(f"  Inference error: {e}")
            inference_results.append({'image': Path(p).name, 'error': str(e)})

    return {
        'model': model_name,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss_avg,
        'val_acc': val_acc,
        'checkpoint': str(checkpoint_path),
        'inference_samples': inference_results
    }


def main():
    """Test all models sequentially."""
    print("\n" + "="*70)
    print("MULTI-MODEL TEST PIPELINE")
    print("="*70)
    print(f"Models to test: {', '.join(SUPPORTED_MODELS)}")
    print("="*70)

    results = {}
    for model_name in SUPPORTED_MODELS:
        try:
            result = test_model(model_name)
            results[model_name] = result
        except Exception as e:
            print(f"✗ {model_name} failed: {e}")
            results[model_name] = {'error': str(e)}

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for model_name, result in results.items():
        if 'error' in result:
            print(f"✗ {model_name:20s}: FAILED - {result['error']}")
        else:
            print(f"✓ {model_name:20s}: train_acc={result['train_acc']:.2f}%, val_acc={result['val_acc']:.2f}%")
    print("="*70 + "\n")

    # Save results to JSON
    output_file = Path("test_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_file}")

    all_passed = all('error' not in r for r in results.values())
    return 0 if all_passed else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
