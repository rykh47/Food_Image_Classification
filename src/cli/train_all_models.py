"""
Train all supported models sequentially.
"""
import subprocess
import sys
from pathlib import Path
from ..core.config import SUPPORTED_MODELS, CHECKPOINT_DIR, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE

def train_model(model_name, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE):
    """Train a single model and return results."""
    print(f"\n{'='*70}")
    print(f"Training Model: {model_name.upper()}")
    print(f"{'='*70}")
    print(f"Epochs: {epochs}, Batch Size: {batch_size}, LR: {lr}")
    print(f"{'='*70}\n")
    
    # Run the trainer as a module so that package imports (src.core.…) work correctly
    cmd = [
        sys.executable,
        "-m",
        "src.cli.train_food10",
        "--model",
        model_name,
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--lr",
        str(lr),
    ]

    # Set cwd to project root (…/Food_Image_Classification) so `src` is importable
    project_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(cmd, cwd=project_root)
    
    checkpoint_path = CHECKPOINT_DIR / f"{model_name}_best.pth"
    if checkpoint_path.exists():
        print(f"✓ {model_name} checkpoint saved: {checkpoint_path}")
        return True
    else:
        print(f"✗ {model_name} training failed or checkpoint not saved")
        return False


def main():
    """Train all models sequentially."""
    print("\n" + "="*70)
    print("MULTI-MODEL TRAINING PIPELINE")
    print("="*70)
    print(f"Models to train: {', '.join(SUPPORTED_MODELS)}")
    print("="*70 + "\n")
    
    results = {}
    for model_name in SUPPORTED_MODELS:
        success = train_model(model_name)
        results[model_name] = "PASS" if success else "FAIL"
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    for model_name, status in results.items():
        symbol = "✓" if status == "PASS" else "✗"
        print(f"{symbol} {model_name:20s}: {status}")
    print("="*70 + "\n")
    
    all_passed = all(s == "PASS" for s in results.values())
    if all_passed:
        print("All models trained successfully!")
        print("Next: Run inference or deploy with Streamlit app")
    else:
        print("Some models failed. Check logs above for details.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
