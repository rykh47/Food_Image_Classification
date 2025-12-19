"""
Convert trained model checkpoints (.pth) to TorchScript (.pt) format.
TorchScript models are faster for inference and easier to deploy.

Usage:
    python convert_to_torchscript.py
"""
import torch
import sys
from pathlib import Path
from src.config import CHECKPOINT_DIR, IMAGE_SIZE
from src.models import get_model


def get_num_classes_from_checkpoint(state_dict):
    """Extract num_classes from checkpoint by looking at classifier layer weights."""
    # Search for classification layer weight tensor to get num_classes
    for key in sorted(state_dict.keys()):
        if 'classifier' in key and 'weight' in key:
            return int(state_dict[key].shape[0])
        if 'fc' in key and 'weight' in key and 'num_batches' not in key:
            return int(state_dict[key].shape[0])
        if 'head' in key and 'weight' in key:
            return int(state_dict[key].shape[0])
    return 10  # Fallback default


def convert_checkpoint_to_torchscript(model_name, checkpoint_path, output_dir):
    """Load checkpoint and convert to TorchScript."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nConverting {model_name}...")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Device: {device}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    
    # Get num_classes from checkpoint
    num_classes = get_num_classes_from_checkpoint(state_dict)
    print(f"  Detected num_classes: {num_classes}")
    
    # Create model with matching num_classes and load weights
    model = get_model(model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    
    # Create dummy input for tracing
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    
    # Convert to TorchScript
    try:
        traced_model = torch.jit.trace(model, dummy_input)
        output_path = output_dir / f"{model_name}.pt"
        traced_model.save(str(output_path))
        
        # Verify the saved model
        loaded_model = torch.jit.load(str(output_path))
        with torch.no_grad():
            test_output = loaded_model(dummy_input)
        
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ Saved: {output_path} ({file_size_mb:.2f} MB)")
        print(f"  ✓ Verified: output shape = {test_output.shape}")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"TorchScript conversion started...\n")
    
    # Output directory for .pt files
    output_dir = Path("src/models_torchscript")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # List of models to convert
    models_to_convert = [
        ("resnet50", CHECKPOINT_DIR / "resnet50_best.pth"),
        ("efficientnet_b0", CHECKPOINT_DIR / "efficientnet_b0_best.pth"),
        ("densenet121", CHECKPOINT_DIR / "densenet121_best.pth"),
    ]
    
    results = {}
    for model_name, checkpoint_path in models_to_convert:
        if checkpoint_path.exists():
            success = convert_checkpoint_to_torchscript(model_name, checkpoint_path, output_dir)
            results[model_name] = success
        else:
            print(f"\n⚠️  Checkpoint not found: {checkpoint_path}")
            results[model_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("Conversion Summary:")
    print("="*60)
    for model_name, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        print(f"  {status:12} {model_name}")
    
    print(f"\nTorchScript models saved to: {output_dir}")
    print("\nUsage example:")
    print("  import torch")
    print("  model = torch.jit.load('src/models_torchscript/efficientnet_b0.pt')")
    print("  output = model(input_tensor)")
    
    all_passed = all(results.values())
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
