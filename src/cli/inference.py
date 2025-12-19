"""
Inference script for Food-10 classification
"""
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import argparse
from pathlib import Path
import json

from ..core.config import MODEL_NAME, NUM_CLASSES, CHECKPOINT_DIR, IMAGE_SIZE, MEAN, STD, DEVICE, SUPPORTED_MODELS
from ..core.models import get_model, load_checkpoint
from ..core.data_utils import get_data_loaders


def load_model_for_inference(
    model_name: str = MODEL_NAME,
    checkpoint_path: Path = None,
    device: str = DEVICE
) -> tuple:
    """
    Load trained model for inference
    
    Returns:
        Tuple of (model, class_names)
    """
    if checkpoint_path is None:
        checkpoint_path = CHECKPOINT_DIR / f"{model_name}_best.pth"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Get class names from data
    _, _, class_names = get_data_loaders(batch_size=1)
    
    # Create model
    model = get_model(model_name=model_name, num_classes=NUM_CLASSES, pretrained=False)
    model = model.to(device)
    
    # Load weights
    load_checkpoint(str(checkpoint_path), model, device)
    model.eval()
    
    return model, class_names


def predict_image(
    model: torch.nn.Module,
    image_path: str,
    class_names: list,
    device: str = DEVICE,
    top_k: int = 5
) -> dict:
    """
    Predict food class for a single image
    
    Args:
        model: Trained model
        image_path: Path to image
        class_names: List of class names
        device: Device to run on
        top_k: Number of top predictions to return
    
    Returns:
        Dictionary with predictions
    """
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probs, top_k)
    
    # Format results
    predictions = []
    for i in range(top_k):
        predictions.append({
            'class': class_names[top_indices[0][i].item()],
            'confidence': top_probs[0][i].item()
        })
    
    return {
        'image_path': image_path,
        'predictions': predictions,
        'predicted_class': predictions[0]['class'],
        'confidence': predictions[0]['confidence']
    }


def predict_batch(
    model: torch.nn.Module,
    image_paths: list,
    class_names: list,
    device: str = DEVICE
) -> list:
    """
    Predict food classes for multiple images
    
    Args:
        model: Trained model
        image_paths: List of image paths
        class_names: List of class names
        device: Device to run on
    
    Returns:
        List of prediction dictionaries
    """
    results = []
    for image_path in image_paths:
        try:
            result = predict_image(model, image_path, class_names, device)
            results.append(result)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            results.append({
                'image_path': image_path,
                'error': str(e)
            })
    return results


def main():
    parser = argparse.ArgumentParser(description='Food Image Classification Inference')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--folder', type=str, help='Path to folder with images')
    parser.add_argument('--model', type=str, default=MODEL_NAME, choices=SUPPORTED_MODELS,
                       help=f'Model to use. Options: {", ".join(SUPPORTED_MODELS)}')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top predictions')
    parser.add_argument('--output', type=str, help='Output JSON file path')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model: {args.model}...")
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    model, class_names = load_model_for_inference(
        model_name=args.model,
        checkpoint_path=checkpoint_path,
        device=DEVICE
    )
    print("Model loaded successfully!")
    
    # Get image paths
    image_paths = []
    if args.image:
        image_paths.append(args.image)
    elif args.folder:
        folder = Path(args.folder)
        image_paths.extend(folder.glob("*.jpg"))
        image_paths.extend(folder.glob("*.png"))
        image_paths.extend(folder.glob("*.jpeg"))
    else:
        print("Error: Please provide --image or --folder")
        return
    
    # Predict
    if len(image_paths) == 1:
        print(f"\nPredicting: {image_paths[0]}")
        result = predict_image(model, image_paths[0], class_names, top_k=args.top_k)
        print(f"\nPredicted: {result['predicted_class']} ({result['confidence']:.2%})")
        print(f"\nTop {args.top_k} predictions:")
        for i, pred in enumerate(result['predictions'], 1):
            print(f"  {i}. {pred['class']}: {pred['confidence']:.2%}")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResults saved to {args.output}")
    else:
        print(f"\nPredicting {len(image_paths)} images...")
        results = predict_batch(model, image_paths, class_names)
        
        # Print summary
        correct = sum(1 for r in results if 'error' not in r)
        print(f"Processed {correct}/{len(results)} images successfully")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()

