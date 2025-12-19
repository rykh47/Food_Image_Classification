"""
Model definitions for Food-10 classification
"""
import torch
import torch.nn as nn
from torchvision import models
from typing import Optional

try:
    import timm
except ModuleNotFoundError:  # pragma: no cover - optional dependency until installed
    timm = None

from .config import NUM_CLASSES, PRETRAINED


def get_model(
    model_name: str = "resnet50",
    num_classes: int = NUM_CLASSES,
    pretrained: bool = PRETRAINED
) -> nn.Module:
    """
    Get pretrained model with custom classifier head
    
    Args:
        model_name: Name of the model architecture
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
    
    Returns:
        Model with custom classifier
    """
    model_name = model_name.lower()
    
    if model_name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    
    elif model_name == "resnet101":
        model = models.resnet101(pretrained=pretrained)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    
    elif model_name == "resnet152":
        model = models.resnet152(pretrained=pretrained)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=pretrained)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
    
    elif model_name == "efficientnet_b3":
        model = models.efficientnet_b3(pretrained=pretrained)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
    
    elif model_name == "efficientnet_b7":
        model = models.efficientnet_b7(pretrained=pretrained)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
    
    elif model_name == "densenet121":
        model = models.densenet121(pretrained=pretrained)
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, num_classes)
    
    elif model_name == "densenet169":
        model = models.densenet169(pretrained=pretrained)
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, num_classes)
    
    elif model_name == "vgg16":
        model = models.vgg16(pretrained=pretrained)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, num_classes)
    
    elif model_name == "vgg19":
        model = models.vgg19(pretrained=pretrained)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, num_classes)

    elif model_name in {"resnetv2_50", "resnetv2_101", "resnetv2_152"}:
        if timm is None:
            raise ImportError(
                "timm is required for ResNetV2 backbones. Install it via `pip install timm`."
            )
        timm_name = model_name.replace("resnetv2_", "resnetv2_")
        model = timm.create_model(
            timm_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )
    
    else:
        raise ValueError(f"Unknown model name: {model_name}. "
                        f"Supported: resnet50, resnet101, resnet152, "
                        f"resnetv2_50, resnetv2_101, resnetv2_152, "
                        f"efficientnet_b0, efficientnet_b3, efficientnet_b7, "
                        f"densenet121, densenet169, vgg16, vgg19")
    
    return model


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    device: str = "cpu"
) -> dict:
    """
    Load model checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        device: Device to load on
    
    Returns:
        Dictionary with checkpoint info (epoch, best_f1, etc.)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return {
        'epoch': checkpoint.get('epoch', 0),
        'best_f1': checkpoint.get('best_f1', 0.0),
        'optimizer_state_dict': checkpoint.get('optimizer_state_dict'),
        'scheduler_state_dict': checkpoint.get('scheduler_state_dict'),
    }

