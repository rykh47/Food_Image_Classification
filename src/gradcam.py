"""
Grad-CAM visualization for model interpretation
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional

from .config import RESULTS_DIR, MEAN, STD


class GradCAM:
    """
    Grad-CAM implementation for CNN visualization
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        """
        Initialize Grad-CAM
        
        Args:
            model: Trained model
            target_layer: Target layer to visualize (e.g., model.layer4[-1])
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Save activation maps"""
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        """Save gradients"""
        self.gradients = grad_output[0]
    
    def generate_cam(
        self,
        input_image: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap
        
        Args:
            input_image: Input image tensor (1, C, H, W)
            target_class: Target class index (None for predicted class)
        
        Returns:
            CAM heatmap as numpy array
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_image)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        loss = output[0, target_class]
        loss.backward()
        
        # Compute CAM
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))  # (C,)
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam / (cam.max() + 1e-8)
        
        return cam.detach().cpu().numpy()


def visualize_gradcam(
    model: torch.nn.Module,
    image_path: str,
    class_names: list,
    device: str = "cpu",
    save_path: Optional[Path] = None
) -> None:
    """
    Visualize Grad-CAM for a single image
    
    Args:
        model: Trained model
        image_path: Path to image
        class_names: List of class names
        device: Device to run on
        save_path: Path to save visualization
    """
    from torchvision import transforms
    from PIL import Image
    
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    original_img = np.array(img)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        pred_class = output.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()
    
    # Get target layer (works for ResNet)
    if hasattr(model, 'layer4'):
        target_layer = model.layer4[-1]
    elif hasattr(model, 'features') and hasattr(model.features, 'denseblock4'):
        # DenseNet
        target_layer = model.features.denseblock4
    else:
        print("Warning: Could not find suitable layer for Grad-CAM")
        return
    
    # Generate CAM
    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate_cam(input_tensor, target_class=pred_class)
    
    # Resize CAM to original image size
    cam_resized = cv2.resize(cam, (original_img.shape[1], original_img.shape[0]))
    cam_resized = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlay = 0.6 * original_img + 0.4 * heatmap
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original_img)
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(heatmap)
    axes[1].set_title('Grad-CAM Heatmap', fontsize=12)
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title(
        f'Predicted: {class_names[pred_class]}\nConfidence: {confidence:.2%}',
        fontsize=12
    )
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.savefig(RESULTS_DIR / f"gradcam_{Path(image_path).stem}.png", 
                   dpi=300, bbox_inches='tight')
    
    plt.close()

