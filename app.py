"""
Streamlit app for Food-10 classification demo
"""
import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from src.core.config import MODEL_NAME, NUM_CLASSES, CHECKPOINT_DIR, IMAGE_SIZE, MEAN, STD, DEVICE, SUPPORTED_MODELS
from src.core.models import get_model, load_checkpoint
from src.core.data_utils import get_data_loaders
from src.core.gradcam import visualize_gradcam


@st.cache_resource
def load_model(selected_model=MODEL_NAME):
    """Load model with caching"""
    checkpoint_path = CHECKPOINT_DIR / f"{selected_model}_best.pth"
    
    if not checkpoint_path.exists():
        st.error(f"Model checkpoint not found: {checkpoint_path}")
        st.info(f"Please train the model first using: python train_food10.py --model {selected_model}")
        return None, None
    
    # Get class names (be defensive: dataset may be missing in this environment)
    try:
        _, _, class_names = get_data_loaders(batch_size=1)
    except Exception as e:
        st.warning("Dataset not available or failed to load. Attempting to infer class names from `TRAIN_DIR`.")
        try:
            from src.core.config import TRAIN_DIR
            if TRAIN_DIR.exists():
                class_names = [p.name for p in sorted(TRAIN_DIR.iterdir()) if p.is_dir()]
            else:
                class_names = None
        except Exception:
            class_names = None

    if not class_names:
        st.error(
            "Could not determine class names. Ensure the dataset is present under `classification_dataset/train` "
            "or prepare the dataset using the provided scripts."
        )
        return None, None

    # Create and load model (use actual dataset class count)
    num_classes = len(class_names)
    model = get_model(model_name=selected_model, num_classes=num_classes, pretrained=False)
    model = model.to(DEVICE)
    load_checkpoint(str(checkpoint_path), model, DEVICE)
    model.eval()

    return model, class_names


def predict_image(model, image, class_names, top_k=5):
    """Predict food class for image"""
    # Preprocess
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probs, top_k)
    
    predictions = []
    for i in range(top_k):
        predictions.append({
            'class': class_names[top_indices[0][i].item()],
            'confidence': top_probs[0][i].item()
        })
    
    return predictions


def main():
    st.set_page_config(
        page_title="Food Classifier",
        page_icon="🍕",
        layout="wide"
    )
    
    st.title("🍕 Food Image Classification")
    st.markdown("Classify food images using deep learning models")
    
    st.sidebar.header("Settings")
    
    # Model selection dropdown
    selected_model = st.sidebar.selectbox(
        "Select Model",
        options=SUPPORTED_MODELS,
        help="Choose which trained model to use for inference"
    )
    
    # Load model
    with st.spinner(f"Loading {selected_model} model..."):
        model, class_names = load_model(selected_model=selected_model)
    
    if model is None or class_names is None:
        st.stop()
    
    top_k = st.sidebar.slider("Top K predictions", 1, min(10, len(class_names)), 5)
    show_gradcam = st.sidebar.checkbox("Show Grad-CAM visualization", value=False)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a food image",
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Uploaded Image")
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("Predictions")
            
            # Predict
            with st.spinner("Classifying image..."):
                predictions = predict_image(model, image, class_names, top_k=top_k)
            
            # Display top prediction
            top_pred = predictions[0]
            st.success(f"**{top_pred['class'].replace('_', ' ').title()}**")
            st.metric("Confidence", f"{top_pred['confidence']:.2%}")
            
            # Display all predictions
            st.subheader(f"Top {top_k} Predictions")
            for i, pred in enumerate(predictions, 1):
                st.progress(pred['confidence'], text=f"{i}. {pred['class'].replace('_', ' ').title()}: {pred['confidence']:.2%}")
        
        # Grad-CAM visualization
        if show_gradcam:
            st.subheader("Grad-CAM Visualization")
            with st.spinner("Generating Grad-CAM..."):
                # Save uploaded image temporarily
                temp_path = Path("temp_image.jpg")
                image.save(temp_path)
                
                try:
                    visualize_gradcam(
                        model,
                        str(temp_path),
                        class_names,
                        device=DEVICE,
                        save_path=Path("temp_gradcam.png")
                    )
                    
                    if Path("temp_gradcam.png").exists():
                        st.image("temp_gradcam.png", use_container_width=True)
                        temp_path.unlink(missing_ok=True)
                        Path("temp_gradcam.png").unlink(missing_ok=True)
                except Exception as e:
                    st.error(f"Error generating Grad-CAM: {e}")
    
    else:
        st.info("👆 Please upload an image to get started")
        
        # Show sample classes
        st.sidebar.subheader("101 Food Categories")
        st.sidebar.markdown("Examples: " + ", ".join([c.replace('_', ' ').title() for c in class_names[:20]]) + "...")


if __name__ == "__main__":
    main()

