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

from src.config import MODEL_NAME, NUM_CLASSES, CHECKPOINT_DIR, IMAGE_SIZE, MEAN, STD, DEVICE
from src.models import get_model, load_checkpoint
from src.data_utils import get_data_loaders
from src.gradcam import visualize_gradcam


@st.cache_resource
def load_model():
    """Load model with caching"""
    checkpoint_path = CHECKPOINT_DIR / f"{MODEL_NAME}_best.pth"
    
    if not checkpoint_path.exists():
        st.error(f"Model checkpoint not found: {checkpoint_path}")
        st.info("Please train the model first using train_food10.py")
        return None, None
    
    # Get class names
    _, _, class_names = get_data_loaders(batch_size=1)
    
    # Create and load model
    model = get_model(model_name=MODEL_NAME, num_classes=NUM_CLASSES, pretrained=False)
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
        page_title="Food-101 Classifier",
        page_icon="🍕",
        layout="wide"
    )
    
    st.title("🍕 Food-101 Image Classification")
    st.markdown("Classify food images into 101 categories using deep learning")
    
    # Load model
    with st.spinner("Loading model..."):
        model, class_names = load_model()
    
    if model is None:
        st.stop()
    
    st.sidebar.header("Settings")
    top_k = st.sidebar.slider("Top K predictions", 1, 10, 5)
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

