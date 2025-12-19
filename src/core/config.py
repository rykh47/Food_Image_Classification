"""
Configuration file for Food-10 classification project
"""
import os
import torch
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent.parent  # Project root: go up from src/core -> src -> root
DATA_DIR = BASE_DIR / "classification_dataset"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
VAL_DIR = DATA_DIR / "val"
CHECKPOINT_DIR = BASE_DIR / "src" / "checkpoints"
RESULTS_DIR = BASE_DIR / "src" / "results"

# Create directories
CHECKPOINT_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Model configuration
SUPPORTED_MODELS = ["resnet50", "efficientnet_b0", "densenet121"]
MODEL_NAME = "resnet50"  # Options: "resnet50", "efficientnet_b0", "densenet121"
NUM_CLASSES = 10
PRETRAINED = True

# Training configuration
BATCH_SIZE = 64
NUM_EPOCHS = 50  # Increased from 30 for better convergence
LEARNING_RATE = 1e-3  # Increased from 1e-4 for faster learning
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 8
PIN_MEMORY = True

# Learning rate scheduler
SCHEDULER = "plateau"  # Adaptive scheduling for better convergence; Options: "cosine", "step", "plateau"
SCHEDULER_PARAMS = {
    "cosine": {"T_max": NUM_EPOCHS, "eta_min": 1e-6},
    "step": {"step_size": 10, "gamma": 0.1},
    "plateau": {"mode": "max", "factor": 0.5, "patience": 5, "min_lr": 1e-7}
}

# Data augmentation
IMAGE_SIZE = 256  # Increased from 224 for more detail
MEAN = [0.485, 0.456, 0.406]  # ImageNet normalization
STD = [0.229, 0.224, 0.225]

# Early stopping
EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 15  # Increased to allow longer training before stopping
EARLY_STOPPING_METRIC = "f1_macro"

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Kaggle dataset configuration
KAGGLE_DATASET = "anamikachhabra/food-items-classification-dataset-10-classes"

# Evaluation
EVAL_METRICS = ["accuracy", "f1_macro", "f1_per_class"]

# Logging
LOG_INTERVAL = 50
SAVE_BEST_ONLY = True

