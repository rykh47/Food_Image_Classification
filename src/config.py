"""
Configuration file for Food-10 classification project
"""
import os
import torch
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "food-10"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"

# Create directories
CHECKPOINT_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Model configuration
MODEL_NAME = "resnet50"  # Options: "resnet50", "resnet101", "efficientnet_b0", "efficientnet_b3"
NUM_CLASSES = 10
PRETRAINED = True

# Training configuration
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
PIN_MEMORY = True

# Learning rate scheduler
SCHEDULER = "cosine"  # Options: "cosine", "step", "plateau"
SCHEDULER_PARAMS = {
    "cosine": {"T_max": NUM_EPOCHS, "eta_min": 1e-6},
    "step": {"step_size": 10, "gamma": 0.1},
    "plateau": {"mode": "max", "factor": 0.5, "patience": 5}
}

# Data augmentation
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]  # ImageNet normalization
STD = [0.229, 0.224, 0.225]

# Early stopping
EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 7
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

