# Food-10 Image Classification (Transfer Learning with PyTorch)

End-to-end pipeline for classifying food images into 10 categories using transfer learning (ResNet / EfficientNet). The project covers data preparation, EDA, training, evaluation (macro F1), Grad-CAM interpretation, and a Streamlit demo.

---

## 📦 Dataset

- **Source:** Kaggle – [Food items classification dataset (10 classes)](https://www.kaggle.com/datasets/anamikachhabra/food-items-classification-dataset-10-classes)
- **Size:** ~10k images (750 train + 250 test per class)
- **Access:** Requires Kaggle API credentials

### Configure Kaggle API
1. Generate `kaggle.json` from your Kaggle account.
2. Place it in `~/.kaggle/kaggle.json` (Linux/Mac) or `%USERPROFILE%\.kaggle\kaggle.json` (Windows) with `600` permissions.
3. Alternatively, set `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables.

The dataset is automatically downloaded and organized into `data/food-10/train|test` the first time `train_food10.py`, `eda.py`, or `inference.py` runs.

---

## 🗂️ Project Structure

```
Food Image Classification/
├── src/
│   ├── __init__.py
│   ├── config.py          # Paths, hyperparams, Kaggle slug
│   ├── data_utils.py      # Kaggle download + DataLoader helpers
│   ├── models.py          # Transfer-learning backbones
│   ├── metrics.py         # Macro F1 + plotting utilities
│   └── gradcam.py         # Grad-CAM implementation
├── train_food10.py        # Training loop (imports modules from src/)
├── eda.py                 # Dataset exploration workflow
├── inference.py           # CLI inference (single image or folder)
├── app.py                 # Streamlit demo with optional Grad-CAM overlay
├── REPORT.md              # Short write-up template with slots for results
├── requirements.txt       # Python dependencies
├── README.md              # (this file)
├── checkpoints/           # Saved weights (created after training)
├── results/               # Plots, metrics JSON, Grad-CAM outputs
└── data/food-10/          # Train/test folders (created automatically)
```

---

## ⚙️ Environment Setup

```bash
git clone <repo> food10-transfer-learning
cd food10-transfer-learning

python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

> 💡 GPU is highly recommended. Ensure CUDA drivers are installed if you plan to train on GPU.

---

## 📊 Exploratory Data Analysis

Generate sample grids, class distribution, resolution histograms, and a CSV summary:

```bash
python eda.py
```

Outputs saved to `results/`:
- `sample_images.png`
- `class_distribution.png`
- `image_resolutions.png`
- `class_statistics.csv`

---

## 🧠 Training

```bash
python train_food10.py
```

Key configuration knobs (edit `config.py`):

| Variable        | Description                                 | Default          |
|-----------------|---------------------------------------------|------------------|
| `MODEL_NAME`    | Backbone (`resnet50`, `resnetv2_50`, …)     | `"resnet50"`     |
| `NUM_EPOCHS`    | Training epochs                             | `30`             |
| `BATCH_SIZE`    | Batch size                                  | `32`             |
| `LEARNING_RATE` | AdamW learning rate                         | `1e-4`           |
| `SCHEDULER`     | `cosine`, `step`, or `plateau`              | `"cosine"`       |
| `IMAGE_SIZE`    | Input resolution                            | `224`            |

**Supported backbones (`MODEL_NAME`):**  
`resnet50`, `resnet101`, `resnet152`, `resnetv2_50`, `resnetv2_101`, `resnetv2_152`, `efficientnet_b0`, `efficientnet_b3`, `efficientnet_b7`, `densenet121`, `densenet169`, `vgg16`, `vgg19`.

> ResNet‑V2 variants are provided via the [timm](https://github.com/huggingface/pytorch-image-models) library, already listed in `requirements.txt`.

Training artefacts:
- `checkpoints/{MODEL_NAME}_best.pth`
- `results/{MODEL_NAME}_results.json`
- `results/{MODEL_NAME}_training_history.png`
- `results/{MODEL_NAME}_confusion_matrix.png`
- `results/{MODEL_NAME}_per_class_f1.png`

---

## 🔍 Evaluation & Reporting

Macro F1 is the primary metric. After training:

```json
{
  "model": "resnet50",
  "best_f1_macro": 0.87,
  "final_accuracy": 0.88,
  "final_f1_macro": 0.87,
  "per_class_f1": [...],
  "class_names": [...]
}
```

Use `REPORT.md` to document:
- Approach & architecture choices
- Training configuration
- Macro F1, accuracy, confusion matrix
- Observations / error analysis

Attach confusion-matrix and Grad-CAM screenshots to the report.

---

## 🔮 Inference

### CLI

```bash
# Single image
python inference.py --image path/to/image.jpg

# Folder of images + JSON output
python inference.py --folder path/to/images --output predictions.json
```

Arguments:
- `--model` (defaults to `MODEL_NAME` in config)
- `--checkpoint` (custom .pth path)
- `--top_k` (default 5)

### Streamlit Demo

```bash
streamlit run app.py
```

Features:
- Upload image, view top-K predictions
- Display class probabilities via progress bars
- Optional Grad-CAM visualization overlay

---

## 🧪 Data Augmentation & Training Details

- **Augmentations:** Resize → random crop, horizontal flip, ±15° rotation, color jitter, normalization (ImageNet mean/std).
- **Optimizer:** AdamW
- **Schedulers:** Cosine annealing / Step decay / Reduce-on-plateau
- **Early stopping:** Macro F1 patience (configurable)
- **Metrics:** Accuracy, macro F1, per-class F1, confusion matrix

---

## 🧯 Troubleshooting

| Issue | Fix |
|-------|-----|
| Kaggle download fails | Verify Kaggle credentials, run `kaggle datasets list` to confirm authentication |
| CUDA OOM | Reduce `BATCH_SIZE`, pick a lighter model (e.g., `efficientnet_b0`) |
| Slow loading | Increase `NUM_WORKERS`, ensure dataset sits on SSD |
| Poor F1 | Tune LR/scheduler, extend epochs, try stronger backbone |

---

## 📚 Deliverables Checklist

- [x] `train_food10.py` (transfer-learning pipeline)
- [x] `inference.py` / `app.py` for demos
- [x] Kaggle download + augmentation pipeline
- [x] Macro F1 + confusion matrix outputs
- [x] README (this file) with repro steps
- [x] `REPORT.md` template ready for experiment write-up

Run the training, capture metrics/screenshots, and fill `REPORT.md` before submission.

---

Happy experimenting! 🍱


